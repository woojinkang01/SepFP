# SepFP Master Document

> Current implementation snapshot: 2026-04-28 evening.
> This document describes the implemented SepFP training stack in this checkout, not the older v1 architecture plan.

## 1. Core Task

SepFP is a stem-aware representation learning system for automatic sample identification (ASID). The central claim is:

> ASID evidence is often source-local. A mixture-level embedding can dilute a shared drum, vocal, bass, guitar, piano, or other stem; SepFP therefore learns stem-specific retrieval embeddings.

For each active stem `s`, the model produces:

- `u_s`: source-selective, transformation-preserving evidence map.
- `pred_s`: linear-magnitude VQT separation output.
- `z_s`: L2-normalized retrieval embedding derived from `u_s`.

Stem absence is not predicted during training. Oracle active-stem metadata from the multi-track dataset determines which stems are computed. If stem `s` is absent, there is no `u_s`, `pred_s`, `z_s`, or loss term for that `(sample, stem)` pair.

## 2. Stable Contracts

### Input Domains

The training step builds two related branch tensors:

| Tensor | Shape | Domain | Use |
|---|---:|---|---|
| `BranchContext.x_input` | `(B, 1, 252, 256)` | log-normalized VQT magnitude | model encoder input |
| `BranchContext.x_linear_mag` | `(B, 1, 252, 256)` | linear VQT magnitude | mask carrier for separation output |
| `BranchContext.x_complex` | `(B, 252, 256, 2)` | complex VQT | target construction and metadata-aligned operations |

The model sees `x_input`. The decoder output is applied to `x_linear_mag`.

### Stem Outputs

For each active stem `s`:

| Output | Current shape | Source |
|---|---:|---|
| `u_s` | `(N_s, 192, 63, 64)` | source-query evidence extractor |
| `pred_s` | `(N_s, 1, 252, 256)` | `mask_s * x_linear_mag[idx_s]` |
| `z_s` | `(N_s, 512)` | stem-specific evidence projector from `u_s.detach()` |

`N_s` is the number of active samples for stem `s`, not the batch size.

## 3. Current Architecture

The current implementation rewrites the earlier v1 stack. It does not preserve `StemHeadBank`, `DirectSkipSepDecoder`, `StemProjector`, ResNet50-IBN, or the old `2048 -> 256` head path.

```text
x_input: (B,1,252,256) lognorm VQT magnitude
  -> TFEvidenceEncoder
  -> M: (B,256,63,64)

for each active stem s:
  M + source query q_s
    -> SourceQueryEvidenceExtractor
    -> u_s: (N_s,192,63,64)

  u_s -> LinearMagMaskDecoder
    -> mask_logits_s: (N_s,1,252,256)

current default mask mode:
  mask_s = 2.0 * sigmoid(mask_logits_s)   # independent_capped
  pred_s = mask_s * x_linear_mag[idx_s]

  u_s.detach() -> EvidenceProjector_s
    -> z_s: normalized (N_s,512)
```

### Implemented Modules

| File | Main class | Role |
|---|---|---|
| `src/sepfp/models/encoder.py` | `TFEvidenceEncoder` | dense time-frequency encoder |
| `src/sepfp/models/stem_head.py` | `SourceQueryEvidenceExtractor` | active-stem evidence extraction using learned stem queries and attention |
| `src/sepfp/models/sep_decoder.py` | `LinearMagMaskDecoder` | predicts linear-magnitude mask logits from `u_s` only |
| `src/sepfp/models/projector.py` | `EvidenceProjector` | attention-pools detached `u_s` and returns normalized `z_s` |
| `src/sepfp/models/sepfp_model.py` | `SepFPModel` | wires encoder, evidence extractor, decoder, configurable mask modes, and per-stem projectors |

### Mask Modes

`SepFPModel` supports four mask modes:

- `independent_capped`: current config default; `mask = max_mask * sigmoid(logits)`, with `max_mask=2.0`.
- `active_softmax`: older/default comparison mode; active masks sum to one per time-frequency bin in each sample.
- `independent_sigmoid`: independent `[0, 1]` masks.
- `independent_softplus`: independent nonnegative masks.

The current research direction moved away from `active_softmax` as the default because it forces every time-frequency bin of mixture energy into the known active stems. `independent_capped` is now the configured default, while `active_softmax` remains useful as an ablation and regression check.

## 4. Data Pipeline

SepFP consumes preprocessed multi-track data. It does not load raw MoisesDB directly.

Expected dataset artifacts:

- audio root: `data/moisesdb_16k_mono_cropped`
- full metadata root: `data/moisesdb_meta_cropped`
- current full-training split:
  - train: `data/moisesdb_meta_train208_seed0`
  - val: `data/moisesdb_meta_val32_seed0`

The metadata contract is:

- one `.npy` activation matrix per song
- one same-basename `.txt` filelist per song
- row order in `.npy` matches line order in `.txt`

The approved preprocessing policy is crop-to-shortest after conversion to final 16 kHz mono. This keeps same-position alignment stable across stems in a song.

## 5. Branch Construction

`SepFPLightningModule.shared_step()` builds two branches:

```text
x_ref[i] = stretch_and_crop(VQT(effect_AB(mix_AB[i])))
x_art[i] = crop(VQT(effect_A(mix_A[i]))) + crop(VQT(effect_B(mix_B[(i-1)%B])))
```

Important consequences:

- `x_ref[i]` contains all active stems from song `i`.
- `x_art[i]` contains stems from `A_i` and rolled `B_(i-1)`.
- `x_art[i]` may contain same stem type from two different songs.
- crop positions, stretch factors, active masks, and provenance are tracked in `BranchContext`.

## 6. Separation Target and Loss

Separation targets are generated by applying the same recorded effect parameters to the individual stem audio, then applying the same VQT crop/stretch operation used for the branch input.

Current domain decision:

```text
target_complex_s = sum(cropped_or_stretched VQT complex components for stem s)
target_linear_s = abs(target_complex_s)
pred_linear_s = mask_s * x_linear_mag
L_sep_raw = mean L1(pred_linear_s, target_linear_s)
```

`L_sep_raw` is computed for both art and ref branch outputs after the branch stem batches are merged. The training objective currently uses `lambda_sep=100.0`:

```text
objective_total = 100.0 * L_sep_raw + lambda_asid * L_asid_raw
```

For Case A+B, where the same stem type appears from two provenance sources in `x_art`, the target is the summed complex VQT of both effected stem components before conversion to linear magnitude.

## 7. ASID Loss

`L_asid` is a per-stem multi-positive InfoNCE loss.

Positive pairs are determined by provenance intersection, not by stem name alone:

```text
P_s[art_row, ref_col] = True
iff art_ctx.provenance[art_sample][s] intersects ref_ctx.provenance[ref_sample][s]
```

The loss is averaged over stems that have valid anchors. Anchors with no positives, or with no valid negatives, are skipped.

`z_s` is always derived from `u_s`. Decoder masks, decoder outputs, linear-magnitude carriers, and skip features are not used to compute `z_s`.

## 8. Training Configs

Primary entrypoint:

```bash
conda run -n sepfp python scripts/train_sepfp.py
```

Full overnight training profile for 2x GTX 1080 Ti:

```bash
conda run -n sepfp python scripts/train_sepfp.py data=full trainer=full callbacks=full logger=wandb
```

Current full profile:

- `max_epochs: 100`
- `trainer.devices: 2`
- `trainer.strategy: ddp`
- `trainer.precision: 32-true`
- optimizer param groups:
  - `sep`: `lr=3e-6` for encoder, evidence extractor, and decoder
  - `asid_projectors`: `lr=1e-5` for per-stem projectors
  - `asid_temperature`: `lr=1e-5` for InfoNCE temperature
- `lambda_sep: 100.0`
- `mask_mode: independent_capped`
- `max_mask: 2.0`
- global `batch_size: 8`
- train split: 208 songs, duplicated 8x per epoch
- validation split: 32 songs
- checkpointing:
  - every 10 epochs
  - best checkpoint by `val/objective/asid_term`
  - last checkpoint

Codex sandbox may not see CUDA even when the host shell does. Verify GPU visibility from the actual training shell before launching the full run.

## 9. Validation and Run Status

The current stack has passed focused unit and smoke validation:

```bash
conda run -n sepfp pytest
```

Validated contracts include:

- lognorm input and linear-magnitude carrier are separate tensors
- separation targets are linear magnitude
- `objective_total` uses weighted separation and ASID terms
- active stem outputs have expected shapes
- absent stems are skipped
- `active_softmax` masks sum to one over active stems per sample
- `independent_capped` masks are nonnegative, capped by `max_mask`, and do not force a unit sum
- `z_s` is normalized
- one shared training step returns finite objective, raw `sep_loss`, and raw `asid_loss`

Recent run context:

- `mq31ccx9`: full 100-epoch active-softmax run completed; final summary had `val/loss ~= 3.05`, `val/sep_loss ~= 0.0195`, `val/asid_loss ~= 1.10`.
- `sepfp-0428-3-lr=3e-6`: full 100-epoch active-softmax run completed; final `val/loss` was worse, mainly from higher `val/asid_loss`.
- `sepfp-0428-independent-mask`: current independent-capped full run profile; local W&B output shows it was launched on 2x GTX 1080 Ti, but the observed local summary is incomplete in this checkout.
- `scripts/diagnose_sep_overfit.py`: separation-only diagnostic utility for one-batch and tiny-subset mask-mode comparisons.

Still needed before trusting a long run scientifically:

- tiny overfit on a fixed small subset
- mask visual inspection and scale statistics
- stem-wise retrieval diagnostics and hubness checks
- DDP validation logging cleanup: Lightning currently warns that epoch-level validation logs should use `sync_dist=True`.

## 10. Current Risks

- `independent_capped` can under- or over-assign total mixture energy because masks are no longer constrained to sum to one.
- `active_softmax`, while still supported, forces all mixture magnitude into active stems, including residual/noisy energy.
- Linear magnitude loss has a larger dynamic range than log-domain loss; scale statistics should be watched early.
- The best-checkpoint monitor is currently `val/objective/asid_term`, not a retrieval metric.
- `others` remains heterogeneous and may be difficult to use as a stable retrieval stem.
- Inference-time stem presence detection is not implemented; training uses oracle stem presence.

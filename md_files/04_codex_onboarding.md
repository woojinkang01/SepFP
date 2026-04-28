# SepFP Codex Onboarding Guide

> Purpose: help a new Codex agent understand the current SepFP checkout quickly and avoid stale v1 assumptions.

## Start Here

Read in this order:

1. `md_files/00_master_document.md` — current implemented contract.
2. `src/sepfp/training/module.py` — real training step orchestration.
3. `src/sepfp/data/preprocess.py` — branch construction and lognorm/linear-mag split.
4. `src/sepfp/models/sepfp_model.py` — model forward path.
5. `tests/test_linear_mag_contract.py` and `tests/test_shapes.py` — executable contract checks.
6. `configs/data/full.yaml`, `configs/trainer/full.yaml`, `configs/callbacks/full.yaml`, `configs/logger/wandb.yaml` — full training profile.

Do not start from older v1 assumptions such as `StemHeadBank`, `DirectSkipSepDecoder`, `StemProjector`, ResNet50-IBN, or `2048 -> 256` stem heads. Those were overwritten by the current source-query linear-mask rewrite.

## Current Big Picture

SepFP trains stem-specific ASID embeddings from multi-track data. The key split is:

```text
u_s = source-selective evidence for separation
z_s = retrieval embedding derived from u_s
```

The model uses separation supervision to make `u_s` source-selective, then projects `u_s` into `z_s` for per-stem contrastive retrieval.

## Data and Branches

The dataset returns waveform-level examples. Model-ready tensors are built later in `SepFPLightningModule.shared_step()`.

Training path:

```text
SepFPDataset.__getitem__()
  -> sepfp_collate_fn()
  -> SepFPLightningModule.shared_step()
  -> build_art_branch() / build_ref_branch()
  -> SepFPModel.forward_branch()
  -> SeparationLoss + MultiPositiveInfoNCELoss
```

Important branch tensors:

| Tensor | Shape | Meaning |
|---|---:|---|
| `x_input` | `(B,1,252,256)` | log-normalized VQT magnitude, fed to encoder |
| `x_linear_mag` | `(B,1,252,256)` | linear VQT magnitude, used as mask carrier |
| `x_complex` | `(B,252,256,2)` | complex VQT, used for aligned target construction |

`x_art[i]` combines `A_i` and rolled `B_(i-1)`. Provenance is merged accordingly, so positives are not based on stem name alone.

## Current Model

Implemented forward path:

```text
x_input
  -> TFEvidenceEncoder
  -> M: (B,256,63,64)

active stem s:
  M + learned q_s
    -> SourceQueryEvidenceExtractor
    -> u_s: (N_s,192,63,64)

  u_s -> LinearMagMaskDecoder -> mask_logits_s
  active-softmax over stems in the same sample -> mask_s
  pred_s = mask_s * x_linear_mag[idx_s]

  u_s -> EvidenceProjector -> z_s: normalized (N_s,512)
```

Absence rule:

```text
if stem s is absent:
  no u_s
  no pred_s
  no z_s
  no L_sep or L_asid term for that pair
```

## Losses

### Separation

The separation target is linear magnitude VQT:

```text
target_complex_s = sum(aligned complex VQT components for stem s)
target_linear_s = abs(target_complex_s)
pred_linear_s = mask_s * x_linear_mag
L_sep_raw = L1(pred_linear_s, target_linear_s)
loss = 100.0 * L_sep_raw + lambda_asid * L_asid_raw
```

This is deliberate. The model input remains lognorm; the separation comparison does not.

### ASID

`L_asid` is per-stem multi-positive InfoNCE. Positive masks are built from provenance intersection:

```text
positive iff art provenance tokens for stem s intersect ref provenance tokens for stem s
```

The implementation averages over stems with valid anchors.

## Full Training Profile

Configured for 2x GTX 1080 Ti:

```bash
conda run -n sepfp python scripts/train_sepfp.py data=full trainer=full callbacks=full logger=wandb
```

Key settings:

- train metadata: `data/moisesdb_meta_train208_seed0`
- validation metadata: `data/moisesdb_meta_val32_seed0`
- audio root: `data/moisesdb_16k_mono_cropped`
- `max_epochs: 100`
- `devices: 2`
- `strategy: ddp`
- `lambda_sep: 100.0`
- global `batch_size: 8`
- checkpoint every 10 epochs, best by `val/loss`, and last
- W&B project: `sepfp`
- W&B group: `source-query-linear-mask`

Codex may not see GPU devices even if the host shell does. Check CUDA from the same shell that will launch training.

## Validation Commands

Use these before changing training or model code:

```bash
conda run -n sepfp pytest tests/test_linear_mag_contract.py
conda run -n sepfp pytest tests/test_shapes.py
conda run -n sepfp pytest tests/test_training_step_smoke.py
conda run -n sepfp pytest
```

The full pytest suite checks tensor contracts and finite smoke behavior. It does not prove retrieval quality.

## Common Mistakes

- Do not describe the current encoder as ResNet50-IBN.
- Do not say decoder output is lognorm VQT.
- Do not route `z_s` from decoder features, masks, or skip features.
- Do not compute absent stems and mask them later; absent stems should be skipped.
- Do not treat `val/loss` as a final ASID metric.
- Do not launch long training from Codex if CUDA is not visible in the actual tool environment.

## Where to Extend Next

High-value next work:

- one-batch real-data DDP smoke from host shell
- tiny-overfit profile
- mask visualization and linear magnitude scale logging
- stem-wise retrieval validation
- hubness checks for A/B/A+B positives
- inference-time stem presence strategy

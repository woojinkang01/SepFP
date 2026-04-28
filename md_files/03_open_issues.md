# SepFP Open Issues

> Current status as of 2026-04-28.
> Earlier architecture questions about ResNet50-IBN, `StemHeadBank`, direct skip decoding, and complex-vs-mask decoder selection are resolved by the current source-query linear-mask rewrite.

## Resolved Architecture Decisions

| Former issue | Current decision |
|---|---|
| Encoder output shape | `TFEvidenceEncoder`: `(B,1,252,256) -> (B,256,63,64)` |
| Stem head structure | `SourceQueryEvidenceExtractor` with learned stem queries and attention |
| `u_s` shape | `(N_s,192,63,64)` |
| Decoder type | linear-magnitude VQT mask decoder |
| Decoder bypass policy | decoder receives `u_s` only; no encoder skip features |
| Mask normalization | sample-wise `active_softmax` over active stems |
| Separation target domain | linear magnitude VQT |
| Separation loss | L1 on linear magnitude `pred_s` and `target_s` |
| Projector | `EvidenceProjector`: attention pooling from `u_s` |
| ASID gradient control | projector input detaches `u_s`; `L_asid` updates projector only |
| Training stem presence | oracle active mask; absent stems are skipped |
| Full training split | train208 / val32 metadata split |

## Critical Before Long-Run Interpretation

### C1. Real-Data GPU One-Batch Validation

The full config composes, and unit/smoke tests pass, but the Codex sandbox did not expose CUDA. Before interpreting an overnight run, verify from the actual host training shell:

```bash
conda run -n sepfp python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
conda run -n sepfp python scripts/train_sepfp.py data=full trainer=full callbacks=full logger=csv trainer.max_steps=1 trainer.max_epochs=1
```

Expected outcome:

- dataloader loads train208/val32 split
- DDP initializes on 2 GPUs
- one training step and one validation pass are finite

### C2. Tiny Overfit

Run a fixed small subset before trusting full-training trends.

Success criteria:

- `sep_loss` decreases clearly
- `pred_s` scale approaches target scale
- masks are not permanently uniform
- `z_s` remains normalized
- `asid_loss` is finite and does not dominate immediately
- raw and weighted losses are inspected separately (`*_loss` and `*_loss_weighted`)

### C3. Linear-Magnitude Scale Monitoring

The current decoder/loss uses linear magnitude, not log magnitude. Early runs should log or inspect:

- `x_linear_mag` min/mean/max or percentiles
- `target_linear` min/mean/max
- `pred_linear` min/mean/max
- mask entropy per stem

If `sep_loss` dominates or explodes, diagnose scale first before changing architecture.

Current default weighting:

```text
loss = 100.0 * sep_loss + lambda_asid * asid_loss
```

## Important Research Questions

### I1. Validation Metric

Current checkpoint `best` uses `val/loss`. This is useful for trainability but not sufficient for ASID quality.

Needed:

- stem-wise retrieval validation
- A/B/A+B case breakdown
- hubness and false-positive checks
- per-stem positive-rank statistics

### I2. `active_softmax` Bias

`active_softmax` makes active stem masks sum to one per time-frequency bin. This prevents bypass and encourages competition, but it also assigns all mixture energy to known active stems.

Risk:

- residual/noisy energy is forced into some stem
- separation loss may reward energy partitioning more than source evidence quality

Useful future ablation:

- independent sigmoid masks
- softmax plus residual/noise channel
- entropy regularization

### I3. ASID Gradient Routing

`z_s` must come from `u_s`, but the current implementation intentionally detaches `u_s` at the projector input. Therefore:

- `L_sep` trains encoder, evidence extractor, and decoder.
- `L_asid` trains projector only.
- `L_asid` does not directly update the encoder or source-query evidence extractor.

Future alternatives, if separation becomes stable but retrieval needs more influence:

- allow `L_asid` into evidence extractor but not encoder using a second evidence forward on `features.detach()`
- introduce a scheduled route after a separation-only warmup

### I4. `others` Stem

`others` is heterogeneous. Keep it in the current 6-stem setup for coverage, but inspect it separately.

Possible outcomes:

- useful for separation but weak for retrieval
- high false-positive rate in `z_others`
- better handled as a mask-only or residual category

### I5. Inference-Time Stem Presence

Training uses oracle active masks from multi-track metadata. A real retrieval system needs a presence detector or another routing strategy.

This is intentionally not part of the current training loop.

## Useful Focused Checks

```bash
conda run -n sepfp pytest tests/test_linear_mag_contract.py
conda run -n sepfp pytest tests/test_shapes.py
conda run -n sepfp pytest tests/test_training_step_smoke.py
conda run -n sepfp pytest
```

These tests verify contracts and finite smoke behavior. They do not prove semantic ASID quality.

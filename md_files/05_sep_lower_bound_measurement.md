# SepFP Separation Lower Bound Measurement

> Current status: 2026-04-29.
> This document records the measured lower bound of the current SepFP separation loss on `val32`.
> The current interpretation baseline is the post-padding-fix measurement.

## Purpose

The goal of this measurement is to estimate the lowest raw `sep_loss` that the current magnitude-mask formulation can express before judging model optimization quality.

Current implemented separation path:

```text
carrier C = x_linear_mag
target T_s = target linear magnitude VQT for active stem s
pred_s = mask_s * C
L_sep = mean_active_instances mean_TF |pred_s - T_s|
```

The current default mask mode is:

```text
model.decoder.mask_mode: independent_capped
model.decoder.max_mask: 2.0
mask_s = 2.0 * sigmoid(mask_logits_s)
```

Therefore the relevant oracle lower bound is:

```text
m_s* = clamp(T_s / (C + eps), 0, 2.0)
LB_independent_capped = mean |m_s* C - T_s|
```

This is a representation lower bound, not a learned-model result. It uses the true target and carrier tensors directly.

## Measurement Script

Script:

```text
scripts/measure_sep_lower_bound.py
```

The script runs the same branch and target construction used by training:

```text
SepFPDataset
  -> sepfp_collate_fn
  -> build_art_branch / build_ref_branch
  -> build_sep_targets
  -> oracle lower-bound metrics
```

It does not run `SepFPModel.forward_branch()`, does not train, and does not use W&B.

## Commands Run

Compile check:

```bash
conda run -n sepfp python -m py_compile scripts/measure_sep_lower_bound.py
```

Single-seed full `val32` scan:

```bash
conda run -n sepfp python scripts/measure_sep_lower_bound.py \
  --data full \
  --split val \
  --seeds 0 \
  --batch-size 4 \
  --device cpu \
  --output outputs/sep_lower_bound/val32_seed0.jsonl
```

Three-seed full `val32` scan:

```bash
conda run -n sepfp python scripts/measure_sep_lower_bound.py \
  --data full \
  --split val \
  --seeds 0,1,2 \
  --batch-size 4 \
  --device cpu \
  --output outputs/sep_lower_bound/val32_seeds012.jsonl
```

Generated result files:

```text
outputs/sep_lower_bound/val32_seed0.jsonl
outputs/sep_lower_bound/val32_seeds012.jsonl
```

After fixing ref time-stretch padding metadata, `val32 seed0` was remeasured:

```bash
conda run -n sepfp python scripts/measure_sep_lower_bound.py \
  --data full \
  --split val \
  --seeds 0 \
  --batch-size 4 \
  --device cpu \
  --output outputs/sep_lower_bound/val32_seed0_padfix.jsonl
```

Additional post-fix smoke output:

```text
outputs/sep_lower_bound/padfix_smoke.jsonl
outputs/sep_lower_bound/val32_seed0_padfix.jsonl
```

## Main Results

The table below combines `art` and `ref` using active stem instance counts, matching the averaging unit of `SeparationLoss` most closely.

Post-fix `val32 seed0` is the current reference measurement:

| Metric | Pre-fix Seed 0 | Post-fix Seed 0 | Delta |
|---|---:|---:|---:|
| `lb_independent_capped` | 0.001126 | 0.000275 | -0.000851 |
| `lb_active_softmax_exact` | 0.009494 | 0.007598 | -0.001895 |
| `ratio_oracle_loss` | 0.009488 | 0.007598 | -0.001890 |
| `uniform_loss` | 0.037611 | 0.037460 | -0.000151 |
| `zero_mask_loss` | 0.035019 | 0.035332 | +0.000313 |

The measured raw `sep_loss` representation floor after the fix is therefore approximately:

```text
val32 seed0 LB_independent_capped ~= 0.00028
objective sep term ~= lambda_sep * LB ~= 100 * 0.00028 ~= 0.028
```

The previous three-seed scan below was done before fixing ref padding metadata and should be treated as a stale diagnostic snapshot:

| Metric | Seed 0 | Seed 1 | Seed 2 | Mean +- Std |
|---|---:|---:|---:|---:|
| `lb_independent_capped` | 0.001126 | 0.001057 | 0.000512 | 0.000899 +- 0.000336 |
| `lb_active_softmax_exact` | 0.009494 | 0.009149 | 0.007666 | 0.008770 +- 0.000971 |
| `ratio_oracle_loss` | 0.009488 | 0.009120 | 0.007666 | 0.008758 +- 0.000963 |
| `uniform_loss` | 0.037611 | 0.035946 | 0.034718 | 0.036092 +- 0.001452 |
| `zero_mask_loss` | 0.035019 | 0.033388 | 0.032975 | 0.033794 +- 0.001081 |
| `unit_mask_loss` | 0.084129 | 0.081827 | 0.076912 | 0.080956 +- 0.003686 |

## Branch Breakdown

Post-fix `val32 seed0` branch breakdown:

| Branch | Pre-fix Seed 0 | Post-fix Seed 0 | Delta |
|---|---:|---:|---:|
| `art` | 0.000240 | 0.000240 | +0.000000 |
| `ref` | 0.001844 | 0.000304 | -0.001540 |

The fix is intentionally branch-specific. `art` is effectively unchanged, while `ref` drops sharply. This confirms the original issue was the ref time-stretch target alignment, not the global magnitude-mask objective.

## Stem Breakdown

Post-fix `val32 seed0` stem breakdown:

| Branch | Stem | Count | Pre-fix LB | Post-fix LB | Delta |
|---|---|---:|---:|---:|---:|
| `art` | `bass` | 20 | 0.000430 | 0.000428 | -0.000002 |
| `art` | `drums` | 27 | 0.000265 | 0.000254 | -0.000011 |
| `art` | `guitar` | 21 | 0.000251 | 0.000254 | +0.000003 |
| `art` | `others` | 12 | 0.000102 | 0.000127 | +0.000025 |
| `art` | `piano` | 12 | 0.000181 | 0.000178 | -0.000003 |
| `art` | `vocals` | 23 | 0.000140 | 0.000140 | +0.000001 |
| `ref` | `bass` | 30 | 0.001608 | 0.000488 | -0.001120 |
| `ref` | `drums` | 30 | 0.005600 | 0.000409 | -0.005192 |
| `ref` | `guitar` | 26 | 0.000668 | 0.000303 | -0.000365 |
| `ref` | `others` | 15 | 0.000231 | 0.000138 | -0.000093 |
| `ref` | `piano` | 15 | 0.000526 | 0.000141 | -0.000385 |
| `ref` | `vocals` | 26 | 0.000649 | 0.000162 | -0.000488 |

The largest pre-fix artifact was `ref/drums`, which dropped from `0.005600` to `0.000409`. Future full-training `val/loss_raw/sep/drums` should be compared against the post-fix number, not the stale pre-fix floor.

## Interpretation

`independent_capped` is much less constrained than `active_softmax` for the current target definition. In the corrected `val32 seed0` measurement, the exact active-softmax oracle floor is `0.007598`, while the independent-capped floor is `0.000275`. This supports keeping `independent_capped` as the current default for separation supervision.

The measured floor is far below the naive baselines:

```text
independent capped oracle: ~0.00028
uniform mask baseline:     ~0.03746
zero mask baseline:        ~0.03533
```

So if a trained model remains near `0.03` raw `sep_loss`, it is behaving close to a weak or collapsed baseline, not near the representation limit. With the current default `lambda_sep=100`, a corrected representation-limit scale is about `0.028` in weighted objective units for `val32 seed0`.

The pre-fix `ref` branch showed unstable target/carrier ratio maxima because ref branch time-stretch padding was not replayed when constructing targets. After storing and reusing `pad_left`, the `ref` lower bound dropped from `0.001844` to `0.000304`, and the `ref` `target_carrier_ratio_max` dropped from `162658736.0` to `280.03` in seed 0. Remaining nonzero floor is still expected because bins with small carrier energy and nonzero target energy cannot always be reconstructed under `max_mask=2.0`.

## Recommended Next Checks

1. Rerun the full three-seed `val32` lower-bound scan after the padding fix to replace the stale pre-fix mean/std table.
2. Compare actual full-run `val/loss_raw/sep` and `val/loss_raw/sep/{stem}` against the post-fix branch/stem floors in this document.
3. Run one-batch and tiny-subset overfit diagnostics using `~0.00028` raw `sep_loss` as the current representation-limit scale.
4. If `ref/drums` remains a large training gap after the target-alignment fix, inspect optimization/content imbalance and low-carrier bins before changing model capacity.

## Validation Status

The measurement script was checked with:

```bash
conda run -n sepfp python -m py_compile scripts/measure_sep_lower_bound.py
conda run -n sepfp pytest tests/test_ref_stretch_alignment.py
conda run -n sepfp pytest tests/test_linear_mag_contract.py tests/test_shapes.py tests/test_training_step_smoke.py tests/test_ref_stretch_alignment.py
```

The post-fix measurement was checked with:

```bash
conda run -n sepfp python scripts/measure_sep_lower_bound.py \
  --data full \
  --split val \
  --num-batches 1 \
  --batch-size 2 \
  --max-examples 2 \
  --device cpu \
  --output outputs/sep_lower_bound/padfix_smoke.jsonl

conda run -n sepfp python scripts/measure_sep_lower_bound.py \
  --data full \
  --split val \
  --seeds 0 \
  --batch-size 4 \
  --device cpu \
  --output outputs/sep_lower_bound/val32_seed0_padfix.jsonl
```

The targeted pytest runs passed:

```text
tests/test_ref_stretch_alignment.py: 2 passed
linear-mag/shape/training/ref-stretch suite: 14 passed, 1 Lightning logging warning
```

The warning is from calling `self.log()` in a smoke test without a registered Trainer. It is unrelated to the lower-bound measurement script.

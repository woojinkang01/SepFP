# SepFP Open Design Issues

> Current status: 2026-04-28.
> This document tracks unresolved training-logic and architecture choices. It is not a general TODO list.

## Baseline to Preserve

The current implemented baseline is:

```text
x_input: log-normalized magnitude VQT
x_linear_mag: linear magnitude VQT carrier
u_s: source-query evidence map for active stem s
pred_s: mask_s * x_linear_mag
z_s: EvidenceProjector_s(u_s.detach())
loss = 100.0 * L_sep + lambda_asid(epoch) * L_asid
```

Current default mask mode is `independent_capped`:

```text
mask_s = 2.0 * sigmoid(mask_logits_s)
```

Stem routing is oracle-active during training. If a stem is inactive, the current baseline does not compute `u_s`, `pred_s`, `z_s`, `L_sep`, or `L_asid` for that sample-stem pair.

Any new design should be compared against this baseline with raw losses and objective terms separated. `val/objective/asid_term` is useful for checkpointing projector learning, but it is not a scientific ASID retrieval metric.

## 1. ASID Gradient Routing

### Question

How far should `L_asid` propagate backward?

Current implementation:

```text
L_sep  -> TFEvidenceEncoder + SourceQueryEvidenceExtractor + LinearMagMaskDecoder
L_asid -> per-stem EvidenceProjector_s only
```

Current branch decision:

```text
u_s.detach() -> EvidenceProjector_s -> z_s
```

The model owns one projector per configured stem, but `forward_branch()` must call only the projector for stems returned by `SourceQueryEvidenceExtractor`. Inactive stems should have no `u_s`, no `z_s`, no projector forward call, and no projector gradient in that step.

Three candidate routes:

| Choice | ASID updates | Meaning |
|---|---|---|
| A | `EvidenceProjector_s` only | Separation defines `u_s`; each stem projector learns to read it for retrieval. |
| B | `EvidenceProjector + SourceQueryEvidenceExtractor` | ASID can shape stem-specific evidence extraction, but not the shared encoder. |
| C | `EvidenceProjector + SourceQueryEvidenceExtractor + TFEvidenceEncoder` | ASID can shape the full representation stack. |

### Why It Matters

`L_sep` and `L_asid` ask related but non-identical questions.

- `L_sep` asks whether `u_s` can recover source-local energy for stem `s`.
- `L_asid` asks whether `z_s` preserves sample identity under augmentation and mixture interference.

Good source separation features are not guaranteed to be good retrieval features. Good retrieval features can also become less useful for reconstruction. The routing decision controls how much this conflict is allowed to reshape the shared representation.

### Critical Risks

Choice A is the cleanest control but may bottleneck retrieval. If `L_sep` removes identity-relevant cues before the projector sees them, `L_asid` cannot recover them. Splitting the projector by stem removes cross-stem projector interference, but it does not let ASID reshape `u_s`.

Choice B is the most plausible next experiment. It lets ASID influence the learned source query, attention, and FiLM/gating that create `u_s`, while keeping the shared TF encoder separation-driven. The risk is that `u_s` starts optimizing identity discrimination at the expense of mask quality.

Choice C is highest-risk. It maximizes ASID influence, but failures become hard to diagnose: a change in `sep_loss` or retrieval may come from encoder drift, evidence routing, mask behavior, or loss-scale interaction. It can also encourage shortcut features tied to augmentation/provenance artifacts rather than robust source identity.

### Implementation Note

Simply removing `u.detach()` in the projector implements Choice C, not Choice B. Choice B requires an explicit gradient boundary such as using `features.detach()` for the ASID evidence path, or otherwise separating the ASID route from the separation route.

### Recommended Ablation Order

1. Keep Choice A as the baseline.
2. Test Choice B after adding module-wise gradient norm logging.
3. Test Choice C only if Choice B improves retrieval but appears representation-limited.

### Must-Log Metrics

- `train/loss_raw/sep`, `val/loss_raw/sep`
- `train/loss_raw/asid`, `val/loss_raw/asid`
- `train/objective_total`, `val/objective_total`
- `train/objective/sep_term`, `val/objective/sep_term`
- `train/objective/asid_term`, `val/objective/asid_term`
- `train/loss_raw/sep/{stem}`, `val/loss_raw/sep/{stem}`
- `train/loss_raw/asid/{stem}`, `val/loss_raw/asid/{stem}`
- `train/lr/sep`, `train/lr/asid_projectors`, `train/lr/asid_temperature`
- module-wise grad norms: encoder, evidence extractor, decoder, per-stem projectors
- stem-wise retrieval ranks, not just InfoNCE loss
- A/B/A+B provenance-case breakdown

## 2. Complex VQT Input and Output

### Question

Should SepFP remain magnitude-only for model input and separation loss, or should it learn from complex VQT input/output?

Current baseline:

```text
model input: log-normalized magnitude VQT
decoder output: real-valued mask
carrier: linear magnitude VQT
target: linear magnitude VQT
L_sep: L1(pred_linear_mag, target_linear_mag)
```

Possible complex extension:

```text
model input: complex-aware VQT channels
decoder output: complex mask or complex source estimate
target: complex VQT
L_sep: complex-domain reconstruction loss, optionally plus magnitude loss
```

### Why It Matters

Magnitude masking ignores phase. This is a real limitation because complex addition is not magnitude-additive:

```text
|S1 + S2| != |S1| + |S2|
```

In Case A+B or other overlapping-source cases, a magnitude-only target can underrepresent cancellation and phase interaction. A complex target could provide deeper separation supervision and force `u_s` to encode more precise source evidence.

### Critical Risks

Complex VQT is more sensitive to crop position, time shift, time stretch, and filtering. ASID requires robustness to transformations, so phase-aware learning can overfit to alignment details that do not define sample identity.

Complex ratio masks are also numerically fragile when mixture energy is small. A naive `target / mixture` formulation can create large masks or unstable gradients. Any complex path needs explicit eps handling, bounded parameterization, or direct source-estimate losses.

The final goal is ASID, not source reconstruction quality. A harder complex reconstruction objective may consume representation capacity without improving retrieval.

### Design Choices

| Choice | Input | Output/loss | Risk |
|---|---|---|---|
| A | lognorm magnitude | magnitude mask + magnitude L1 | Stable baseline; ignores phase. |
| B | lognorm magnitude | complex output/loss auxiliary | Tests whether phase-aware supervision helps without exposing encoder input to phase artifacts. |
| C | complex-aware channels | complex output/loss | Strongest separation supervision; highest risk to ASID invariance. |

### Recommended Ablation Order

1. Preserve magnitude-only baseline.
2. Add complex output/loss as an auxiliary branch while keeping magnitude input.
3. Use a hybrid objective before any full replacement:

```text
L_sep = lambda_mag * L_mag + lambda_complex * L_complex
```

4. Try complex-aware encoder input only if auxiliary complex loss improves retrieval-relevant metrics.

### Must-Log Metrics

- magnitude `sep_loss`
- complex reconstruction loss
- mask or complex-mask norm statistics
- active/inactive stem scale statistics
- stem-wise ASID retrieval metrics
- augmentation robustness breakdown
- Case A/B/A+B breakdown

## 3. Inactive-Stem Separation and Silence Supervision

### Question

Should the model compute separation for inactive stems and train them against silence?

Current baseline:

```text
if stem s is inactive:
  no u_s
  no pred_s
  no z_s
  no L_sep
  no L_asid
```

Candidate extension:

```text
active stem:
  L_active_sep = L1(pred_s, target_s)

inactive stem:
  L_silence = leakage(pred_s, zero)
```

### Why It Matters

The current baseline assumes oracle stem activity during training. This is clean for studying source-local evidence, but it creates an inference gap: a real retrieval system will need either a stem-presence detector or a way to suppress absent stems.

Silence supervision can teach the model that a query for an absent stem should produce low energy. This may reduce false-positive retrieval from absent stems and improve mask calibration.

### Critical Risks

Inactive stems are numerous. If every inactive sample-stem pair contributes a zero target, silence loss can dominate training and make total loss look better without improving active separation or ASID.

The model may learn a conservative low-mask solution. This is especially dangerous with `independent_capped`, where lowering all masks can reduce inactive loss but harm active recall.

Inactive `z_s` should not be treated as a normal ASID negative. A missing source is not a different source identity. Putting inactive embeddings into InfoNCE can pollute the retrieval space with absence states.

### Design Choices

| Choice | Inactive `pred_s` | Inactive `z_s` | ASID use | Risk |
|---|---|---|---|---|
| A | not computed | not computed | none | Clean baseline; no absence learning. |
| B | computed with weak silence loss | not computed, or excluded | active only | Most reasonable ablation. |
| C | computed | computed | excluded from ASID | Useful for future presence scoring, but more compute. |
| D | computed | computed | used as negatives | Not recommended. |

### Recommended Form

If tested, inactive supervision should start as a weak auxiliary loss:

```text
L_sep = L_active_sep + lambda_silence * L_inactive_silence
lambda_silence << 1
```

Possible silence losses:

```text
L_inactive_silence = mean(pred_s)
L_inactive_silence = mean(mask_s * x_linear_mag)
L_inactive_silence = mean(mask_s)
```

The energy-leakage form `mean(mask_s * x_linear_mag)` is the closest to the current magnitude-mask design.

### Must-Log Metrics

- active-only `sep_loss`, excluding inactive terms
- inactive `silence_loss`
- active mask mean and inactive mask mean
- inactive energy leakage
- stem-wise active separation loss
- false-positive retrieval from inactive stems
- ASID retrieval metrics with inactive embeddings excluded
- GPU memory and step-time increase

## Cross-Issue Validation Rules

These issues interact. Do not change all three at once.

Recommended sequence:

1. Fix the current baseline reporting first: raw/weighted losses, stem-wise separation, module-wise grad norms, and retrieval metrics.
2. Test ASID gradient routing Choice B before Choice C.
3. Test inactive silence supervision only with ASID still active-stem-only.
4. Test complex loss first as an auxiliary objective before complex input.

Minimum comparison table for every ablation:

| Metric group | Required values |
|---|---|
| Optimization | raw/weighted total, `L_sep`, `L_asid`, gradient norms |
| Separation | active stem loss, inactive leakage if applicable, mask scale |
| Retrieval | stem-wise rank metrics, A/B/A+B breakdown, false positives |
| Robustness | augmentation breakdown, validation stability |
| Cost | VRAM, step time, failure modes |

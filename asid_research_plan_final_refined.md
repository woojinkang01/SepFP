# Research Plan: Source-Aware (Stem-Aware) Representation Learning for Automatic Sample Identification (ASID)

## 1. Problem Definition

Automatic Sample Identification (ASID) aims to detect whether a short audio segment (query) contains material sampled from another track (reference), even under strong transformations such as pitch-shift, time-stretch, mixing, and effects.

This task is fundamentally challenging due to:
- **Partial overlap**: only a subset of stems may be shared
- **Strong transformations**: pitch/time changes, EQ, compression
- **Mixture ambiguity**: multiple sources coexist and interfere

**Key observation**: ASID is not a whole-track similarity problem; it is a **partial, source-level matching problem**.

---

## 2. Core Thesis: Why Stem-Aware Representations are Key

We posit that **stem-aware (source-aware) embeddings are the key inductive bias for ASID**.

> In ASID, similarity is induced by *shared sources*, not by global timbral or stylistic similarity.

Implications:
- Two mixtures can be highly similar if they share **any subset of stems**, even if the rest differs
- Global embeddings tend to **dilute** this signal, especially under heavy mixing
- Robust matching requires **detecting and preserving evidence of shared sources** under transformations

Therefore, the representation should:
1. Expose **per-source evidence** (which sources contribute to similarity)
2. Remain **robust to pitch/time transformations**
3. Support **aggregation across multiple shared sources**

This motivates a **stem-aware multi-view embedding**, rather than a single monolithic embedding.

---

## 3. Position on Disentanglement

We do **not** pursue strict or perfectly factorized disentanglement as an end goal.

Instead, we adopt a pragmatic stance:
- Each head is **conditioned on a target stem** (stem-aware)
- Representations are **biased toward the target source** via reconstruction
- They may still encode **useful correlated context** when it improves ASID

> Goal: **source-selective representations** that highlight a target stem while remaining useful for matching under real-world mixtures.

This avoids two pitfalls:
- Overly strict “stem-pure” constraints that hurt robustness
- Fully entangled embeddings that obscure which sources drive similarity

---

## 4. Proposed Method

### 4.1 Overview

We propose a **shared encoder with stem-specific heads and a two-stage representation**, where the separation objective and the contrastive objective are explicitly decoupled:

```text
Input Mixture x
      │
      ▼
Shared Encoder E
      │
      ▼
Stem-specific Head h_s
      │
      ▼
u_s  (source-aware, effect-sensitive representation)
      │
      ├───────────────────────────────┐
      │                               │
      ▼                               ▼
Separation Branch                Contrastive Branch
------------------              -------------------
Separation Decoder D_s          Invariant Projection Head g_s
      │                               │
      ▼                               ▼
ŝ_s (reconstructed stem)              z_s
(reconstruct raw or transformed)      (source-aware, effect-invariant embedding)
      │                               │
      ▼                               ▼
L_sep                           L_asid (contrastive loss)
```

This architecture explicitly separates:

- A **separation branch**, which enforces source-awareness via reconstruction  
- A **contrastive branch**, which enforces transformation-invariant similarity for ASID  

This decoupling avoids the conflict between reconstruction and invariance objectives.

---

### 4.2 Representation Structure

For each stem s, we define a two-stage representation:

---

#### Stage 1: Source-selective representation (u_s)

The intermediate representation **u_s** is not enforced to be a strictly disentangled representation of the target stem.

Instead, it is encouraged to be a **target-dominant representation**, where:

- Information related to the target stem is **emphasized**
- Contextual information from co-occurring sources is **allowed to remain**

This reflects the nature of ASID, where the target source exists within a mixture and must be detected under interference.

Therefore, u_s can be understood as:

> a **target-biased, transformation-preserving representation** that may include contextual information from related sources.

---

#### Stage 2: Invariant embedding (z_s)

The final embedding **z_s** is optimized to capture source identity in a transformation-invariant manner.

It is designed to:

- Capture **source identity**
- Be invariant to transformations such as:
  - pitch shift  
  - time stretch  
  - audio effects  

Thus, z_s discards variations that are not relevant to sample identification and retains only identity-relevant components.

Therefore, z_s can be understood as:

> a **transformation-invariant embedding** that retains only the identity-relevant components for ASID.

---

#### Summary

- **u_s**: target-biased, transformation-preserving, context-aware  
- **z_s**: transformation-invariant, identity-focused  

This two-stage design separates:

- **“what evidence exists” (u_s)**  
- **“whether it matches under transformation” (z_s)**  

---

## 5. Training Objectives

### 5.1 Separation Loss (Source Bias)

We use a single model to separate **S = 6 stems** (e.g., vocals, drums, bass, guitar, piano, others).

For each stem s, the model produces a stem-specific representation u_s via the corresponding head h_s, and a dedicated separation decoder D_s reconstructs the target stem.

The separation loss is defined per stem as:

```
L_sep = Σ_s L_sep^s = Σ_s || D_s(u_s) - x_s ||
```

**Gradient Routing (Key Design Detail)**:
- The loss term L_sep^s is **only backpropagated through the path corresponding to stem s**:
  - Stem-specific head h_s
  - Separation decoder D_s
  - Shared encoder E (shared across all stems)

- There is **no cross-stem supervision at the head level**:
  - L_sep^s does not directly update h_{s'} for s' ≠ s

This design ensures that:
- Each head h_s learns to extract features **biased toward its target stem**
- The shared encoder E learns **globally useful source-aware features** through aggregated gradients

Importantly, this formulation does **not enforce strict disentanglement**:
- u_s is only required to contain sufficient information to reconstruct x_s
- It may still encode **contextual information from other co-occurring sources** if beneficial for reconstruction and downstream ASID performance

---

### 5.2 Contrastive Loss (Invariance + Alignment)

Contrastive learning is applied on z_s:

- Positive pairs: mixtures sharing common stems (including partial overlap)
- Negative pairs: unrelated mixtures

```
L_asid = InfoNCE(z_s)
```

Effect:
- Enforces **robustness to pitch/time transforms**
- Aligns embeddings when **shared-source evidence** is present

---

### 5.3 Complementary Roles

- **L_sep → where to look** (source-aware bias)
- **L_asid → how to match** (invariance and alignment)

We deliberately do not enforce hard exclusivity across heads; the objective is reliable detection of shared-source evidence.

---

## 6. Similarity Computation (Multi-Source Matching)

We compute similarity by aggregating across stems:

```
sim(x, y) = Σ_s sim(z_s^x, z_s^y)
```

Benefits:
- Multiple shared sources **additively contribute** to similarity
- Robust under **partial overlap**
- Reduces failure modes of single-embedding retrieval

---

## 7. Why This Solves ASID Better

### 7.1 Versus Single Embedding
- Single vectors must encode all sources → **signal dilution**
- Our method preserves **per-source evidence channels**

### 7.2 Versus Pure Contrastive Learning
- Lacks explicit source structure → brittle under mixing
- Our method injects **source-aware inductive bias**

### 7.3 Versus Strict Disentanglement
- Unrealistic independence assumptions
- Our method focuses on **source-selectivity + robustness**, aligned with ASID needs


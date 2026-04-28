# SepFP Contrastive Learning 설계 — Stem-Level Pair 정의 및 Training Step

> **목적**: SepFP의 contrastive learning이 Riou et al.의 곡-단위 contrastive와 어떻게 다른지,
> stem-level positive/negative mask를 어떻게 구성하는지, 전체 training step의 흐름을 정의한다.
> 
> **전제**: Riou의 데이터 파이프라인(sub-mix 생성, VQT, pitch/time augmentation)을 그대로 차용한다.
> 변경 사항은 모델 구조와 loss 계산에 국한된다.

---

## 1. Riou와 SepFP의 핵심 차이

Riou는 하나의 입력에 대해 단일 전역 임베딩 `z ∈ R^{2048}`을 출력한다.
SepFP는 하나의 입력에 대해 **활성 stem별** 임베딩 `{z_s | s ∈ active_stems}`을 출력한다.

따라서 Riou의 곡-단위 positive/negative는 SepFP에서 **stem-단위 positive/negative**로 세분화되어야 한다. 이를 위해 **provenance tracking**(각 stem이 어느 원곡에서 유래했는지 추적)이 필요하다.

---

## 2. Provenance Tracking

### 2.1 데이터 로더 출력 확장

Riou의 데이터 로더는 `(mix_A, mix_B, mix_AB)`를 반환한다.
SepFP에서는 다음을 추가로 반환한다:

```python
return (mix_A, mix_B, mix_AB,
        stem_types_A,       # list[set]: 각 sample의 sub-mix A에 포함된 stem 유형
        stem_types_B,       # list[set]: 각 sample의 sub-mix B에 포함된 stem 유형
        stem_types_AB,      # list[set]: 각 sample의 전체 활성 stem 유형
        individual_stems_A, # dict[stem_type → audio]: A에 포함된 개별 stem (separation target용)
        individual_stems_B,
        individual_stems_AB)
```

### 2.2 Training Step에서의 Provenance 계산

Riou의 `x_art[i] = x_A[i] + x_B.roll(1, 0)[i]` 구성에 따라:

```python
for i in range(B):
    stems_from_song_i      = stem_types_A[i]           # mix_A[i]에서 온 stem
    stems_from_song_prev   = stem_types_B[(i-1) % B]   # mix_B[(i-1)%B]에서 온 stem
```

---

## 3. Stem-Level Positive/Negative Mask

### 3.1 정의

> **z_s(x_art[i]) ↔ z_s(x_ref[j])가 positive이려면**:
> stem s가 x_art[i]에 존재해야 하며, 그 stem s가 x_ref[j]의 원곡(song j)에서 유래해야 한다.

### 3.2 Case 분류

`x_art[i] = mix_A[i] + mix_B[(i-1)%B]`이므로, stem s의 유래는 두 가지:

```
Case A:  s ∈ stem_types_A[i]              → song_i에서 유래
Case B:  s ∈ stem_types_B[(i-1)%B]        → song_{(i-1)%B}에서 유래
Case A+B: 두 곡 모두에서 유래 (동일 유형 중첩)
Case None: x_art[i]에 stem s가 없음       → head 비활성, skip
```

### 3.3 Positive Mask 구성

stem s에 대한 positive mask `P_s` (shape: art_active × ref_active):

```python
def provenance_match(art_idx, ref_idx, stem_s):
    # Case A: stem s가 mix_A[art_idx]에 있고, ref_idx == art_idx
    if stem_s in stem_types_A[art_idx] and ref_idx == art_idx:
        return True
    # Case B: stem s가 mix_B[(art_idx-1)%B]에 있고, ref_idx == (art_idx-1)%B
    rolled_idx = (art_idx - 1) % B
    if stem_s in stem_types_B[rolled_idx] and ref_idx == rolled_idx:
        return True
    return False
```

---

## 4. 구체적 예시

### 4.1 Setup

```
Batch (B=4):

Song 0: active = {vocals, drums, bass}  →  A₀={vocals, drums},  B₀={bass}
Song 1: active = {vocals, drums, guitar} →  A₁={drums},         B₁={vocals, guitar}
Song 2: active = {bass, piano, guitar}   →  A₂={bass},          B₂={piano, guitar}
Song 3: active = {vocals, drums, bass, guitar} → A₃={vocals, bass}, B₃={drums, guitar}
```

### 4.2 x_art 구성 (roll 적용 후)

```
x_art[0] = A₀ + B₃ = {vocals₀, drums₀} + {drums₃, guitar₃}
x_art[1] = A₁ + B₀ = {drums₁}          + {bass₀}
x_art[2] = A₂ + B₁ = {bass₂}           + {vocals₁, guitar₁}
x_art[3] = A₃ + B₂ = {vocals₃, bass₃}  + {piano₂, guitar₂}
```

### 4.3 Positive Mask — Stem = vocals

```
z_vocals(x_art[0]): vocals₀(song 0) ← Case A only
  → positive with x_ref[0] ✓

z_vocals(x_art[1]): vocals 없음 → SKIP
z_vocals(x_art[2]): vocals₁(song 1) ← Case B only
  → positive with x_ref[1] ✓

z_vocals(x_art[3]): vocals₃(song 3) ← Case A only
  → positive with x_ref[3] ✓
```

### 4.4 Positive Mask — Stem = drums

```
z_drums(x_art[0]): drums₀(song 0) + drums₃(song 3) ← Case A+B
  → positive with x_ref[0] ✓ (song 0)
  → positive with x_ref[3] ✓ (song 3)

z_drums(x_art[1]): drums₁(song 1) ← Case A only
  → positive with x_ref[1] ✓

z_drums(x_art[2]): drums 없음 → SKIP
z_drums(x_art[3]): drums 없음 → SKIP
```

---

## 5. Separation Target 설계

### 5.1 원칙

Separation target은 모델 입력에 적용된 것과 **동일한 audio effect**를 개별 stem에 적용하고, **동일한 VQT crop position**에서 추출한 cropped VQT이다. 현재 구현에서는 모델 입력 자체는 log-normalized magnitude이지만, separation target과 `L_sep` 비교는 **linear magnitude VQT**에서 수행한다.

```python
# mix_A 생성 시 effect 파라미터 기록
board.randomize_parameters()
effect_params = board.get_parameters()
mix_A = board(sum_of_stems_A, sr)

# 동일 effect를 개별 stem에 적용
board.set_parameters(effect_params)
for s in stem_types_A:
    target_s = board(stem_s_audio, sr)

# VQT 후 동일 crop position 적용
target_s_complex = extract_random_blocks(VQT(target_s), block_size, i=i_A, j=j_A)
target_s_linear = abs(target_s_complex)
```

### 5.2 Case A+B에서의 Separation Target

동일 stem 유형이 두 곡에서 모두 존재할 때, separation target은 **양쪽 effected stem의 complex VQT를 합산한 뒤 linear magnitude로 변환**한다:

```
target_drums_complex = crop(VQT(effect_A0(drums0)), pos_A0) + crop(VQT(effect_B3(drums3)), pos_B3)
target_drums_linear = abs(target_drums_complex)
```

이는 x_art[0] 안에 drums 성분이 실제로 존재하는 형태와 일치한다.

### 5.3 x_ref의 Separation Target

```python
for s in active_stems_song_i:
    target_s_ref[i] = stretch_and_crop(
        VQT(effect_C(stem_s_audio)),
        stretch_factor=same_as_ref[i],
        i=crop_size,
        j=j_ref[i]
    )
```

x_ref에는 time stretch가 적용되므로, separation target에도 동일한 stretch factor를 적용해야 한다.

---

## 6. 전체 Training Step 흐름

```python
def shared_step(batch):
    # ── 입력 분해 ──
    mix_A, mix_B, mix_AB, stem_types_A, stem_types_B, stem_types_AB, \
        indiv_stems_A, indiv_stems_B, indiv_stems_AB = batch

    # ── Stage 1: VQT 변환 ──
    with torch.no_grad():
        x_A, x_B, x_AB = VQT(cat(mix_A, mix_B, mix_AB)).chunk(3)
        x_stems_A = {s: VQT(audio) for s, audio in indiv_stems_A}
        x_stems_B = {s: VQT(audio) for s, audio in indiv_stems_B}
        x_stems_AB = {s: VQT(audio) for s, audio in indiv_stems_AB}

    # ── Stage 2: Crop + Augmentation (위치 추적) ──
    with torch.no_grad():
        x_ref, stretch_factors, j_ref = stretch_and_crop_tracked(x_AB, ...)
        x_A_cropped, i_A, j_A = crop_tracked(x_A, ...)
        x_B_cropped, i_B, j_B = crop_tracked(x_B, ...)
        x_art = x_A_cropped + x_B_cropped.roll(1, 0)

        # Separation targets (동일 crop position 적용)
        sep_targets = build_separation_targets(...)

    # ── Stage 3: Encoding ──
    F_art = encoder(x_art.x_input)   # lognorm input -> dense memory
    F_ref = encoder(x_ref.x_input)

    z_art, z_ref = {}, {}
    u_art, u_ref = {}, {}

    for s in all_stems:
        for i in range(B):
            if s in active_stems_art[i]:
                u_art[(i,s)] = source_query_extractor(F_art[i], stem=s)
                z_art[(i,s)] = g_s(u_art[(i,s)])
            if s in active_stems_ref[i]:
                u_ref[(i,s)] = source_query_extractor(F_ref[i], stem=s)
                z_ref[(i,s)] = g_s(u_ref[(i,s)])

    # ── Stage 4: Loss ──
    pred_linear = linear_mag_masks(u_art, u_ref) * branch.x_linear_mag
    L_sep = compute_separation_loss(pred_linear, sep_targets_linear)
    L_asid = compute_contrastive_loss(z_art, z_ref, stem_types_A, stem_types_B)

    return λ_sep * L_sep + λ_asid * L_asid
```

---

## 7. Riou와의 비교 요약

| 항목 | Riou | SepFP |
|---|---|---|
| 모델 출력 | 단일 z ∈ R^{2048} | stem별 z_s ∈ R^{D_z} (활성 stem만) |
| Positive mask | 고정 (diagonal + upper-diagonal) | **stem별 동적 mask** (provenance 기반) |
| Sim matrix 크기 | B × B (단일) | **B_active × B_active (stem별, 가변)** |
| Loss 항목 | InfoNCE 1개 | S개 InfoNCE + S개 L_sep |
| Skip logic | 없음 | **stem 부재 → head 비활성** |
| Multi-positive | upper-diag로 최대 2개 | provenance 기반, 0~2개 |

---

## 8. Edge Cases

### 8.1 Case A+B (동일 stem 유형이 두 곡에서 중첩)

- **빈도**: vocals, drums, bass는 대부분의 곡에 존재하므로 **매우 빈번**
- **영향**: z_s가 두 개의 서로 다른 x_ref와 동시에 positive → multi-positive InfoNCE 필요
- **해석**: 실제 ASID 시나리오(동일 유형 소스의 혼합 식별)를 반영하므로 올바른 학습 신호

### 8.2 Batch 내 특정 stem의 전면 부재

- batch 내 모든 곡에 piano가 없으면 → L_asid^piano = 0 (skip)
- 드문 경우이며 문제 없음

### 8.3 x_ref에서의 stem 부재

- x_ref[i]는 song_i의 모든 활성 stem을 포함
- 해당 곡에 없는 stem 유형만 부재 → sim matrix에서 제외

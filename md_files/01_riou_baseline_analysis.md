# Riou et al. (2025) 코드베이스 분석 — SepFP 참조 문서

> **목적**: SepFP의 baseline인 Riou et al. (2025)의 코드 구현을 분석하여,
> SepFP에 직접 차용하는 요소와 SepFP가 극복하려는 한계를 명확히 구분한다.
> 
> **참조 코드**: `/home/maclab/user_woojinkang/SepFP/_ref/sampleid`

---

## 1. 논문 개요

- **제목**: "Automatic Music Sample Identification with Multi-Track Contrastive Learning"
- **저자**: Alain Riou, Joan Serrà, Yuki Mitsufuji (Sony Research)
- **ArXiv**: 2510.11507
- **성능**: mAP 0.603 / HR@1 0.587 / HR@10 0.733

---

## 2. 핵심 메커니즘

### 2.1 Artificial Mix 기반 Positive Pair 생성

하나의 멀티트랙 곡에서 활성 stem들을 세 개의 disjoint 부분집합(A, B, C)으로 분할하여 세 가지 mix를 만든다:

```
candidates = 현재 구간에서 활성인 stem 집합

partition → A, B, C  (A ∩ B = ∅,  A ∪ B ⊆ candidates,  |A| ≥ 1,  |B| ≥ 1)

mix_A  = Σ audio[s] for s ∈ A         (일부 stem만의 mix)
mix_B  = Σ audio[s] for s ∈ B         (다른 일부 stem만의 mix)
mix_AB = Σ audio[s] for s ∈ candidates (전체 mix)

세 mix에 각각 독립적인 random audio effects 적용
```

학습 step에서 positive pair를 구성한다:

```python
x_ref[i] = stretch_and_crop(VQT(effect_C(mix_AB[i])))    # time-stretched 전체 mix
x_art[i] = crop(VQT(effect_A(mix_A[i]))) + crop(VQT(effect_B(mix_B[(i-1)%B])))
           # 곡i의 일부 stem + 곡(i-1)의 다른 stem을 합산
```

`roll(1, 0)` 연산으로 batch 내 서로 다른 곡의 stem을 혼합하여, 모델이 다른 곡의 소스와 혼합된 상태에서도 공유 stem을 식별하도록 강제한다.

**SepFP에서의 차용**: A/B/AB sub-mix와 batch roll 아이디어를 유지한다. 현재 구현은 여기에 stem-level provenance tracking과 separation target 생성을 추가하여, 어떤 stem 성분이 어느 원곡 파일에서 왔는지 loss 계산까지 전달한다.

### 2.2 InfoNCE with Upper-Diagonal Positive Mask

`x_art[i]`는 곡 i와 곡 (i-1)%B의 stem을 모두 포함하므로, `x_ref[i]`와 `x_ref[(i-1)%B]` 모두와 positive pair이다. 이를 `upper_diagonal=True`로 구현한다.

```python
pos_mask = torch.eye(n, dtype=bool)
pos_mask[:-1, 1:].fill_diagonal_(True)
pos_mask[-1, 0] = True
```

**SepFP에서의 변경**: 고정 mask 대신 **stem별 동적 positive mask** (provenance tracking 기반)로 대체한다. 현재 구현에서는 `BranchContext.provenance`의 token intersection으로 positive를 정의한다.

### 2.3 Encoder 구조 — ResNet50-IBN + VQT

```
VQT (36 bins/oct, 8 oct, gamma=7, Complex output)
  → LogNorm
  → ResNet50-IBN:
      Frontend: Conv2d(1→128, 36×3, s=(3,2)) → Conv2d(128→256, 12×3, s=2)
      Backbone: [3,4,6,3] blocks × [256,512,1024,2048] channels (stride [1,2,2,1])
      IBN: Instance-Batch Norm hybrid (절반 채널에 IN 적용)
  → GeMPool → BN → Linear → z ∈ R^{2048}
```

IBN의 역할: Instance Norm이 스타일(EQ, 음색) 정보를 제거하여 audio effect 불변성을 달성한다.

**현재 SepFP 구현과의 관계**: VQT frontend와 Riou-style artificial mixture/augmentation pipeline은 유지하지만, 모델 backbone은 더 이상 Riou의 ResNet50-IBN을 변형해 쓰지 않는다. 현재 구현은 `TFEvidenceEncoder`가 log-normalized VQT magnitude를 받아 dense memory `M ∈ R^{B × 256 × 63 × 64}`를 만들고, learned source query가 active stem별 evidence `u_s`를 추출한다.

### 2.4 Pitch Shift — VQT 주파수 축 Crop

VQT의 log-scale 특성을 이용해, 주파수 축을 무작위로 crop하는 것으로 pitch shift를 근사한다.

```
bins_per_octave = 36,  crop_size = 18 (= 36//2)
→ ±18 bins = ±0.5 옥타브 = ±6 semitone 범위의 pitch shift를 커버
```

`x_A`와 `x_B`에는 random crop (`i=None`), `x_ref`에는 고정 중앙 crop (`i=crop_size`)을 적용한다.

**SepFP에서의 차용**: 동일한 VQT 주파수축 crop 기반 pitch-shift 아이디어를 사용한다. SepFP는 crop 위치를 `BranchContext.crop_meta`에 저장하여 separation target도 같은 위치에서 만든다.

### 2.5 Time Stretch — Spectrogram 보간

VQT spectrogram의 시간 축을 `F.interpolate(mode='linear')`로 0.7x~1.5x 범위에서 보간한 후 고정 크기로 crop한다.

**SepFP에서의 차용**: 동일한 spectrogram interpolation 기반 time-stretch 아이디어를 사용한다. SepFP는 `x_ref`의 stretch factor를 기록하고, 개별 stem target에도 같은 factor를 적용한다.

### 2.6 Audio Effects — RandomizedPedalboard

```python
self.board.randomize_parameters()
mix_A = self.board(mix_A, sample_rate).mean(axis=0)
# 각 mix에 독립적으로 적용
```

Effects 구성 (MoisesDB config):
- PeakFilter: p=0.2, 440~8000Hz, ±20dB, random Q
- Compressor: p=0.2, threshold -30~0dB, ratio 1/2/4/8/20
- Gain: p=0.2, ±10dB

**SepFP에서의 차용**: 동일한 randomized effect-chain 계열을 사용한다. 추가로 동일 effect 파라미터를 개별 stem에도 replay하여 linear-magnitude separation target을 생성한다.

---

## 3. Riou의 한계 — SepFP의 동기

### 3.1 소스 희석 (Signal Dilution)

GeMPool이 모든 주파수·시간 정보를 하나의 벡터로 압축한다. 서로 다른 stem 조합(drums+bass vs vocals)이 모두 같은 전체 mix와 positive로 매핑되므로, 임베딩 공간에 ambiguity가 발생한다.

```
z(drums+bass) ≈ z_AB (positive)  &&  z(vocals) ≈ z_AB (positive)
→ z(drums+bass) ≈ z(vocals)  (의도치 않은 유사성)
```

**SepFP의 해결**: stem별 evidence `u_s`와 retrieval embedding `z_s`로 분해하여 각 stem 유형이 독립적인 identity space를 가지도록 한다. 현재 `z_s`는 `u_s`에서 나오지만, ASID gradient는 projector 입력에서 detach되어 projector만 직접 업데이트한다.

### 3.2 Upper-Diagonal Mask의 논리적 부담

`z_art[i]`가 곡 i와 곡 (i-1)%B 모두와 positive이므로, 간접적으로 서로 다른 곡의 임베딩이 가까워질 수 있다.

**SepFP의 해결**: stem-level provenance tracking으로 정확히 어떤 stem이 어떤 곡에서 유래했는지 추적하여 stem별로 정밀한 positive/negative를 정의한다.

### 3.3 `num_heads` 코드 주석 — Multi-Head 실험의 흔적

Riou의 코드에서 `num_heads` 파라미터는 항상 1로 사용되며, 주석은 `"old secret experiments 👀"`라고 기술한다. 이는 다중 헤드 실험이 이전에 시도되었을 가능성을 시사하나, 확증은 아니다.

SepFP의 stem-specific path는 Riou의 generic multi-head와 근본적으로 다르다: 각 active stem path가 `L_sep`를 통해 **명시적 separation supervision**을 받는다. 현재 구현에서는 source-query evidence extractor가 `u_s`를 만들고, decoder는 `u_s`만으로 linear-magnitude mask logits를 예측한다.

---

## 4. SepFP에 직접 차용하는 요소 요약

| 요소 | Riou 구현 | SepFP 사용 |
|---|---|---|
| VQT (36 bins/oct, gamma=7, Complex) | `src/data/vqt.py` | 그대로 사용 |
| Pitch shift (VQT 주파수 축 crop) | `extract_random_blocks` | 그대로 사용 |
| Time stretch (spectrogram 보간) | `stretch_and_crop` | 그대로 사용 |
| Audio effects (RandomizedPedalboard) | `cached_dataset.py` | 그대로 사용 + 개별 stem에 동일 적용 |
| Activation mask | `compute_activations.py` | oracle stem presence로 활용 |
| IBN ResNet backbone | `resnet_ibn.py` | 현재 구현에서는 사용하지 않음; `TFEvidenceEncoder`로 대체 |
| Contrastive loss 구조 | `contrastive_loss.py` | stem별 버전으로 확장 |
| 학습 가능 temperature (0.01 초기화) | `sample_id.py` | 그대로 사용 |
| Sub-mix partition + roll | `cached_dataset.py` + `sample_id.py` | 그대로 사용 + provenance tracking 추가 |
| Mask normalization | 해당 없음 | 현재 기본값은 `independent_capped` (`2.0 * sigmoid`); `active_softmax`는 ablation으로 유지 |

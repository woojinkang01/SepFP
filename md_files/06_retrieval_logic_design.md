# SepFP Retrieval Logic Design

> Current status: design note, implementation pending.
> This document defines the planned retrieval scoring logic for trained SepFP checkpoints.

## Purpose

훈련을 마친 SepFP 모델을 실제 reference DB retrieval에 사용하기 위한 similarity score 정의를 정리한다.

SepFP는 기존 단일-embedding ASID 시스템과 달리 하나의 audio segment에서 active stem별 embedding을 만든다. 따라서 retrieval logic은 다음 조건을 만족해야 한다.

- 한 segment에서 1~6개의 `z_s`가 존재할 수 있다.
- inactive stem embedding은 생성하지 않는다.
- query와 정답 reference가 공유하는 stem 종류와 개수는 inference 시점에 알 수 없다.
- `drums`만 공유할 수도 있고, `drums + guitar`처럼 여러 stem을 공유할 수도 있다.
- 서로 다른 stem projector의 embedding space는 분리되어 있으므로 same-stem끼리만 비교한다.

이 문서는 구현 예정 설계이며, 아직 repo에 inference/retrieval pipeline으로 구현되어 있지 않다.

## Current Model Contract

현재 SepFP 모델은 active stem `s`마다 다음 출력을 만든다.

```text
u_s: source-selective evidence map
pred_s: linear-magnitude VQT separation output
z_s: L2-normalized ASID retrieval embedding
```

Retrieval에는 `z_s`만 사용한다.

```text
z_s = EvidenceProjector_s(u_s.detach())
z_s shape = (512,)
```

중요한 제약:

- `z_s`는 stem-specific projector에서 나온다.
- `z_drums`는 reference의 `z_drums`와만 비교한다.
- `z_drums`와 `z_guitar` 같은 cross-stem 비교는 정의하지 않는다.
- `pred_s`, mask, linear-magnitude carrier는 retrieval score 계산에 직접 사용하지 않는다.

## Reference DB Precomputation

Reference DB의 각 곡은 7.2초 audio segment로 나눈다.

실제로는 fixed hop length를 사용해 overlapping segment를 만든다.

```text
reference track R
  -> segments r_1, r_2, ..., r_N
```

각 segment에 대해 SepFP inference를 수행한다.

```text
r_j -> {z_R[j,s] | s in active_stems(r_j)}
```

Silent segment는 skip한다.

```text
if mixture_energy(r_j) <= silence_threshold:
  skip segment
```

저장 단위는 다음 형태를 권장한다.

```text
track_id
segment_id
time_start
time_end
stem
embedding z_s
activity / energy metadata
```

Index는 stem별로 분리한다.

```text
index[vocals] = all reference vocals embeddings
index[drums]  = all reference drums embeddings
index[bass]   = all reference bass embeddings
...
```

## Query Inference

Query도 reference와 같은 방식으로 segment를 만든다.

```text
query Q
  -> segments q_1, q_2, ..., q_M
  -> {z_Q[i,s] | s in active_stems(q_i)}
```

Query에서도 silent segment는 skip하고, active stem에 대해서만 embedding을 추출한다.

주의할 점:

현재 SepFP training은 oracle active-stem metadata를 사용한다. 실제 inference에서는 stem activity detection 또는 신뢰 가능한 gating rule이 별도로 필요하다. Mixture magnitude threshold는 silent segment skip에는 사용할 수 있지만, per-stem active 여부를 직접 보장하지는 않는다.

## Segment-Pair Comparable Stems

Query segment `q_i`와 reference segment `r_j`가 있을 때, 비교 가능한 stem set은 active stem intersection이다.

```text
A_Q(i) = active stems of query segment i
A_R(j) = active stems of reference segment j

C(i,j) = A_Q(i) ∩ A_R(j)
```

`C(i,j)`가 비어 있으면 해당 segment pair는 score를 만들지 않는다.

```text
if C(i,j) is empty:
  P(i,j) = -inf
```

## Stem-Level Similarity

각 stem별 raw similarity는 cosine similarity이다.

`z_s`는 이미 L2-normalized이므로 dot product가 cosine이다.

```text
c_s(i,j) = dot(z_Q[i,s], z_R[j,s])
for s in C(i,j)
```

하지만 stem마다 embedding distribution과 false-positive 경향이 다를 수 있다. 특히 `others`는 heterogeneous stem이므로 raw cosine을 그대로 합산하는 것은 위험하다.

따라서 stem별 calibrated evidence를 사용한다.

```text
e_s(i,j) = w_s * max(0, (c_s(i,j) - tau_s) / beta_s)
```

각 항의 의미:

- `tau_s`: stem `s`의 background false-positive threshold
- `beta_s`: stem `s`의 score scale
- `w_s`: stem reliability weight
- `max(0, ...)`: 해당 stem이 match evidence를 주지 못하면 penalty로 쓰지 않음

초기값 예시:

```text
w_vocals = 1.0
w_drums  = 1.0
w_bass   = 1.0
w_guitar = 1.0
w_piano  = 1.0
w_others = 0.5
```

`tau_s`, `beta_s`, `w_s`는 validation retrieval set에서 보정해야 한다.

## Segment-Pair Score

권장 segment-pair score는 다음과 같다.

```text
P(i,j) = max_s e_s(i,j) + alpha * sum_{s != s*} e_s(i,j)

where:
  s* = argmax_s e_s(i,j)
  s in C(i,j)
```

초기값:

```text
alpha = 0.5
```

이 score의 의도:

- 하나의 공유 stem만 강하게 match되어도 높은 score를 준다.
- 여러 stem이 동시에 match되면 additional evidence로 score가 증가한다.
- query에 active stem이 많아도, 정답 reference와 공유하지 않는 stem을 모른다는 ASID 조건 때문에 낮은 stem score를 penalty로 쓰지 않는다.
- 단순 mean aggregation보다 partial-overlap 상황에 안전하다.
- 단순 max aggregation보다 multi-stem shared evidence를 더 잘 반영한다.

## Why Not Mean Over Stems

다음 상황을 고려한다.

```text
query active stems:     drums, guitar
reference active stems: drums, guitar
actual shared source:   drums only
```

이때 `guitar` similarity가 낮다고 해서 정답 reference score를 낮추면 안 된다. ASID에서는 어떤 active stem이 실제 shared source인지 inference 시점에 알 수 없기 때문이다.

따라서 stem score 평균은 primary score로 적합하지 않다.

```text
bad:
  P(i,j) = mean_s c_s(i,j)
```

## Why Not Max Only

단순 max는 partial-overlap에는 강하지만, 여러 stem이 함께 공유되는 evidence를 반영하지 못한다.

```text
P(i,j) = max_s e_s(i,j)
```

이 경우 `drums`만 match된 pair와 `drums + guitar`가 함께 match된 pair를 같은 수준으로 볼 수 있다.

따라서 max를 primary evidence로 사용하되, 나머지 positive evidence를 작은 가중치로 더한다.

```text
recommended:
  P(i,j) = max_s e_s(i,j) + alpha * sum_{s != s*} e_s(i,j)
```

## Track-Level Score

기존 ASID retrieval과 가장 가까운 track-level score는 segment-pair max이다.

```text
S_max(Q,R) = max_{i,j} P(i,j)
```

Reference track `R`의 retrieval rank는 `S_max(Q,R)`가 큰 순서로 정한다.

다만 DB가 커지고 segment 수가 많아지면 max pooling은 false spike와 track-length bias에 취약할 수 있다. 따라서 실제 구현에서는 top-k non-overlapping score를 rerank 또는 tie-breaker로 추가하는 것을 권장한다.

```text
S_topK(Q,R) = mean of top K non-overlapping P(i,j)
```

최종 score 예시:

```text
S(Q,R) = S_max(Q,R) + gamma * S_topK(Q,R)
```

초기값:

```text
K = 3
gamma = 0.1 ~ 0.2
```

Query가 짧아서 segment 수가 매우 적으면 `S_topK`는 생략하거나 `S_max`와 동일하게 처리한다.

## Candidate Generation

전체 reference DB에 대해 모든 query-reference segment pair를 직접 비교하지 않고, stem별 index를 활용한다.

절차:

1. Query segment `q_i`의 active stem `s`마다 `z_Q[i,s]`를 얻는다.
2. 같은 stem index `index[s]`에서 top-M nearest reference embeddings를 검색한다.
3. 검색 결과를 `(query_segment_id, reference_track_id, reference_segment_id)` 단위로 모은다.
4. 같은 segment pair에 대해 여러 stem 후보가 모이면 `P(i,j)`를 계산한다.
5. Track별로 `S(Q,R)`를 계산한다.
6. `S(Q,R)` 내림차순으로 retrieval 결과를 반환한다.

이 구조는 same-stem 비교만 수행하므로, stem-specific projector contract와 일치한다.

## Recommended MVP

초기 구현 예정 버전은 다음으로 제한한다.

```text
1. same-stem cosine only
2. active stem embedding only
3. stem별 top-M candidate generation
4. calibrated positive evidence e_s
5. segment score P = max + alpha * additional evidence
6. track score S = max segment-pair score
```

이후 validation 결과에 따라 다음을 추가한다.

```text
1. stem별 tau_s / beta_s calibration
2. w_s tuning, especially for others
3. top-k non-overlap reranking
4. track-length bias correction
5. hubness diagnostics per stem
6. false-positive analysis for inactive or weakly active stems
```

## Validation Plan

Retrieval logic 구현 후 최소한 다음을 측정해야 한다.

- stem-wise retrieval rank
- overall HR@1, HR@5, HR@10
- mAP
- positive/negative similarity distribution per stem
- `others` 포함 여부에 따른 성능 차이
- single-shared-stem case vs multi-shared-stem case
- false-positive spike가 특정 stem 또는 긴 reference track에 몰리는지 여부
- top-k reranking이 max-only score보다 안정적인지 여부

## Implementation Status

이 문서는 구현 예정인 retrieval scoring 설계 문서이다.

아직 다음 항목은 구현되어 있지 않다.

- audio file을 7.2초 overlapping segment로 나누는 inference dataset
- trained SepFP checkpoint loading용 inference wrapper
- per-stem active detection 또는 gating rule
- reference embedding precomputation script
- stem별 ANN 또는 brute-force index
- query retrieval script
- calibrated segment-pair scoring
- track-level reranking
- retrieval metric evaluation script

구현 시에는 이 문서의 MVP 정의를 먼저 따르고, validation 결과를 바탕으로 calibration과 reranking을 추가한다.

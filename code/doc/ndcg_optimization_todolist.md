# NDCG 고득점을 위한 전략적 To-Do 리스트

**작성자**: 30년차 시니어 개발자
**작성일**: 2025-12-12
**목표**: NDCG@10 점수 0.0868 → 0.10+ 달성

---

## 현재 상황 진단

```
현재 점수: 0.0868
베이스라인: 0.0863 (순수 ALS)
개선폭: +0.58% (미미함)

핵심 문제:
1. 검증-제출 환경 불일치 (filter=True vs False)
2. 오프라인 NDCG 검증 부재
3. 가중치 경험적 설정 (검증 없음)
4. 구매 의도 신호(cart, purchase) 미활용
```

---

## Phase 1: 기반 정비 (즉시 실행)

### 1.1 검증 환경 통일 ⭐ 최우선

**문제**: Optuna는 `filter=True`, 제출은 `filter=False`로 불일치

```bash
# TODO: optuna_als.py 수정 후 재실행
python optuna_als.py --n_trials 30
```

**수정 사항**:
```python
# optuna_als.py Line 94
filter_already_liked_items=False  # True → False
```

**예상 효과**: 검증 NDCG와 리더보드 점수 상관관계 개선

---

### 1.2 오프라인 검증 파이프라인 구축 ⭐ 최우선

**문제**: 제출 전 성능 예측 불가

**TODO**: `validate_offline.py` 생성

```python
# 핵심 구조
def offline_validation(model, train_df, test_ratio=0.2):
    """
    시간 기반 분할로 오프라인 NDCG 측정
    - 최근 20% 데이터를 테스트로 사용
    - filter_already_liked_items=False로 통일
    """
    # 1. 시간 기반 분할
    train, test = time_based_split(train_df, test_ratio)

    # 2. 모델 학습 (train만 사용)
    model.fit(train)

    # 3. 추천 생성 및 NDCG 계산
    ndcg = calculate_ndcg(model, test, k=10)

    return ndcg
```

**체크리스트**:
- [ ] 시간 기반 분할 구현
- [ ] NDCG@10 계산 함수
- [ ] 다양한 설정 비교 테이블 출력

---

### 1.3 Graded Relevance 도입

**문제**: Binary relevance (0/1)만 사용 중

**현재**:
```python
relevance = 1 if item in actual else 0
```

**개선**:
```python
def get_relevance(item, user_history):
    """event_type 기반 등급"""
    if item in user_purchases:
        return 3.0  # 구매
    elif item in user_carts:
        return 2.0  # 장바구니
    elif item in user_views:
        return 1.0  # 조회
    return 0.0
```

**NDCG 공식 수정**:
```python
def ndcg_graded(actual_relevances, predicted, k=10):
    dcg = sum(rel / np.log2(i + 2)
              for i, (item, rel) in enumerate(predicted[:k]))
    idcg = sum(rel / np.log2(i + 2)
               for i, rel in enumerate(sorted(actual_relevances, reverse=True)[:k]))
    return dcg / idcg
```

---

## Phase 2: 신호 강화 (1-3일)

### 2.1 event_type 가중치 적용 ⭐ 높은 기대치

**데이터 분포**:
```
view:     99.78%
cart:      0.20%
purchase:  0.02%
```

**TODO**: interaction matrix에 가중치 반영

```python
# ontology_recommender.py 수정
weight_map = {
    'view': 1,
    'cart': 5,      # 구매 의도 강함
    'purchase': 10  # 가장 강한 신호
}

train_df['weight'] = train_df['event_type'].map(weight_map)
user_item_weights = train_df.groupby(['user_idx', 'item_idx'])['weight'].sum()
```

**주의**: 12/10 실험에서 -0.3% 하락했으나, 당시 filter=True 환경
→ filter=False에서 재실험 필요

---

### 2.2 시간 Decay 가중치

**가설**: 최근 행동이 더 강한 구매 의도

```python
def time_decay_weight(event_time, reference_time, half_life_days=7):
    """지수 감쇠 함수"""
    days_ago = (reference_time - event_time).days
    return 2 ** (-days_ago / half_life_days)

# 적용
train_df['time_weight'] = train_df['event_time'].apply(
    lambda t: time_decay_weight(t, train_df['event_time'].max())
)
train_df['final_weight'] = train_df['weight'] * train_df['time_weight']
```

---

### 2.3 카테고리 계층 유사도

**문제**: `electronics.smartphone` ≠ `electronics.tablet`으로 취급

**개선**:
```python
def category_similarity(cat1, cat2):
    """계층적 카테고리 유사도"""
    if cat1 == cat2:
        return 1.0

    parts1 = cat1.split('.')
    parts2 = cat2.split('.')

    common = 0
    for p1, p2 in zip(parts1, parts2):
        if p1 == p2:
            common += 1
        else:
            break

    max_depth = max(len(parts1), len(parts2))
    return common / max_depth  # 0.0 ~ 1.0
```

**Content 점수에 적용**:
```python
# 기존: 정확히 일치하는 카테고리만
if item_cat == user_cat:
    score += 0.5

# 개선: 유사한 카테고리도 부분 점수
for user_cat, freq in user_cat_pref.items():
    sim = category_similarity(item_cat, user_cat)
    score += freq * sim * 0.5
```

---

## Phase 3: 가중치 최적화 (3-5일)

### 3.1 그리드 서치 실험

**현재**: ALS=0.8, Content=0.1, CoView=0.1 (경험적)

**TODO**: 체계적 탐색

```python
# 실험 조합
weight_combinations = [
    (0.9, 0.05, 0.05),  # ALS 강화
    (0.8, 0.15, 0.05),  # Content 강화
    (0.8, 0.05, 0.15),  # CoView 강화
    (0.7, 0.15, 0.15),  # 균형
    (0.7, 0.2, 0.1),    # Content 더 강화
    (0.7, 0.1, 0.2),    # CoView 더 강화
]

for als_w, content_w, coview_w in weight_combinations:
    ndcg = offline_validation(als_w, content_w, coview_w)
    print(f"ALS={als_w}, Content={content_w}, CoView={coview_w} → NDCG={ndcg:.4f}")
```

---

### 3.2 ALS 후보 수 조정

**현재**: N=100 고정

**실험**:
```python
for n_candidates in [50, 100, 200, 500]:
    ndcg = evaluate_with_candidates(n_candidates)
    # 후보 많을수록: 다양성 ↑, 정확도 ↓ (trade-off)
```

---

### 3.3 Co-view min_coview 조정

**현재**: min_coview=2 (2번 이상 함께 본 쌍만)

**실험**:
```python
for min_coview in [1, 2, 3, 5]:
    # min=1: 노이즈 많지만 커버리지 ↑
    # min=5: 신뢰도 높지만 커버리지 ↓
```

---

## Phase 4: 모델 다양화 (1주+)

### 4.1 사용자 세그먼트별 전략

**가설**: Heavy User와 Light User는 다른 추천 전략 필요

```python
def get_user_segment(user_id, interaction_count):
    if interaction_count >= 50:
        return 'heavy'    # ALS 신뢰
    elif interaction_count >= 10:
        return 'medium'   # 균형
    else:
        return 'light'    # Content 의존

# 세그먼트별 가중치
segment_weights = {
    'heavy':  (0.9, 0.05, 0.05),
    'medium': (0.8, 0.1, 0.1),
    'light':  (0.6, 0.25, 0.15),
}
```

---

### 4.2 앙상블 전략

**후보 모델**:
1. ALS (현재 베이스라인)
2. BPR (Bayesian Personalized Ranking)
3. SASRec (Sequential 추천)

**앙상블 방식**:
```python
def ensemble_recommend(user_id):
    # 각 모델에서 Top 20 추출
    als_recs = als_model.recommend(user_id, N=20)
    bpr_recs = bpr_model.recommend(user_id, N=20)

    # 점수 융합 (Reciprocal Rank Fusion)
    scores = defaultdict(float)
    for rank, item in enumerate(als_recs):
        scores[item] += 1 / (rank + 1) * 0.6  # ALS 가중치
    for rank, item in enumerate(bpr_recs):
        scores[item] += 1 / (rank + 1) * 0.4  # BPR 가중치

    return sorted(scores.items(), key=lambda x: -x[1])[:10]
```

---

## Phase 5: 고급 최적화 (2주+)

### 5.1 Learning to Rank (LTR)

**아이디어**: 추천 순서를 직접 학습

```python
# Feature 예시
features = [
    'als_score',
    'content_score',
    'coview_score',
    'item_popularity',
    'user_item_category_match',
    'price_similarity',
    'brand_match',
    'recency_weight',
]

# LightGBM LambdaRank
model = lgb.LGBMRanker(
    objective='lambdarank',
    metric='ndcg',
    n_estimators=100
)
```

---

### 5.2 Negative Sampling 전략

**현재**: 모든 미상호작용 아이템이 동등한 negative

**개선**: Hard Negative Mining
```python
def get_hard_negatives(user_id, positive_items):
    """사용자가 클릭하지 않았지만 노출된 아이템"""
    # 같은 카테고리지만 선택 안 한 아이템
    # 인기 있지만 선택 안 한 아이템
    pass
```

---

## 실행 우선순위 매트릭스

| 순위 | 작업 | 예상 효과 | 난이도 | 소요 시간 |
|------|------|----------|--------|----------|
| 1 | 검증 환경 통일 | ★★★★★ | 낮음 | 1시간 |
| 2 | 오프라인 검증 구축 | ★★★★★ | 중간 | 4시간 |
| 3 | event_type 가중치 | ★★★★☆ | 낮음 | 2시간 |
| 4 | 시간 decay | ★★★☆☆ | 낮음 | 2시간 |
| 5 | 가중치 그리드 서치 | ★★★★☆ | 중간 | 8시간 |
| 6 | 카테고리 계층 유사도 | ★★★☆☆ | 중간 | 4시간 |
| 7 | 사용자 세그먼트 | ★★★☆☆ | 중간 | 6시간 |
| 8 | Graded NDCG | ★★☆☆☆ | 중간 | 4시간 |
| 9 | 앙상블 (BPR) | ★★★☆☆ | 높음 | 1일 |
| 10 | Learning to Rank | ★★★★☆ | 높음 | 3일 |

---

## 즉시 실행 체크리스트

### 오늘 할 일 (Day 1)

- [ ] `optuna_als.py` filter=False로 수정
- [ ] Optuna 재실행 (n_trials=30)
- [ ] `validate_offline.py` 생성
- [ ] 오프라인 NDCG 베이스라인 측정

### 내일 할 일 (Day 2)

- [ ] event_type 가중치 실험 (filter=False 환경)
- [ ] 시간 decay 실험
- [ ] 결과 비교 테이블 작성

### 이번 주 할 일 (Day 3-5)

- [ ] 가중치 그리드 서치 (15개 조합)
- [ ] 카테고리 계층 유사도 구현
- [ ] ALS 후보 수 실험 (50, 100, 200)
- [ ] 최적 설정으로 제출

---

## 성공 기준

| 단계 | 목표 점수 | 달성 방법 |
|------|----------|----------|
| 현재 | 0.0868 | - |
| Phase 1 완료 | 0.090+ | 검증 통일, event_type |
| Phase 2 완료 | 0.095+ | 시간 decay, 카테고리 계층 |
| Phase 3 완료 | 0.100+ | 가중치 최적화 |
| Phase 4 완료 | 0.105+ | 앙상블 |

---

## 핵심 원칙

> **"측정할 수 없으면 개선할 수 없다"**
> - 모든 변경은 오프라인 NDCG로 먼저 검증
> - 한 번에 하나씩만 변경하여 효과 측정
> - 리더보드 제출은 오프라인에서 검증된 것만

> **"단순함이 복잡함을 이긴다"**
> - ALS가 0.0863인데 복잡한 앙상블이 0.0868
> - 복잡도 대비 효과가 미미하면 버린다
> - 검증된 작은 개선을 누적하는 것이 핵심

---

**다음 단계**: `optuna_als.py` 수정 및 재실행

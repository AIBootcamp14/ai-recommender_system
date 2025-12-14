# NDCG@10 최적화 전략 - E-commerce 추천 시스템

## 현재 상황
- **Baseline (ALS)**: 0.0863
- **Ontology v4**: 0.0872 (+1.04%)
- **목표**: 0.10+

## 데이터 특성 (한줄 요약)
> "638K 사용자 × 29.5K 아이템의 대규모 이커머스 데이터로, 8.35M 상호작용 중 view가 대다수이며 purchase는 희소하고, 세션 기반 co-view 정보(3.6M pairs)가 풍부한 implicit feedback 데이터"

### 상세 통계
- 638,257 테스트 사용자
- 29,502 아이템
- 8,350,311 학습 상호작용
- 24 카테고리, 1,845 브랜드
- 3,636,954 co-view pairs
- 1,291,657 멀티아이템 세션

---

## 🎯 NDCG@10 최고점을 위한 5대 전략

### 1. "1위를 맞춰라" - Precision@1 집중
```
NDCG@10에서 1위 정답 = 만점, 10위 정답 = 29% 가치
→ 확실한 1개를 1위에 두는 것이 애매한 10개보다 중요
→ Confidence 기반 정렬: 점수 차이가 큰 아이템을 상위로
```

### 2. ⭐ Purchase 예측 직접 모델링 (Two-Stage) - 우선순위 1
```
Stage 1: ALS로 후보 100개 생성
Stage 2: "이 유저가 이 아이템을 구매할 확률" 직접 예측
  - Features: 유저 활동량, 아이템 인기도, 카테고리 선호도,
              cart→purchase 전환율, 시간 패턴
  - Model: LightGBM / XGBoost
예상 개선: +10~20% (0.0872 → 0.095~0.10)
```

### 3. 구매 전환 시그널 극대화
```python
# 현재: view=1, cart=10, purchase=20
# 제안: cart의 가치를 더 높임 (구매 직전 시그널)
weights = {'view': 1, 'cart': 30, 'purchase': 50}

# 추가: cart → purchase 전환된 아이템은 2배 가중
```

### 4. Popularity Bias 활용 (역발상)
```
• 테스트셋의 정답도 인기 아이템에 편중될 가능성 높음
• 개인화 점수 × 인기도^α (α=0.1~0.3) 하이브리드
• 특히 cold/light user에게 인기 아이템 추천이 유리
```

### 5. ⭐ 유저 세그먼트별 차별화 - 우선순위 2
```
Heavy User (50+ interactions): 순수 개인화 ALS
Medium User (10-50): ALS + Category 선호도
Light User (< 10): 인기도 + 카테고리 기반
Cold User (0): 글로벌 인기 아이템
예상 개선: +5~10%
```

---

## 실험 히스토리

| 버전 | 설정 | NDCG@10 | 개선율 | 날짜 |
|------|------|---------|--------|------|
| Baseline | ALS 기본 | 0.0863 | - | - |
| Ontology v2 | ALS + Content + CoView | 0.0868 | +0.58% | - |
| Ontology v4 | + Event 가중치 (v=1,c=10,p=20) | 0.0872 | +1.04% | 2024-12-14 |
| Two-Stage | ALS + LightGBM Rerank | TBD | TBD | - |

---

## Co-view 정보란?
**"같은 세션에서 함께 본 아이템 쌍"** - 3.6M pairs 보유
- 암묵적 아이템 유사도 정보
- Item2Vec, GNN 등으로 확장 가능

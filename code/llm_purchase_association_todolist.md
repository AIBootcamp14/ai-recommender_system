# LLM + 구매 연관도 활용 전략

> 베테랑 AI 엔지니어 관점의 To-Do List

---

## 현실 인식 (Reality Check)

### 데이터 특성
```
purchase 비율: 0.02% (전체 8.35M 중 ~1,670건)
cart 비율: 0.20%
view 비율: 99.78%
```

**핵심 문제**: Purchase 데이터가 극히 희소함 → 연관규칙 신뢰도 낮음

---

## Phase 1: 데이터 증강 전략 (1주차)

### 1.1 Purchase Proxy 정의
```python
# 실제 purchase가 부족하므로 "구매 의도" 시그널 확장
purchase_intent_signals = {
    'purchase': 10,      # 실제 구매
    'cart': 5,           # 장바구니 (구매 의도 강함)
    'repeated_view': 3,  # 같은 아이템 3회 이상 조회
    'long_view': 2,      # 세션 내 마지막 조회 (체류)
}
```

### 1.2 세션 기반 구매 시퀀스 추출
- [ ] 세션 내 view → cart → purchase 패턴 추출
- [ ] 구매 직전 5개 아이템 시퀀스 저장
- [ ] 카테고리 전환 패턴 분석 (예: shoes.view → socks.purchase)

---

## Phase 2: 연관규칙 마이닝 (1주차)

### 2.1 기존 Basket Analysis 확장
```python
# 구매 기반 연관규칙 (Support 기준 완화 필요)
association_rules = {
    'min_support': 0.0001,  # 극히 낮게 설정 (희소 데이터)
    'min_confidence': 0.1,
    'min_lift': 2.0,        # Lift는 높게 유지
}
```

### 2.2 카테고리 레벨 연관규칙
- [ ] 아이템 레벨 희소 → 카테고리 레벨로 집계
- [ ] 예: `apparel.shoes → apparel.socks` (Lift=5.2)
- [ ] 브랜드 간 연관도 분석

### 2.3 시간 가중치 적용
- [ ] 최근 구매에 높은 가중치
- [ ] 계절성 반영 (11월~2월 데이터)

---

## Phase 3: LLM 활용 전략 (2주차)

### 3.1 LLM 역할 정의

```
┌─────────────────────────────────────────────────────────┐
│  LLM은 "추천 점수 생성기"가 아닌 "지식 증강기"로 활용   │
└─────────────────────────────────────────────────────────┘
```

**올바른 활용**:
1. 아이템 메타데이터 enrichment
2. 카테고리 간 의미적 유사도
3. 추천 이유 생성 (설명 가능성)

**잘못된 활용**:
- LLM에게 직접 추천 순위 요청 (hallucination 위험)
- 실시간 추론 (latency, 비용)

### 3.2 LLM 기반 아이템 임베딩 생성

```python
# To-Do: 아이템 설명 생성 및 임베딩
prompt = """
Given the following product information:
- Category: {category_code}
- Brand: {brand}
- Co-purchased with: {co_purchase_items}

Generate a 2-sentence product description focusing on:
1. What type of customer buys this
2. What other products they might need
"""

# 결과를 text-embedding-3-small로 임베딩
item_embedding = openai.embeddings.create(model="text-embedding-3-small", ...)
```

- [ ] 상위 1000개 인기 아이템에 대해 설명 생성
- [ ] 임베딩 벡터 저장 (FAISS/Pinecone)
- [ ] 유사 아이템 검색에 활용

### 3.3 카테고리 의미 그래프 구축

```python
# LLM으로 카테고리 간 관계 추론
category_relations = """
Q: What products do customers typically buy together with shoes?
A: socks, shoe care products, insoles, shoe bags

Q: If someone buys running shoes, what else might they need?
A: athletic socks, water bottle, fitness tracker, sports apparel
"""

# 이 지식을 Neo4j 그래프에 추가
# (Category)-[:COMPLEMENTARY]->(Category)
```

- [ ] 24개 카테고리 간 보완재 관계 정의
- [ ] LLM 생성 관계를 그래프에 추가
- [ ] GraphRAG 쿼리에서 활용

---

## Phase 4: 하이브리드 추천 파이프라인 (2-3주차)

### 4.1 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                    추천 요청                              │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 1: Candidate Generation (빠른 후보 생성)          │
│  - ALS: Top 100 candidates                              │
│  - GraphRAG: Top 50 co-purchase candidates              │
│  - Popular items fallback: Top 20                       │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 2: Re-ranking (LLM 임베딩 기반)                   │
│  - User history embedding (mean of viewed items)        │
│  - Candidate embedding similarity                       │
│  - Purchase association boost                           │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 3: Final Score                                   │
│  score = 0.5*ALS + 0.3*Graph + 0.2*LLM_sim             │
│        + association_boost (if co-purchase exists)     │
└─────────────────────────────────────────────────────────┘
```

### 4.2 구현 To-Do

- [ ] `candidate_generator.py`: ALS + Graph 후보 생성
- [ ] `llm_embedder.py`: 아이템 임베딩 생성/캐싱
- [ ] `reranker.py`: 최종 점수 계산
- [ ] `ensemble.py`: 가중치 튜닝

---

## Phase 5: 구매 연관도 부스팅 (3주차)

### 5.1 연관규칙 기반 점수 부스팅

```python
def apply_association_boost(user_history, candidates, rules):
    """
    사용자가 본 아이템과 연관된 아이템에 부스트
    """
    boosted_scores = {}

    for candidate in candidates:
        boost = 0
        for viewed_item in user_history:
            # 연관규칙 조회
            rule = rules.get((viewed_item, candidate))
            if rule:
                # Lift 기반 부스트
                boost += rule['lift'] * rule['confidence']

        boosted_scores[candidate] = boost

    return boosted_scores
```

### 5.2 카테고리 레벨 부스팅

```python
def category_affinity_boost(user_category_history, candidate_category):
    """
    사용자의 카테고리 선호도 기반 부스트
    """
    # 사용자가 자주 구매한 카테고리와 보완 관계면 부스트
    complementary_cats = get_complementary_categories(candidate_category)

    overlap = set(user_category_history) & set(complementary_cats)
    return len(overlap) * 0.1  # 보완 카테고리 수에 비례
```

---

## Phase 6: 실험 및 평가 (3-4주차)

### 6.1 A/B 테스트 설계

| 실험 | 설명 | 예상 효과 |
|-----|------|----------|
| Baseline | ALS only | 0.0863 |
| +Graph | ALS + GraphRAG | +1~2% |
| +LLM Embed | + LLM 임베딩 유사도 | +0.5~1% |
| +Association | + 구매 연관 부스트 | +1~2% |

### 6.2 오프라인 평가 지표

- [ ] NDCG@10 (현재 평가 지표)
- [ ] Recall@10
- [ ] Coverage (추천 다양성)
- [ ] Novelty (인기 아이템 편향 측정)

### 6.3 비용 분석

```
LLM API 비용 (OpenAI):
- 아이템 설명 생성: ~$0.01/item × 1000 items = $10
- 임베딩 생성: ~$0.0001/item × 29,502 items = $3
- 총 1회성 비용: ~$15

추론 시 비용: $0 (임베딩 캐싱 사용)
```

---

## 구현 우선순위

### 높음 (즉시 시작)
1. **카테고리 레벨 연관규칙** 추출
2. **GraphRAG + ALS 앙상블** 구현
3. **구매 의도 시그널** 정의 (cart, repeated_view)

### 중간 (검증 후)
4. LLM 아이템 임베딩 생성
5. Re-ranking 파이프라인

### 낮음 (시간 여유 시)
6. LLM 기반 카테고리 관계 추론
7. 실시간 추천 설명 생성

---

## 핵심 인사이트

### 왜 LLM "직접" 추천은 안 되는가?

1. **Hallucination**: 존재하지 않는 아이템 추천 가능
2. **Latency**: 실시간 서빙 불가 (API 호출 ~1초)
3. **비용**: 대규모 사용자 추론 시 비용 폭발
4. **일관성 부족**: 같은 입력에 다른 출력

### LLM의 올바른 역할

```
LLM = "오프라인 지식 증강기"
     - 아이템 메타데이터 enrichment
     - 카테고리 관계 추론
     - 임베딩 기반 유사도

NOT = "실시간 추천 엔진"
```

### 이 데이터셋에서의 현실적 기대

```
현재 최고: 0.0863 (ALS)
목표: 0.09+

예상 기여도:
- GraphRAG 앙상블: +0.005~0.01
- 연관규칙 부스팅: +0.003~0.005
- LLM 임베딩: +0.002~0.005

총 예상: 0.09~0.095
```

---

## 파일 생성 계획

| 파일명 | 설명 | 우선순위 |
|--------|------|----------|
| `purchase_proxy.py` | 구매 의도 시그널 추출 | 높음 |
| `category_association.py` | 카테고리 연관규칙 | 높음 |
| `ensemble_recommender.py` | ALS + Graph 앙상블 | 높음 |
| `llm_embedder.py` | LLM 임베딩 생성 | 중간 |
| `reranker.py` | 최종 점수 계산 | 중간 |

---

## 결론

> "LLM은 은탄환이 아니다. 이 대회에서는 **희소한 구매 데이터를 어떻게 활용하느냐**가 관건이다."

1. **Cart를 구매 proxy로 활용** (5배 더 많은 데이터)
2. **카테고리 레벨 연관규칙** (아이템 레벨은 너무 희소)
3. **LLM은 임베딩 생성에만 활용** (직접 추천 X)
4. **GraphRAG는 설명 가능성 + 약간의 성능 향상**

핵심은 **단순한 ALS가 이미 강력**하다는 것. 복잡한 파이프라인보다 **데이터 이해와 적절한 앙상블**이 중요하다.

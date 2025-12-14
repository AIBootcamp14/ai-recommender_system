# 온톨로지 기반 추천 시스템 적합성 검토 보고서

**검토자**: 30년차 시니어 데이터 과학자
**검토일**: 2025-12-12
**검토 대상**: `ontology_recommender.py`

---

## Executive Summary

현재 시스템은 **"온톨로지"라는 명칭을 사용하지만, 실제로는 속성 기반 콘텐츠 필터링(Attribute-based Content Filtering)에 가깝다.** 진정한 온톨로지 시스템이라면 개념 간의 계층적 관계(is-a, has-a)와 추론 규칙이 명시되어야 한다.

**적합성 판정: 부분적 적합 (60/100)**

---

## 1. 용어 정의의 정확성

### 1.1 "온톨로지"의 학술적 정의

```
온톨로지(Ontology): 특정 도메인 내의 개념(concept),
개념 간의 관계(relation), 속성(attribute),
그리고 추론 규칙(inference rule)을 형식적으로 명세한 것
```

### 1.2 현재 구현과의 비교

| 요소 | 학술적 정의 | 현재 구현 | 판정 |
|------|-------------|-----------|------|
| **개념(Concept)** | Category, Brand, Item 등의 클래스 정의 | 단순 딕셔너리 매핑 | △ 부분 |
| **관계(Relation)** | is-a, part-of, similar-to 등 | 없음 | × 미흡 |
| **계층 구조** | Category → Subcategory → Item | 평면적 구조 | × 미흡 |
| **추론 규칙** | "전자제품을 좋아하면 → 액세서리도 관심" | 없음 | × 미흡 |
| **속성(Attribute)** | 카테고리, 브랜드, 가격 | ✓ 구현됨 | ○ 충분 |

### 1.3 결론

> **현재 시스템은 "속성 기반 콘텐츠 필터링 + ALS 협업 필터링 + Co-view 그래프"의 앙상블이다.**
> "온톨로지"라는 명칭은 마케팅적 과장에 가깝다.

---

## 2. 사용자-아이템 연관성 모델링 분석

### 2.1 현재 연관성 신호

```
┌─────────────────────────────────────────────────────────┐
│  Signal 1: ALS (Collaborative Filtering) - 80%          │
│  - 행렬 분해 기반 잠재 요인                              │
│  - user_factors × item_factors                          │
│  - 장점: 협업 패턴 포착                                  │
│  - 단점: Cold-start, 해석 불가                          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Signal 2: Content (Attribute Matching) - 10%           │
│  - 카테고리 일치: 50%                                   │
│  - 브랜드 일치: 30%                                     │
│  - 가격대 유사: 20%                                     │
│  - 장점: 해석 가능, Cold-start 대응                     │
│  - 단점: 과거 선호에 갇힘 (Filter Bubble)               │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Signal 3: Co-view (Session Graph) - 10%                │
│  - 세션 내 함께 본 아이템 유사도                         │
│  - Item-Item 유사도                                     │
│  - 장점: 세션 컨텍스트 반영                             │
│  - 단점: 단기 관심만 반영                               │
└─────────────────────────────────────────────────────────┘
```

### 2.2 연관성 측정의 문제점

#### 문제 1: 구매 신호 미활용
```python
# 현재: 모든 이벤트를 동일하게 취급
train_df.groupby(['user_idx', 'item_idx']).size()

# 데이터 분포
# view:     99.78%
# cart:      0.20%
# purchase:  0.02%
```
**비판**: 구매(purchase)는 가장 강력한 구매 의도 신호인데, view와 동일하게 취급됨.

#### 문제 2: 시간 정보 미활용
```python
# 현재: 시간 가중치 없음
# 1개월 전 클릭 = 어제 클릭 (동일 취급)
```
**비판**: 최근 행동이 더 강한 구매 의도를 나타내는데 반영 안 됨.

#### 문제 3: 세션 경계 무시
```python
# Co-view는 세션 단위로 계산되지만
# ALS는 전체 기록을 평탄화
```
**비판**: 한 세션 내 집중 탐색 vs 여러 세션에 걸친 탐색의 의미가 다름.

---

## 3. 콘텐츠 점수 산출 로직 검토

### 3.1 현재 구현 (Lines 201-223)

```python
def get_content_score(self, user_id, item_id):
    score = 0.0

    # 카테고리: 과거 빈도 × 0.5
    if item_cat in user_cat_pref:
        score += user_cat_pref[item_cat] * 0.5

    # 브랜드: 과거 빈도 × 0.3
    if item_brand in user_brand_pref:
        score += user_brand_pref[item_brand] * 0.3

    # 가격: 유사도 × 0.2
    tier_diff = abs(item_tier - user_tier)
    score += (1 - tier_diff / 4) * 0.2

    return score
```

### 3.2 문제점 분석

| 문제 | 설명 | 심각도 |
|------|------|--------|
| **Filter Bubble** | 과거 본 카테고리만 추천 → 새로운 발견 차단 | 높음 |
| **가중치 고정** | 0.5/0.3/0.2가 최적인지 검증 없음 | 중간 |
| **이진 매칭** | 카테고리 일치/불일치만 판단, 유사도 없음 | 높음 |
| **계층 무시** | electronics.smartphone ≈ electronics.tablet인데 다르게 취급 | 높음 |

### 3.3 개선 제안

```python
# 제안: 카테고리 계층 유사도
def category_similarity(cat1, cat2):
    # electronics.smartphone vs electronics.tablet
    parts1 = cat1.split('.')
    parts2 = cat2.split('.')

    common_depth = 0
    for p1, p2 in zip(parts1, parts2):
        if p1 == p2:
            common_depth += 1
        else:
            break

    return common_depth / max(len(parts1), len(parts2))
```

---

## 4. Co-view 점수의 통계적 타당성

### 4.1 현재 구현 (Lines 225-241)

```python
def get_coview_score(self, user_id, candidate_idx):
    user_items = self.interaction_matrix[user_idx].indices
    coview_scores = self.coview_matrix[user_items, candidate_idx]
    return np.mean(coview_scores)
```

### 4.2 통계적 문제

#### 문제 1: 평균의 함정
```
사용자 A: [아이템1(coview=0.9), 아이템2(coview=0.1)] → 평균 0.5
사용자 B: [아이템1(coview=0.5), 아이템2(coview=0.5)] → 평균 0.5

→ 동일한 점수지만 A는 강한 연관, B는 약한 연관
```

#### 문제 2: 정규화 편향
```python
# L1 정규화 후
row_sums = self.coview_matrix.sum(axis=1)
self.coview_matrix /= row_sums

# 문제: 인기 아이템은 많은 아이템과 co-view되어 개별 점수가 희석됨
```

### 4.3 개선 제안

```python
# 제안: TF-IDF 스타일 가중치
def get_coview_score_tfidf(self, user_id, candidate_idx):
    # IDF: 얼마나 특별한 co-view인가
    idf = log(n_items / (1 + coview_count[candidate_idx]))

    # TF: 사용자 히스토리에서 얼마나 자주 co-view되는가
    tf = coview_matrix[user_items, candidate_idx].sum()

    return tf * idf
```

---

## 5. ALS 후보 생성의 적절성

### 5.1 현재 설정

```python
n_candidates = 100  # 고정
filter_already_liked_items = False  # 재방문 허용
```

### 5.2 분석

| 설정 | 판정 | 근거 |
|------|------|------|
| **N=100** | △ 보통 | 100개 내에서만 리랭킹, 다양성 제한 |
| **filter=False** | ○ 적절 | 이 대회에서 재방문이 중요 (실험으로 검증됨) |

### 5.3 우려 사항

```
ALS Top 100 후보 내에서만 리랭킹
→ Content/CoView 신호가 아무리 좋아도
→ ALS가 선택하지 않은 아이템은 추천 불가
→ 다양성 손실
```

---

## 6. 가중치 융합의 이론적 근거

### 6.1 현재 가중치

```python
als_weight = 0.8
content_weight = 0.1
coview_weight = 0.1
```

### 6.2 문제점

| 문제 | 설명 |
|------|------|
| **경험적 설정** | 이론적/실험적 근거 없이 설정 |
| **동일 스케일 가정** | ALS, Content, CoView 점수가 같은 범위라고 가정 |
| **사용자별 차이 무시** | 모든 사용자에게 동일한 가중치 적용 |

### 6.3 개선 제안

```python
# 제안 1: 사용자별 가중치 학습
# 활동량 많은 사용자 → ALS 신뢰
# 신규 사용자 → Content 의존

# 제안 2: 점수 정규화 통일
als_norm = (als - als.min()) / (als.max() - als.min())
content_norm = (content - content.min()) / (content.max() - content.min())
coview_norm = (coview - coview.min()) / (coview.max() - coview.min())
```

---

## 7. 실험 결과 해석

### 7.1 결과 요약

| 버전 | 설정 | 점수 |
|------|------|------|
| 순수 ALS | filter=False | 0.0863 |
| Ontology v2 | +Content +CoView | **0.0868** |
| Ontology v3 | +클릭 가중치 | 0.0868 |

### 7.2 해석

```
개선폭: +0.0005 (+0.58%)

이것이 의미하는 바:
1. Content/CoView 신호가 약간의 개선을 제공
2. 그러나 ALS가 이미 대부분의 정보를 포착
3. 추가 신호의 marginal utility가 낮음
```

### 7.3 통계적 유의성 의문

```
0.58% 개선이 통계적으로 유의한가?
- 단일 제출로는 판단 불가
- 여러 번 실행하여 분산 확인 필요
- 우연의 일치일 가능성 있음
```

---

## 8. 종합 평가

### 8.1 강점

| 항목 | 평가 |
|------|------|
| 구현 품질 | 깔끔한 클래스 구조, 모듈화 양호 |
| 확장성 | 새로운 신호 추가 용이 |
| 실용성 | LLM 없이 빠른 추론 가능 |
| Cold-start | Content 신호로 부분 대응 |

### 8.2 약점

| 항목 | 평가 |
|------|------|
| 명칭 정확성 | "온톨로지"라기보다 "속성 기반 필터링" |
| 구매 신호 | event_type(cart, purchase) 미활용 |
| 시간 정보 | 시간 decay 미적용 |
| 카테고리 계층 | 평면적 일치만 확인 |
| 가중치 최적화 | 경험적 설정, 학습 없음 |

### 8.3 점수

| 영역 | 점수 | 비고 |
|------|------|------|
| 구현 품질 | 80/100 | 깔끔하지만 최적화 여지 |
| 이론적 타당성 | 50/100 | "온톨로지" 명칭 부적절 |
| 실험적 검증 | 60/100 | 단일 실험, 통계적 유의성 미검증 |
| 실용적 가치 | 70/100 | +0.58% 개선, 한계 있음 |
| **종합** | **60/100** | 부분적 적합 |

---

## 9. 권고 사항

### 9.1 즉시 적용 (Quick Wins)

1. **명칭 변경**: "Ontology" → "Multi-Signal" 또는 "Hybrid"
2. **event_type 가중치 실험**: cart=3, purchase=10
3. **가중치 그리드 서치**: content_weight, coview_weight 조합 탐색

### 9.2 중기 개선 (1-2주)

1. **카테고리 계층 유사도**: electronics.phone ≈ electronics.tablet
2. **시간 decay**: 최근 클릭에 높은 가중치
3. **Co-view TF-IDF**: 희귀한 co-view에 높은 가중치

### 9.3 장기 개선 (1개월+)

1. **진정한 온톨로지 구축**:
   - 카테고리 계층 트리
   - 보완재/대체재 관계
   - 추론 규칙 (A→B 관계)

2. **가중치 학습**:
   - 사용자별 최적 가중치
   - 온라인 학습으로 동적 조정

---

## 10. 결론

> 현재 시스템은 **실용적이고 작동하는 하이브리드 추천 시스템**이다.
> 그러나 **"온톨로지"라는 명칭은 과장**이며,
> **구매 의도를 정밀하게 포착하기에는 신호 활용이 부족**하다.
>
> **핵심 개선점**: event_type, 시간 정보, 카테고리 계층을 활용하면
> 사용자-아이템 구매 연관성을 더 정확하게 모델링할 수 있다.

---

**작성자**: 30년차 시니어 데이터 과학자
**검토 완료**: 2025-12-12

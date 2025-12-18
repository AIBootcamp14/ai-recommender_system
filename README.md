# E-Commerce Recommender System

## Overview
이커머스 사용자 행동 데이터 기반 상품 추천 시스템입니다. EDA 기반 휴리스틱에서 시작하여 XGBoost Learning to Rank 모델까지 발전시켰습니다.

**Final Score: NDCG@10 = 0.1455**

## Problem Definition

| 항목 | 값 |
|------|-----|
| 목표 | 각 사용자에게 구매 가능성 높은 상위 10개 아이템 추천 |
| 평가 지표 | NDCG@10 |
| 총 이벤트 | 8,350,311건 |
| 대상 사용자 | 638,257명 |
| 아이템 | 29,502개 |
| 데이터 기간 | 2019-11-01 ~ 2020-02-29 (120일) |

---

## EDA 핵심 발견

### 이벤트 분포 및 전환율
| Event Type | Count | 비율 |
|------------|-------|------|
| View | 8,331,873 | 99.78% |
| Cart | 16,362 | 0.20% |
| Purchase | 2,076 | 0.02% |

**전환율 분석:**
- View → Cart: 0.20%
- Cart → Purchase: **12.69%** (장바구니는 강력한 구매 신호)
- View → Purchase: 0.02%

### 가격대별 전환율
| Price Range | 전환율 |
|-------------|--------|
| $0-50 (저가) | 0.036% (최고) |
| $50-100 | 0.019% |
| $100-200 | 0.020% |
| $200-500 | 0.030% |
| $500+ (고가) | 0.012% (최저) |

### 시간 패턴
| 패턴 | 특징 |
|------|------|
| 피크 요일 | 목요일 (전환율 7.2%로 최고) |
| 피크 시간 | 14-17시 |
| 활성 시간 | 10-18시 |
| 최적 윈도우 | 최근 40시간 내 View가 구매와 가장 높은 상관관계 |

---

## Approach 1: Heuristic Recommender (v5)

EDA에서 발견한 패턴을 규칙 기반으로 구현한 휴리스틱 추천기입니다.

### 추천 우선순위
```
1. Cart 아이템 (최우선) - 구매 전환율 12.69%
2. 반복 View 아이템 (2회+) - 강한 관심 신호
3. View 아이템 (가중치 점수순)
4. 인기 아이템 (Fallback)
```

### 다차원 가중치 시스템
| Feature | 조건 | 가중치 |
|---------|------|--------|
| 가격 | $0-50 (저가) | 1.5x |
| 가격 | $200-500 (중고가) | 1.2x |
| 가격 | $500+ (고가) | 0.8x |
| 요일 | 목요일 | 1.5x |
| 요일 | 금요일 | 1.3x |
| 요일 | 토요일 | 1.2x |
| 시간대 | 14-17시 (피크) | 1.2x |
| 시간대 | 10-18시 (활성) | 1.1x |
| 반복 View | 2회 | 1.5x |
| 반복 View | 3회 | 2.0x |
| 반복 View | 5회+ | 3.0x |

이 휴리스틱 시스템의 가중치들이 이후 XGBoost 모델의 Feature Engineering 기반이 되었습니다.

---

## Approach 2: XGBoost Learning to Rank

휴리스틱의 한계(수동 가중치 조정, 비선형 패턴 학습 불가)를 극복하기 위해 ML 모델로 전환했습니다.

### System Architecture
```
Step 1: Candidate Generation
   └── 유저당 최대 600개 후보 생성
       ├── Cart: 50개
       ├── Repeat View: 200개
       ├── Recent View: 200개
       └── Popular: 200개

Step 2: Feature Extraction
   └── 30개 Feature 추출

Step 3: XGBoost Reranking
   └── rank:ndcg 목적함수로 학습

Step 4: Top-10 Selection
   └── 상위 10개 최종 추천
```

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Objective | rank:ndcg |
| Device | CUDA (RTX 3090) |
| learning_rate | 0.05 |
| n_estimators | 5000 |
| max_depth | 8 |
| min_child_weight | 50 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| reg_alpha | 0.1 |
| reg_lambda | 1.0 |

### Feature Engineering (30 Features)

#### User-Item Interaction
| Feature | Description |
|---------|-------------|
| ui_view_cnt | 유저-아이템 조회 횟수 |
| ui_view_cnt_40h | 최근 40시간 내 조회 횟수 |
| repeat2/3/5 | 재조회 패턴 (2회/3회/5회 이상) |
| ui_cart_flag | 장바구니 추가 여부 |
| ui_last_hours_ago | 마지막 조회 후 경과 시간 |

#### Temporal
| Feature | Description |
|---------|-------------|
| ui_last_dow | 마지막 조회 요일 |
| ui_last_hour | 마지막 조회 시간대 |
| is_peak | 피크 시간대 여부 (14-17시) |
| is_active | 활성 시간대 여부 (10-18시) |
| is_thu/fri/sat | 요일 플래그 |

#### Item
| Feature | Description |
|---------|-------------|
| item_view_pop | 아이템 조회 인기도 |
| item_purchase_pop | 아이템 구매 인기도 |
| item_purchase_rate | 구매 전환율 |
| item_price | 아이템 가격 |
| price_bucket | 가격 구간 |
| price_bonus | 가격 보너스 (EDA 기반) |

#### Source & Score
| Feature | Description |
|---------|-------------|
| src_cart/repeat/recent/popular | 후보 생성 소스 |
| src_priority | 소스 우선순위 |
| v5_score | 휴리스틱 점수 (v5 시스템) |

#### Cluster (K-Means, k=50)
| Feature | Description |
|---------|-------------|
| item_cluster_id | 아이템 클러스터 ID |
| cluster_match_score | 클러스터 매칭 점수 |
| fine_category_match_score | Fine category 매칭 점수 |

### Feature Importance (Top 10)

| Rank | Feature | Importance (Gain) |
|------|---------|-------------------|
| 1 | src_popular | 874.89 |
| 2 | src_priority | 118.77 |
| 3 | v5_score | 81.03 |
| 4 | ui_last_hours_ago | 45.28 |
| 5 | item_purchase_pop | 22.64 |
| 6 | src_recent | 15.72 |
| 7 | repeat2 | 15.46 |
| 8 | ui_view_cnt | 14.57 |
| 9 | repeat3 | 12.07 |
| 10 | ui_view_cnt_40h | 5.79 |

**Key Insight:** 휴리스틱 점수(v5_score)가 3위로, EDA 기반 규칙이 ML에서도 유효함을 증명

---

## Training Statistics

| Metric | Value |
|--------|-------|
| Train users | 47,825 |
| Validation users | 12,175 |
| Train samples | 28,694,890 |
| Train positives | 8,881 (0.03%) |
| Best iteration | 115 |

## Output Statistics

| Metric | Value |
|--------|-------|
| Total recommendations | 6,382,570 |
| Users | 638,257 |
| Unique items recommended | 26,061 |
| Cold users | 817 (0.1%) |
| Avg recs per user | 10.0 |

## Execution Time

| Phase | Time |
|-------|------|
| 전체 실행 | 약 41분 |
| XGBoost 학습 | 약 1분 15초 |
| 추론 | 약 20분 |

---

## Key Takeaways

1. **EDA가 ML의 기반** - 가격/요일/시간대 패턴이 Feature로 직접 변환되어 모델 성능에 기여

2. **휴리스틱 → ML 점진적 발전** - v5 휴리스틱의 가중치 시스템이 Feature Importance 상위권 (v5_score 3위)

3. **Candidate Generation이 핵심** - src_popular가 874.89로 압도적 1위, 좋은 후보 집합이 정교한 리랭킹보다 중요

4. **GPU 가속의 효율성** - RTX 3090으로 2,900만 샘플 학습이 1분 15초 만에 완료

---

## Environment

- Python 3.x
- XGBoost with CUDA support
- RTX 3090 GPU (24GB VRAM)
- pandas, numpy, scikit-learn
# E-Commerce Recommendation System

이커머스 플랫폼의 사용자 행동 데이터를 활용한 추천 시스템 프로젝트

## 프로젝트 개요

### 목표
- **NDCG@10 최적화**: 사용자별 Top-10 아이템 추천 품질 향상
- **현재 최고 성능**: 0.1185 (SASRec v1)

### 문제 정의
사용자의 과거 행동(view, cart, purchase)을 기반으로 구매 가능성이 높은 아이템 10개를 추천하는 문제

---

## 데이터 소개

### 데이터셋 규모

| 항목 | 값 |
|------|-----|
| 총 상호작용 수 | 8,350,311 |
| 고유 사용자 수 | 638,257 |
| 고유 아이템 수 | 29,502 |
| 고유 세션 수 | 2,889,552 |
| 데이터 기간 | 2019-11-01 ~ 2020-02-29 (4개월) |
| Sparsity | 99.96% |

### 데이터 특성

#### 이벤트 타입 분포
```
view:     99.78%  (조회)
cart:      0.20%  (장바구니)
purchase:  0.02%  (구매)
```

#### 주요 도전 과제
1. **Cold-Start 문제**: 테스트 사용자의 52%가 학습 데이터에 없음
2. **극심한 Long-tail**: 상위 1% 아이템이 전체 상호작용의 25% 차지
3. **높은 희소성**: 99.96%의 user-item 조합이 비어있음
4. **짧은 세션**: 55%의 세션이 1개 아이템만 조회

### 피처 설명

| 컬럼명 | 설명 |
|--------|------|
| event_time | 이벤트 발생 시간 |
| event_type | view / cart / purchase |
| user_id | 사용자 고유 ID |
| user_session | 세션 ID |
| item_id | 아이템 고유 ID |
| category_code | 카테고리 (예: apparel.shoes) |
| brand | 브랜드명 |
| price | 가격 (USD) |

---

## 해결 방법

### 접근 전략

```
┌─────────────────────────────────────────────────────────┐
│  1. ALS (Alternating Least Squares)                     │
│     - Implicit feedback 기반 협업 필터링                 │
│     - Optuna로 하이퍼파라미터 최적화                      │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  2. User Segmentation                                   │
│     - Heavy (50+ interactions): 순수 개인화              │
│     - Medium (10-50): 개인화 + 약간의 인기도             │
│     - Light (<10): 인기도 가중치 강화                    │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  3. Popularity Boost                                    │
│     - Cold-start 사용자에게 인기 아이템 추천             │
│     - Light 사용자에게 인기도 기반 보정                  │
└─────────────────────────────────────────────────────────┘
```

### 핵심 기술

| 기술 | 설명 |
|------|------|
| **ALS** | implicit 라이브러리 기반 Matrix Factorization |
| **Event Weighting** | view=1, cart=10, purchase=20 가중치 |
| **User Segmentation** | 사용자 활동량 기반 차별화된 추천 전략 |
| **Popularity Fallback** | Cold-start 사용자 대응 |

---

## 실험 결과

### 성능 비교

| 모델 | 설정 | NDCG@10 | 변화 |
|------|------|---------|------|
| ALS Baseline | factors=32, reg=0.0215 | 0.0863 | baseline |
| Ontology v4 | ALS + Content + CoView | 0.0872 | +1.04% |
| Two-Stage | ALS + LightGBM Rerank | 0.0697 | -19.3% |
| Item2Vec | ALS + Word2Vec | 0.0858 | -0.58% |
| **ALS Opt v1** | **factors=64 + User Seg** | **0.0922** | **+6.8%** |
| ALS Time Decay | Decay Rate=0.01 | 0.0968 | +4.9% |
| ALS Time Decay | Decay Rate=0.005 | 0.0947 | -2.1% (vs 0.01) |
| ALS Time Decay | Decay Rate=0.02 | 0.1006 | +3.9% (vs 0.01) |
| ALS Time Decay | Decay Rate=0.05 | 0.1052 | +4.6% (vs 0.02) |
| **ALS Time Decay** | **Decay Rate=0.1** | **0.1067** | **+1.4% (Best)** |
| ALS Time Decay | Decay Rate=0.2 | 0.1052 | -1.4% (vs 0.1) |
| ALS Time Decay | Decay Rate=0.5 | 0.0985 | -6.4% (vs 0.1) |
| **SASRec v1** | **Transformer Seq (Ep 5)** | **0.1185** | **+11.0% (vs ALS)** |

### 성능 추이

```
NDCG@10
  │
0.105│                              ☆ Time Decay 0.05 (0.1052)
  │                                           ★ Time Decay 0.1 (0.1067)
0.100│                              ☆ Time Decay 0.2 (0.1052)
  │                              ☆ Time Decay 0.02 (0.1006)
0.095│                              ☆ Time Decay 0.01 (0.0968)
  │                              ☆ Time Decay 0.005 (0.0947)
  │                              ☆ Opt v1 (0.0922)
0.090│
  │  ● Baseline (0.0863)  ◆ Ontology v4 (0.0872)
0.085│                    ○ Item2Vec (0.0858)
  │
0.080│
  │
0.070│                          ✗ Two-Stage (0.0697)
  │___________________________________________
      12/10   12/12   12/13   12/14
```

### 주요 발견

1. **ALS 파라미터 튜닝이 가장 효과적**: factors 32→64로 +6% 이상 개선
2. **User Segmentation**: Light 사용자에게 인기도 기반 추천이 효과적
4. **Time Decay 최적점 발견**: Decay Rate 0.1에서 정점(0.1067)
5. **Deep Learning Shift**: 단순 협업 필터링(ALS)의 한계를 Transformer(SASRec)로 돌파. (+11%)
   - 순차적 패턴(Sequence) 학습의 중요성 입증.

---

## 프로젝트 구조

```
RecSys/
├── code/
│   ├── als_optimized.py      # Best 모델 (0.0922)
│   ├── ontology_recommender.py
│   ├── item2vec_recommender.py
│   ├── markov_recommender.py
│   ├── twostage_recommender.py
│   ├── optuna_als.py         # 하이퍼파라미터 튜닝
│   ├── train_als.py          # 기본 ALS
│   ├── eda/                  # 탐색적 데이터 분석
│   │   ├── eda_analysis.py
│   │   └── eda_insights.md
│   └── 251210.md ~ 251214.md # 실험 기록
├── data/                     # 데이터 (gitignore)
├── out/                      # 출력 파일 (gitignore)
└── README.md
```

---

## 실행 방법

### 환경 설정

```bash
pip install -r code/requirements.txt
```

### 추천 생성

```bash
# Best 모델 (ALS Optimized)
python code/als_optimized.py \
    --data_dir ./data/ \
    --output_dir ./out/ \
    --factors 64 \
    --regularization 0.01 \
    --alpha 10 \
    --iterations 20
```

### 하이퍼파라미터 튜닝

```bash
python code/optuna_als.py --n_trials 50
```

---

## 진행 과정

### 12/10 - 기초 모델 구축
- EDA 수행 및 데이터 특성 파악
- ALS 기본 모델 구현
- Optuna 하이퍼파라미터 튜닝 → 0.0863

### 12/11 - 앙상블 시도
- ALS + SASRec 앙상블 실험
- 인기도 기반 폴백 구현

### 12/12 - 온톨로지 기반 추천
- 카테고리/브랜드/가격 속성 활용
- Co-view 그래프 구축
- `filter_already_liked_items=False` 발견 → 0.0868

### 12/13 - Event Type 가중치
- view=1, cart=10, purchase=20 가중치 적용
- Ontology v4 → 0.0872

### 12/14 - 최적화 및 실험
- Item2Vec, Markov Chain, Two-Stage 실험 (실패)
- **ALS Optimized + User Segmentation → 0.0922**
- **Time Decay (Recency) 적용 → 0.0968 (Best so far)**
  - Decay=0.01 (0.0968) > Decay=0.005 (0.0947)
- **강력한 Time Decay (0.05) → 0.1052**
- **Time Decay 최적화 (0.1) → 0.1067 (Final Best)**
  - 0.2, 0.5 등 과도한 감쇠는 성능 하락 확인

### 12/15 - 딥러닝(Sequential) 도입
- **SASRec (Self-Attentive Sequential Recommendation) → 0.1185 (SOTA)**
  - ALS의 한계(단순 공기반)를 넘어 순서(Sequence) 맥락 파악 성공
  - Baseline(0.0863) 대비 **+37.3%** 대폭 향상

---

## 향후 계획 (Road to 0.3)

1. **SASRec 고도화 (Target: 0.15)**
   - Epoch 증가 (10~20)
   - Hidden Unit 확장 (64 -> 128)
   - Max Length 증가 (50 -> 100)
   - Pre-training (Next Item Prediction 외에 Masked Item Prediction 추가)

2. **Graph Neural Networks (Target: 0.20)**
   - LightGCN 도입: 유저-아이템 그래프 구조 학습
   - Multi-Modal: 이미지/텍스트 정보 통합

3. **Reranking & Ensemble (Target: 0.30)**
   - Real-time Reranking (Two-Tower)
   - ALS + SASRec + LightGCN 앙상블

---

## 기술 스택

- **Python 3.10+**
- **implicit**: ALS 모델
- **pandas, numpy**: 데이터 처리
- **scipy**: 희소 행렬
- **Optuna**: 하이퍼파라미터 최적화
- **gensim**: Item2Vec (Word2Vec)
- **tqdm**: 진행률 표시

---

## 참고 자료

- [implicit 라이브러리](https://github.com/benfred/implicit)
- [Optuna](https://optuna.org/)
- [NDCG 메트릭](https://en.wikipedia.org/wiki/Discounted_cumulative_gain)

---

## 라이선스

MIT License

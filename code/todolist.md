# RecSys 대회 To-Do List

## 현재 상태
- **Baseline (ALS)**: 0.0852
- **Optuna 튜닝 후**: 0.0863 (+1.3%)

---

## Phase 1: 단기 개선 (높은 ROI)

### 1-1. ALS 추가 튜닝
- [ ] num_factor 16~32 구간 세밀 탐색
- [ ] n_trials 50으로 확대 튜닝
- [ ] filter_already_liked_items=True 테스트

### 1-2. SASRec 튜닝
- [ ] epochs 50, 100으로 증가
- [ ] n_layers 3, 4 시도
- [ ] MAX_ITEM_LIST_LENGTH 100 테스트
- [ ] optuna_sasrec.py 작성

### 1-3. 앙상블 (기대 효과: +3~5%)
- [ ] ALS + SASRec Rank Fusion 구현
- [ ] Reciprocal Rank Fusion (RRF) 적용
- [ ] 가중치 최적화 (grid search)

---

## Phase 2: 중기 개선

### 2-1. 추가 모델 도입
- [ ] BERT4Rec 학습
- [ ] GRU4Rec 학습
- [ ] LightGCN (그래프 기반)

### 2-2. Feature Engineering
- [ ] 시간 decay 가중치 적용
- [ ] 인기도 기반 후처리
- [ ] 요일/시간대별 패턴 반영

### 2-3. 검증 체계 강화
- [ ] K-Fold 교차 검증 구현
- [ ] Local CV ↔ LB 상관관계 분석
- [ ] Hit Rate, MRR 메트릭 추가

---

## Phase 3: 장기 개선

### 3-1. 데이터 분석 심화
- [ ] Cold-start 사용자/아이템 비율 분석
- [ ] 세션 길이 분포 분석
- [ ] 구매 전환율 패턴 분석

### 3-2. 고급 앙상블
- [ ] Stacking 앙상블
- [ ] 다양성(Diversity) 고려한 Re-ranking
- [ ] 모델별 강점 분석 후 조합 최적화

---

## 완료된 항목

- [x] ALS Baseline 구현 (0.0852)
- [x] Optuna ALS 튜닝 (0.0863)
- [x] optuna_als.py 작성
- [x] optuna_01.md 실험 보고서 작성

---

## 우선순위 요약

| 순위 | 작업 | 기대 효과 | 난이도 |
|-----|------|----------|--------|
| 1 | 앙상블 (ALS + SASRec) | +3~5% | 중 |
| 2 | SASRec 튜닝 | +1~2% | 하 |
| 3 | ALS num_factor 16 탐색 | +0.5~1% | 하 |
| 4 | BERT4Rec 추가 | +1~2% | 중 |
| 5 | Feature Engineering | +1~3% | 상 |

---

## 실행 명령어 모음

```bash
# ALS 최적 파라미터
python train_als.py --num_factor 32 --regularization 0.021466 --alpha 7

# ALS 추가 튜닝 (num_factor 낮은 범위)
python optuna_als.py --n_trials 50

# SASRec 학습
python recbole_dataset.py
python train_sasrec.py
python inference_sasrec.py --model_file ./saved/SASRec-*.pth
```

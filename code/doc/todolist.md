# RecSys 대회 To-Do List

## 현재 상태
- **Baseline (ALS)**: 0.0852
- **Optuna 튜닝 후**: 0.0863 (+1.3%)
- **SASRec (Proxy Labeling)**: 0.1219 (+41.3%) 🚀
- **Ensemble (ALS+SASRec)**: 0.1187 (ALS가 오히려 점수를 깎음 📉)
- **Rerank (Hard Negative)**: 0.1178 (AUC는 높으나 실제 성능 하락 📉)
- **Time-Filter (30 days)**: 0.1132 (데이터 과도한 축소로 인한 정보 손실 📉)
- **Score Ensemble (SASRec 0.7 + ALS 0.3)**: 0.1354 (+57.1%) 🏆 NEW BEST
- **Hybrid Rerank (Co-view)**: 0.1104 (개인화 점수 왜곡으로 인한 하락 📉)
- **Score Ensemble (8:2)**: 0.1329 (SASRec 비중 과다 📉)
- **Score Ensemble (6:4)**: 0.1374 (+4.5% vs 7:3) 🏆 FINAL BEST

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

---

## 개발 환경 이슈 및 해결 (Environment Troubleshooting)

### 1. PyTorch MPS (Apple Silicon) 호환성 문제
- **증상**: SASRec 등 Transformer 기반 모델 실행 시 `NotImplementedError: The operator 'aten::_nested_tensor_from_mask_left_aligned' ...` 에러 발생.
- **원인**: PyTorch MPS 백엔드에서 아직 Transformer의 일부 마스킹 연산을 지원하지 않음.
- **해결책**: 실행 시 환경 변수 `PYTORCH_ENABLE_MPS_FALLBACK=1`을 설정하여 CPU 폴백을 활성화해야 함.
- **사용 예시**:
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python sasrec_proxy.py --epochs 20 --min_view 3
```
- **비고**: 이 설정을 적용하면 일부 연산이 CPU에서 돌아서 느려질 수 있으나, 실행 불가 상태는 해결됨.

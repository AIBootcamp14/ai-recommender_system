# RecSys Challenge 2025: 희소 데이터(Sparsity) 극복 실험 리포트

**최종 스코어:** 0.1354 (Baseline 대비 +57.1% 향상)
**핵심 문제:** 구매 데이터 비율 0.02% (Extreme Sparsity)

---

## 📅 실험 타임라인 (Chronological Order)

### 1. Baseline & Tuning (0.0863)
- **방법:** ALS (Alternating Least Squares)
- **성과:** Optuna로 파라미터 최적화 (`factors=32`, `alpha=7`)하여 **0.0863** 달성.
- **한계:** 단순 행렬 분해로는 유저의 "연속적인 행동 패턴"을 파악하지 못함.

### 2. SASRec Proxy Labeling (0.1219) 🚀
- **가설:** "구매(Purchase)만 정답으로 쓰기엔 데이터가 너무 적다. 자주 본(View >= 3) 아이템도 '관심'으로 간주하자."
- **구현:**
    - Target = `Purchase` + `Cart` + `View (count >= 3)`
    - 학습 데이터 약 5~10배 증강 효과.
- **결과:** **0.1219 (+41%)**. 대성공. 데이터의 양과 질이 모델 복잡도보다 중요함을 입증.

### 3. 시행착오 (Trial & Error)
- **Ensemble (RRF)**: 0.1187 (▼)
    - 단순 등수 합은 압도적인 SASRec의 성능을 오히려 깎아먹음.
- **Hard Negative Rerank**: 0.1178 (▼)
    - LightGBM이 0.99 AUC로 과적합됨. "쉬운 오답"만 구분하고 "어려운 오답" 구분 실패.
- **Time-Filtered Proxy**: 0.1132 (▼)
    - "최근 30일 데이터만 쓰자" -> 데이터 절대량 부족으로 실패. 희소 상황에선 오래된 데이터도 소중함.

### 4. Score Ensemble (0.1354) 🏆
- **전략:** 단순 등수 합(Rank)이 아니라, **확률값(Score) 가중 합산**.
- **조합:**
    - `SASRec Proxy` (지능형, 0.1219) × **0.7**
    - `ALS Time Decay` (전역형, 0.1067) × **0.3**
- **결과:** 상호 보완을 통해 **0.1354** 달성.

---

## 💡 승리 공식 (Winning Formula)

| 요인 | 설명 | 기여도 |
|------|------|--------|
| **Proxy Labeling** | 구매 외에 3회 이상 본 상품도 정답으로 간주 | ⭐⭐⭐⭐⭐ (Core) |
| **Score Ensemble** | 잘하는 모델(SASRec)에 가중치를 더 주어 결합 | ⭐⭐⭐⭐ |
| **ALS Time Decay** | 시간에 따른 인기도 감쇠를 반영하여 보조 모델 강화 | ⭐⭐⭐ |
| **MPS Fallback** | 맥북 GPU 이슈를 빠르게 우회하여 실험 속도 유지 | ⭐⭐ |

## 🛠️ 최종 코드 파일
- `sasrec_proxy.py`: 핵심 모델 (Proxy Labeling 적용)
- `als_time_decay.py`: 보조 모델 (Time Decay 적용)
- `score_ensemble.py`: 최종 결합 스크립트 (Weighted Softmax)

---

> **Insight:** 
> 희소 데이터(Sparse Data) 문제의 본질은 "모델 아키텍처"가 아니라 **"어떻게든 학습 신호(Signal)를 만들어내는 것"**에 있었습니다. 
> 0.02%의 정답에 매몰되지 않고, 99%의 로그 속에서 **"유의미한 가짜 정답(Proxy)"**을 찾아낸 것이 승리의 열쇠였습니다.

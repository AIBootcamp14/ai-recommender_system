# E-commerce Item Recommendation Baseline Code

이커머스 상품 추천 대회를 위한 베이스라인 코드입니다.

---

## 프로젝트 구조

```
RecSys/
├── code/
│   ├── train_als.py          # ALS 모델 학습 및 추론 (메인 실행 파일 1)
│   ├── recbole_dataset.py    # SASRec용 데이터셋 생성
│   ├── train_sasrec.py       # SASRec 모델 학습 (메인 실행 파일 2)
│   ├── inference_sasrec.py   # SASRec 모델 추론 (메인 실행 파일 3)
│   ├── optuna_als.py         # ALS 하이퍼파라미터 자동 튜닝
│   ├── utils.py              # 유틸리티 함수 (set_seed)
│   ├── requirements.txt      # 필요 패키지 목록
│   ├── yaml/
│   │   └── sasrec.yaml       # SASRec 하이퍼파라미터 설정
│   └── README.md
├── data/
│   ├── train.parquet         # 학습 데이터
│   └── sample_submission.csv # 제출 파일 샘플
└── output/
    └── output.csv            # 생성된 제출 파일
```

---

## Installation (환경 설정)

### 1. Python 버전 확인
```bash
python --version  # Python 3.10 이상 필요
```

### 2. 패키지 설치
```bash
cd code
pip install -r requirements.txt
```

---

## Quick Start (빠른 시작)

### Option A: ALS 모델 (추천 - 가장 간단)

**한 줄 명령어로 학습 + 추론 완료:**
```bash
cd code
python train_als.py
```

**출력:** `../output/output.csv`

---

### Option B: SASRec 모델 (3단계 필요)

#### Step 1. 데이터셋 준비
```bash
cd code
python recbole_dataset.py
```
**출력:** `../data/SASRec_dataset/SASRec_dataset.inter`, `user2idx.json`, `item2idx.json`

#### Step 2. 모델 학습
```bash
python train_sasrec.py
```
**출력:** `./saved/SASRec-{날짜}.pth`

#### Step 3. 추론
```bash
python inference_sasrec.py --model_file ./saved/SASRec-{날짜}.pth
```
**출력:** `../output/output.csv`

---

## 메인 실행 파일 상세 설명

### 1. `train_als.py` - ALS 모델 (학습 + 추론)

| 인자 | 기본값 | 설명 |
|-----|--------|------|
| `--data_dir` | `train.parquet` | 학습 데이터 파일명 |
| `--dir_path` | `../data/` | 데이터 디렉토리 경로 |
| `--output_dir` | `../output/` | 출력 디렉토리 경로 |
| `--num_factor` | `32` | Latent factor 수 |
| `--regularization` | `0.001` | 정규화 계수 |
| `--alpha` | `10` | Confidence 가중치 |
| `--seed` | `42` | 랜덤 시드 |

**예시:**
```bash
# 기본 실행
python train_als.py

# 하이퍼파라미터 변경
python train_als.py --num_factor 64 --regularization 0.01 --alpha 40
```

---

### 2. `recbole_dataset.py` - SASRec 데이터셋 생성

| 인자 | 기본값 | 설명 |
|-----|--------|------|
| `--data_dir` | `../data` | 데이터 디렉토리 경로 |
| `--train_dataset` | `train.parquet` | 학습 데이터 파일명 |
| `--seed` | `42` | 랜덤 시드 |

**예시:**
```bash
python recbole_dataset.py
```

---

### 3. `train_sasrec.py` - SASRec 모델 학습

| 인자 | 기본값 | 설명 |
|-----|--------|------|
| `--config_file` | `./yaml/sasrec.yaml` | 설정 파일 경로 |
| `--dataset` | `SASRec_dataset` | 데이터셋 이름 |
| `--seed` | `42` | 랜덤 시드 |

**예시:**
```bash
python train_sasrec.py
```

**주요 하이퍼파라미터 (yaml/sasrec.yaml):**
- `n_layers`: 2 (Transformer 레이어 수)
- `n_heads`: 4 (Attention 헤드 수)
- `hidden_dropout_prob`: 0.5
- `epochs`: 20
- `train_batch_size`: 4096
- `MAX_ITEM_LIST_LENGTH`: 50

---

### 4. `inference_sasrec.py` - SASRec 모델 추론

| 인자 | 기본값 | 설명 |
|-----|--------|------|
| `--model_file` | `./saved/SASRec.pth` | 학습된 모델 파일 경로 |
| `--data_dir` | `../data/` | 데이터 디렉토리 경로 |
| `--output_dir` | `../output/` | 출력 디렉토리 경로 |
| `--train_dataset` | `train.parquet` | 학습 데이터 파일명 |
| `--seed` | `42` | 랜덤 시드 |

**예시:**
```bash
# 모델 파일 경로 지정 필수
python inference_sasrec.py --model_file ./saved/SASRec-Jan-19-2024.pth
```

---

## 전체 실행 흐름 요약

```
┌─────────────────────────────────────────────────────────────┐
│                    Option A: ALS (권장)                      │
├─────────────────────────────────────────────────────────────┤
│  python train_als.py  →  ../output/output.csv               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Option B: SASRec                          │
├─────────────────────────────────────────────────────────────┤
│  Step 1: python recbole_dataset.py                          │
│              ↓                                               │
│  Step 2: python train_sasrec.py                             │
│              ↓                                               │
│  Step 3: python inference_sasrec.py --model_file ./saved/~  │
│              ↓                                               │
│         ../output/output.csv                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 제출 파일 형식

| 컬럼 | 설명 |
|-----|------|
| `user_id` | 사용자 ID (UUID) |
| `item_id` | 추천 아이템 ID (UUID) |

- 각 사용자당 10개의 아이템 추천
- 총 행 수: 6,382,571 (헤더 포함)

---

## 하이퍼파라미터 자동 튜닝 (Optuna)

### `optuna_als.py` - ALS 하이퍼파라미터 튜닝

NDCG@10 기준으로 최적의 하이퍼파라미터를 자동 탐색합니다.

| 인자 | 기본값 | 설명 |
|-----|--------|------|
| `--n_trials` | `20` | 탐색할 trial 수 |
| `--dir_path` | `../data/` | 데이터 경로 |
| `--seed` | `42` | 랜덤 시드 |

**탐색 범위:**

| 파라미터 | 범위 |
|---------|------|
| `num_factor` | 32 ~ 128 |
| `regularization` | 0.001 ~ 0.1 |
| `alpha` | 1 ~ 50 |
| `iterations` | 10 ~ 30 |

**실행:**
```bash
# 기본 실행 (20 trials)
python optuna_als.py

# 빠른 테스트 (10 trials)
python optuna_als.py --n_trials 10
```

**출력:**
- 콘솔에 각 trial 결과 출력
- `best_params.json` 저장
- 최적 파라미터로 `train_als.py` 실행 명령어 출력

### 튜닝 결과 (리더보드 점수)

| 설정 | 점수 |
|-----|------|
| 기본값 (alpha=10, reg=0.001) | 0.0852 |
| **Optuna 튜닝** (alpha=7, reg=0.0215) | **0.0863 (+1.3%)** |

**최적 파라미터로 실행:**
```bash
python train_als.py --num_factor 32 --regularization 0.021466 --alpha 7
```

> 자세한 튜닝 과정은 [optuna_01.md](optuna_01.md) 참조

---

## 성능 향상 팁

1. **하이퍼파라미터 튜닝**
   - `python optuna_als.py`로 자동 탐색
   - SASRec: `n_layers` (3, 4), `epochs` (50, 100)

2. **다른 모델 시도** (RecBole 지원)
   - BERT4Rec, GRU4Rec, LightGCN

3. **앙상블**
   - 여러 모델의 추천 결과 결합

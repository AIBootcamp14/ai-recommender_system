"""
Optuna 기반 ALS 하이퍼파라미터 튜닝
- Train/Valid 분할 (시간 기반 80:20)
- NDCG@10 최적화
- 결과: 콘솔 출력 + best_params.json
"""
import argparse
import json
import os

import numpy as np
import optuna
import pandas as pd
from scipy import sparse
from implicit.als import AlternatingLeastSquares
from utils import set_seed


def load_data(dir_path, data_dir):
    """데이터 로드 및 인덱싱"""
    train_df = pd.read_parquet(os.path.join(dir_path, data_dir))

    user2idx = {v: k for k, v in enumerate(train_df['user_id'].unique())}
    idx2user = {k: v for k, v in enumerate(train_df['user_id'].unique())}
    item2idx = {v: k for k, v in enumerate(train_df['item_id'].unique())}
    idx2item = {k: v for k, v in enumerate(train_df['item_id'].unique())}

    train_df['user_idx'] = train_df['user_id'].map(user2idx)
    train_df['item_idx'] = train_df['item_id'].map(item2idx)

    return train_df, user2idx, idx2user, item2idx, idx2item


def train_valid_split(df, train_ratio=0.8):
    """시간 기반 Train/Valid 분할"""
    df = df.sort_values('event_time')
    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:split_idx]
    valid = df.iloc[split_idx:]
    return train, valid


def create_sparse_matrix(df, n_users, n_items):
    """Sparse matrix 생성"""
    df_grouped = df.groupby(['user_idx', 'item_idx']).size().reset_index(name='count')
    return sparse.csr_matrix(
        (df_grouped['count'].values,
         (df_grouped['user_idx'].values, df_grouped['item_idx'].values)),
        shape=(n_users, n_items),
        dtype=np.float32
    )


def ndcg_at_k(actual, predicted, k=10):
    """NDCG@K 계산"""
    if len(actual) == 0:
        return 0.0

    dcg = 0.0
    for i, item in enumerate(predicted[:k]):
        if item in actual:
            dcg += 1.0 / np.log2(i + 2)

    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def recall_at_k(actual, predicted, k=10):
    """Recall@K 계산"""
    if len(actual) == 0:
        return 0.0
    hits = len(set(predicted[:k]) & set(actual))
    return hits / min(len(actual), k)


def evaluate(model, train_sparse, valid_df, n_users, k=10):
    """모델 평가 - NDCG@K, Recall@K"""
    # Valid set의 user-item pairs
    valid_items = valid_df.groupby('user_idx')['item_idx'].apply(set).to_dict()

    ndcg_scores = []
    recall_scores = []

    for user_idx, actual_items in valid_items.items():
        if user_idx >= n_users:
            continue

        # 추천 생성 (제출 환경과 동일하게 filter=False)
        try:
            recommended, _ = model.recommend(
                user_idx,
                train_sparse[user_idx],
                N=k,
                filter_already_liked_items=False
            )

            ndcg_scores.append(ndcg_at_k(actual_items, recommended, k))
            recall_scores.append(recall_at_k(actual_items, recommended, k))
        except:
            continue

    return np.mean(ndcg_scores) if ndcg_scores else 0.0, np.mean(recall_scores) if recall_scores else 0.0


def objective(trial, train_df, n_users, n_items):
    """Optuna objective 함수"""
    # 하이퍼파라미터 샘플링 (낮은 num_factor 탐색)
    num_factor = trial.suggest_int('num_factor', 8, 32, step=4)
    regularization = trial.suggest_float('regularization', 0.001, 0.1, log=True)
    alpha = trial.suggest_int('alpha', 1, 20)
    iterations = trial.suggest_int('iterations', 5, 15, step=5)

    # Train/Valid 분할
    train, valid = train_valid_split(train_df)

    # Sparse matrix 생성
    train_sparse = create_sparse_matrix(train, n_users, n_items)

    # 모델 학습
    model = AlternatingLeastSquares(
        factors=num_factor,
        regularization=regularization,
        alpha=alpha,
        iterations=iterations,
        use_gpu=False,
        random_state=42
    )
    model.fit(train_sparse)

    # 평가
    ndcg, recall = evaluate(model, train_sparse, valid, n_users, k=10)

    # 로깅
    trial.set_user_attr('recall@10', recall)
    print(f"  Trial {trial.number}: NDCG@10={ndcg:.4f}, Recall@10={recall:.4f}")

    return ndcg


def main():
    parser = argparse.ArgumentParser(description='ALS 하이퍼파라미터 튜닝 (Optuna)')
    parser.add_argument('--n_trials', type=int, default=20, help='Trial 수')
    parser.add_argument('--dir_path', type=str, default='../data/', help='데이터 경로')
    parser.add_argument('--data_dir', type=str, default='train.parquet', help='데이터 파일')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 50)
    print("ALS 하이퍼파라미터 튜닝 시작")
    print("=" * 50)

    # 데이터 로드
    print("\n[1/3] 데이터 로드 중...")
    train_df, user2idx, idx2user, item2idx, idx2item = load_data(args.dir_path, args.data_dir)
    n_users = len(user2idx)
    n_items = len(item2idx)
    print(f"  Users: {n_users:,}, Items: {n_items:,}")

    # Optuna Study 생성
    print(f"\n[2/3] Optuna 튜닝 시작 (n_trials={args.n_trials})")
    study = optuna.create_study(direction='maximize', study_name='als_tuning')
    study.optimize(
        lambda trial: objective(trial, train_df, n_users, n_items),
        n_trials=args.n_trials,
        show_progress_bar=True
    )

    # 결과 출력
    print("\n[3/3] 튜닝 완료!")
    print("=" * 50)
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"\nBest NDCG@10: {study.best_value:.4f}")

    # best_params.json 저장
    output = {
        'best_params': study.best_params,
        'best_ndcg': study.best_value,
        'n_trials': args.n_trials
    }
    with open('best_params.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\n결과 저장: best_params.json")

    # 실행 명령어 출력
    bp = study.best_params
    print("\n최적 파라미터로 실행:")
    print(f"python train_als.py --num_factor {bp['num_factor']} --regularization {bp['regularization']:.6f} --alpha {bp['alpha']}")


if __name__ == '__main__':
    main()

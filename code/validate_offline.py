"""
오프라인 NDCG 검증 파이프라인
- 제출 전 성능 예측
- filter_already_liked_items=False로 통일
- 다양한 설정 비교
"""
import argparse
import os
import numpy as np
import pandas as pd
from scipy import sparse
from implicit.als import AlternatingLeastSquares
from tqdm import tqdm


def load_data(dir_path, data_file):
    """데이터 로드 및 인덱싱"""
    train_df = pd.read_parquet(os.path.join(dir_path, data_file))

    user2idx = {v: k for k, v in enumerate(train_df['user_id'].unique())}
    idx2user = {k: v for k, v in enumerate(train_df['user_id'].unique())}
    item2idx = {v: k for k, v in enumerate(train_df['item_id'].unique())}
    idx2item = {k: v for k, v in enumerate(train_df['item_id'].unique())}

    train_df['user_idx'] = train_df['user_id'].map(user2idx)
    train_df['item_idx'] = train_df['item_id'].map(item2idx)

    return train_df, user2idx, idx2user, item2idx, idx2item


def time_based_split(df, test_ratio=0.2):
    """시간 기반 Train/Test 분할"""
    df = df.sort_values('event_time')
    split_idx = int(len(df) * (1 - test_ratio))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


def create_sparse_matrix(df, n_users, n_items, use_weights=False, weight_map=None):
    """Sparse matrix 생성"""
    if use_weights and weight_map:
        df = df.copy()
        df['weight'] = df['event_type'].map(weight_map).fillna(1)
        df_grouped = df.groupby(['user_idx', 'item_idx'])['weight'].sum().reset_index()
        values = df_grouped['weight'].values
    else:
        df_grouped = df.groupby(['user_idx', 'item_idx']).size().reset_index(name='count')
        values = df_grouped['count'].values

    return sparse.csr_matrix(
        (values,
         (df_grouped['user_idx'].values, df_grouped['item_idx'].values)),
        shape=(n_users, n_items),
        dtype=np.float32
    )


def ndcg_at_k(actual, predicted, k=10):
    """NDCG@K 계산 (Binary Relevance)"""
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


def hit_rate_at_k(actual, predicted, k=10):
    """Hit Rate@K (적어도 1개 맞춤)"""
    return 1.0 if len(set(predicted[:k]) & set(actual)) > 0 else 0.0


def evaluate_model(model, train_sparse, test_df, n_users, k=10):
    """모델 평가"""
    # Test set의 user-item pairs
    test_items = test_df.groupby('user_idx')['item_idx'].apply(set).to_dict()

    ndcg_scores = []
    recall_scores = []
    hit_rates = []

    for user_idx, actual_items in tqdm(test_items.items(), desc="Evaluating"):
        if user_idx >= n_users:
            continue

        try:
            recommended, _ = model.recommend(
                user_idx,
                train_sparse[user_idx],
                N=k,
                filter_already_liked_items=False  # 제출 환경과 동일
            )

            ndcg_scores.append(ndcg_at_k(actual_items, recommended, k))
            recall_scores.append(recall_at_k(actual_items, recommended, k))
            hit_rates.append(hit_rate_at_k(actual_items, recommended, k))
        except:
            continue

    return {
        'ndcg': np.mean(ndcg_scores) if ndcg_scores else 0.0,
        'recall': np.mean(recall_scores) if recall_scores else 0.0,
        'hit_rate': np.mean(hit_rates) if hit_rates else 0.0,
        'n_users': len(ndcg_scores)
    }


def run_experiment(train_df, n_users, n_items, als_params,
                   use_weights=False, weight_map=None, test_ratio=0.2):
    """단일 실험 실행"""
    # Train/Test 분할
    train, test = time_based_split(train_df, test_ratio)

    # Sparse matrix 생성
    train_sparse = create_sparse_matrix(train, n_users, n_items, use_weights, weight_map)

    # 모델 학습
    model = AlternatingLeastSquares(**als_params)
    model.fit(train_sparse)

    # 평가
    metrics = evaluate_model(model, train_sparse, test, n_users, k=10)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='오프라인 NDCG 검증')
    parser.add_argument('--dir_path', type=str, default='../data/')
    parser.add_argument('--data_file', type=str, default='train.parquet')
    parser.add_argument('--test_ratio', type=float, default=0.2)
    args = parser.parse_args()

    print("=" * 60)
    print("오프라인 NDCG 검증 파이프라인")
    print("=" * 60)

    # 데이터 로드
    print("\n[1/4] 데이터 로드 중...")
    train_df, user2idx, idx2user, item2idx, idx2item = load_data(args.dir_path, args.data_file)
    n_users = len(user2idx)
    n_items = len(item2idx)
    print(f"  Users: {n_users:,}, Items: {n_items:,}")
    print(f"  Interactions: {len(train_df):,}")

    # 기본 ALS 파라미터 (Optuna 최적값)
    als_params = {
        'factors': 32,
        'regularization': 0.0215,
        'alpha': 7,
        'iterations': 15,
        'use_gpu': False,
        'random_state': 42
    }

    # event_type 가중치
    weight_map = {
        'view': 1,
        'cart': 5,
        'purchase': 10
    }

    results = []

    # 실험 1: 기본 ALS (가중치 없음)
    print("\n[2/4] 실험 1: 기본 ALS (가중치 없음)")
    metrics = run_experiment(train_df, n_users, n_items, als_params,
                             use_weights=False, test_ratio=args.test_ratio)
    results.append({
        'experiment': 'ALS (no weight)',
        **metrics
    })
    print(f"  NDCG@10: {metrics['ndcg']:.4f}")
    print(f"  Recall@10: {metrics['recall']:.4f}")
    print(f"  Hit Rate@10: {metrics['hit_rate']:.4f}")

    # 실험 2: event_type 가중치 적용
    print("\n[3/4] 실험 2: event_type 가중치 (view=1, cart=5, purchase=10)")
    metrics = run_experiment(train_df, n_users, n_items, als_params,
                             use_weights=True, weight_map=weight_map,
                             test_ratio=args.test_ratio)
    results.append({
        'experiment': 'ALS (weighted)',
        **metrics
    })
    print(f"  NDCG@10: {metrics['ndcg']:.4f}")
    print(f"  Recall@10: {metrics['recall']:.4f}")
    print(f"  Hit Rate@10: {metrics['hit_rate']:.4f}")

    # 실험 3: 다른 가중치 조합
    print("\n[4/4] 실험 3: 더 강한 가중치 (view=1, cart=10, purchase=20)")
    weight_map_strong = {'view': 1, 'cart': 10, 'purchase': 20}
    metrics = run_experiment(train_df, n_users, n_items, als_params,
                             use_weights=True, weight_map=weight_map_strong,
                             test_ratio=args.test_ratio)
    results.append({
        'experiment': 'ALS (strong weight)',
        **metrics
    })
    print(f"  NDCG@10: {metrics['ndcg']:.4f}")
    print(f"  Recall@10: {metrics['recall']:.4f}")
    print(f"  Hit Rate@10: {metrics['hit_rate']:.4f}")

    # 결과 요약
    print("\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # 최고 점수
    best = results_df.loc[results_df['ndcg'].idxmax()]
    print(f"\n최고 NDCG: {best['experiment']} → {best['ndcg']:.4f}")

    # CSV 저장
    results_df.to_csv('offline_validation_results.csv', index=False)
    print("\n결과 저장: offline_validation_results.csv")


if __name__ == '__main__':
    main()

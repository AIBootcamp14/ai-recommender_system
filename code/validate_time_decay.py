"""
Phase 2: 시간 Decay 가중치 실험
- 최근 행동에 더 높은 가중치 적용
- event_type + time decay 조합
"""
import argparse
import os
import numpy as np
import pandas as pd
from scipy import sparse
from implicit.als import AlternatingLeastSquares
from tqdm import tqdm
import random


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


def apply_time_decay(df, half_life_days=7):
    """시간 decay 가중치 적용"""
    df = df.copy()
    max_time = df['event_time'].max()

    # 날짜 차이 계산 (초 단위 → 일 단위)
    time_diff = (max_time - df['event_time']).dt.total_seconds() / (24 * 3600)

    # 지수 감쇠: 2^(-t/half_life)
    df['time_weight'] = np.power(2, -time_diff / half_life_days)

    return df


def create_sparse_matrix(df, n_users, n_items, use_event_weight=False,
                         weight_map=None, use_time_decay=False, half_life_days=7):
    """Sparse matrix 생성 (event_type + time_decay 가중치)"""
    df = df.copy()

    # 기본 가중치
    df['weight'] = 1.0

    # event_type 가중치
    if use_event_weight and weight_map:
        df['weight'] = df['event_type'].map(weight_map).fillna(1)

    # 시간 decay 가중치
    if use_time_decay:
        df = apply_time_decay(df, half_life_days)
        df['weight'] = df['weight'] * df['time_weight']

    # Groupby 및 합산
    df_grouped = df.groupby(['user_idx', 'item_idx'])['weight'].sum().reset_index()
    values = df_grouped['weight'].values

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


def evaluate_model(model, train_sparse, test_df, n_users, k=10, sample_size=5000):
    """모델 평가 (샘플링)"""
    test_items = test_df.groupby('user_idx')['item_idx'].apply(set).to_dict()

    all_users = list(test_items.keys())
    if len(all_users) > sample_size:
        sampled_users = random.sample(all_users, sample_size)
    else:
        sampled_users = all_users

    ndcg_scores = []
    recall_scores = []
    hit_rates = []

    for user_idx in tqdm(sampled_users, desc="Evaluating"):
        if user_idx >= n_users:
            continue

        actual_items = test_items[user_idx]

        try:
            recommended, _ = model.recommend(
                user_idx,
                train_sparse[user_idx],
                N=k,
                filter_already_liked_items=False
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
                   use_event_weight=False, weight_map=None,
                   use_time_decay=False, half_life_days=7,
                   test_ratio=0.2, sample_size=5000):
    """단일 실험 실행"""
    train, test = time_based_split(train_df, test_ratio)

    train_sparse = create_sparse_matrix(
        train, n_users, n_items,
        use_event_weight, weight_map,
        use_time_decay, half_life_days
    )

    model = AlternatingLeastSquares(**als_params)
    model.fit(train_sparse)

    metrics = evaluate_model(model, train_sparse, test, n_users, k=10, sample_size=sample_size)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='시간 Decay 가중치 실험')
    parser.add_argument('--dir_path', type=str, default='../data/')
    parser.add_argument('--data_file', type=str, default='train.parquet')
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--sample_size', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("Phase 2: 시간 Decay 가중치 실험")
    print("=" * 60)

    # 데이터 로드
    print("\n[1/7] 데이터 로드 중...")
    train_df, user2idx, idx2user, item2idx, idx2item = load_data(args.dir_path, args.data_file)
    n_users = len(user2idx)
    n_items = len(item2idx)
    print(f"  Users: {n_users:,}, Items: {n_items:,}")
    print(f"  Interactions: {len(train_df):,}")
    print(f"  Sample size: {args.sample_size:,}")

    # ALS 파라미터
    als_params = {
        'factors': 32,
        'regularization': 0.0215,
        'alpha': 7,
        'iterations': 15,
        'use_gpu': False,
        'random_state': 42
    }

    # 최적 event_type 가중치 (Phase 1 결과)
    best_weight_map = {'view': 1, 'cart': 10, 'purchase': 20}

    results = []

    # 실험 1: 베이스라인 (가중치 없음)
    print("\n[2/7] 실험 1: 베이스라인 (가중치 없음)")
    metrics = run_experiment(train_df, n_users, n_items, als_params,
                             use_event_weight=False, use_time_decay=False,
                             test_ratio=args.test_ratio, sample_size=args.sample_size)
    results.append({'experiment': 'Baseline', **metrics})
    print(f"  NDCG@10: {metrics['ndcg']:.4f}")

    # 실험 2: event_type 가중치만 (최적값)
    print("\n[3/7] 실험 2: event_type 가중치 (v=1,c=10,p=20)")
    metrics = run_experiment(train_df, n_users, n_items, als_params,
                             use_event_weight=True, weight_map=best_weight_map,
                             use_time_decay=False,
                             test_ratio=args.test_ratio, sample_size=args.sample_size)
    results.append({'experiment': 'EventWeight Only', **metrics})
    print(f"  NDCG@10: {metrics['ndcg']:.4f}")

    # 실험 3: 시간 decay만 (half_life=7)
    print("\n[4/7] 실험 3: 시간 Decay만 (half_life=7일)")
    metrics = run_experiment(train_df, n_users, n_items, als_params,
                             use_event_weight=False, use_time_decay=True,
                             half_life_days=7,
                             test_ratio=args.test_ratio, sample_size=args.sample_size)
    results.append({'experiment': 'TimeDecay(7d) Only', **metrics})
    print(f"  NDCG@10: {metrics['ndcg']:.4f}")

    # 실험 4: event_type + 시간 decay (half_life=7)
    print("\n[5/7] 실험 4: event_type + 시간 Decay (half_life=7일)")
    metrics = run_experiment(train_df, n_users, n_items, als_params,
                             use_event_weight=True, weight_map=best_weight_map,
                             use_time_decay=True, half_life_days=7,
                             test_ratio=args.test_ratio, sample_size=args.sample_size)
    results.append({'experiment': 'Event+Time(7d)', **metrics})
    print(f"  NDCG@10: {metrics['ndcg']:.4f}")

    # 실험 5: event_type + 시간 decay (half_life=14)
    print("\n[6/7] 실험 5: event_type + 시간 Decay (half_life=14일)")
    metrics = run_experiment(train_df, n_users, n_items, als_params,
                             use_event_weight=True, weight_map=best_weight_map,
                             use_time_decay=True, half_life_days=14,
                             test_ratio=args.test_ratio, sample_size=args.sample_size)
    results.append({'experiment': 'Event+Time(14d)', **metrics})
    print(f"  NDCG@10: {metrics['ndcg']:.4f}")

    # 실험 6: event_type + 시간 decay (half_life=3)
    print("\n[7/7] 실험 6: event_type + 시간 Decay (half_life=3일)")
    metrics = run_experiment(train_df, n_users, n_items, als_params,
                             use_event_weight=True, weight_map=best_weight_map,
                             use_time_decay=True, half_life_days=3,
                             test_ratio=args.test_ratio, sample_size=args.sample_size)
    results.append({'experiment': 'Event+Time(3d)', **metrics})
    print(f"  NDCG@10: {metrics['ndcg']:.4f}")

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
    results_df.to_csv('time_decay_results.csv', index=False)
    print("\n결과 저장: time_decay_results.csv")


if __name__ == '__main__':
    main()

"""
오프라인 평가: Train 데이터를 시간 기준 분할하여 NDCG@10 측정
- Train: 처음 ~ 마지막 7일 전
- Test: 마지막 7일
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import os


def ndcg_at_k(recommended: list, actual: set, k: int = 10) -> float:
    """NDCG@K 계산"""
    if not actual:
        return 0.0

    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in actual:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0

    # Ideal DCG
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual), k)))

    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def evaluate_recommendations(recommendations: pd.DataFrame,
                            ground_truth: dict) -> dict:
    """추천 결과 평가"""
    user_recs = recommendations.groupby('user_id')['item_id'].apply(list).to_dict()

    ndcg_scores = []
    hit_scores = []

    for user_id, actual_items in tqdm(ground_truth.items(), desc="Evaluating"):
        if user_id in user_recs:
            recs = user_recs[user_id]

            # NDCG@10
            ndcg = ndcg_at_k(recs, actual_items, k=10)
            ndcg_scores.append(ndcg)

            # Hit@10
            hit = 1 if any(item in actual_items for item in recs[:10]) else 0
            hit_scores.append(hit)
        else:
            ndcg_scores.append(0.0)
            hit_scores.append(0)

    return {
        'NDCG@10': np.mean(ndcg_scores),
        'Hit@10': np.mean(hit_scores),
        'num_users': len(ground_truth)
    }


def load_and_split_data(data_path: str, test_days: int = 7):
    """시간 기준 Train/Test 분할"""
    print("데이터 로드 중...")
    df = pd.read_parquet(data_path)
    df['event_time'] = pd.to_datetime(df['event_time'])

    # 시간 기준 분할
    max_time = df['event_time'].max()
    split_time = max_time - pd.Timedelta(days=test_days)

    train_df = df[df['event_time'] < split_time].copy()
    test_df = df[df['event_time'] >= split_time].copy()

    print(f"Train: {len(train_df):,} rows ({train_df['event_time'].min()} ~ {train_df['event_time'].max()})")
    print(f"Test: {len(test_df):,} rows ({test_df['event_time'].min()} ~ {test_df['event_time'].max()})")

    # Ground Truth: 테스트 기간에 사용자가 상호작용한 아이템
    ground_truth = test_df.groupby('user_id')['item_id'].apply(set).to_dict()
    print(f"Ground Truth Users: {len(ground_truth):,}")

    return train_df, test_df, ground_truth


def evaluate_popularity_baseline(train_df: pd.DataFrame, ground_truth: dict, top_k: int = 10):
    """인기도 기반 베이스라인"""
    print("\n[Popularity Baseline]")

    # 인기 아이템
    popular_items = train_df.groupby('item_id').size().nlargest(top_k).index.tolist()

    # 모든 사용자에게 동일한 추천
    results = []
    for user_id in ground_truth.keys():
        for item_id in popular_items:
            results.append({'user_id': user_id, 'item_id': item_id})

    rec_df = pd.DataFrame(results)
    metrics = evaluate_recommendations(rec_df, ground_truth)

    print(f"NDCG@10: {metrics['NDCG@10']:.4f}")
    print(f"Hit@10: {metrics['Hit@10']:.4f}")

    return metrics


def evaluate_cars_offline(train_df: pd.DataFrame, ground_truth: dict, sample_size: int = 100000):
    """CARS 모델 오프라인 평가"""
    print("\n[CARS Model]")

    # CARS 모델 임포트
    from cars_recommender import CARSFeatureEngineering, CARSRecommender

    # Feature Engineering
    print("Feature Engineering...")
    feature_eng = CARSFeatureEngineering()
    feature_eng.fit(train_df)

    # 최근 30일 데이터로 학습
    train_df['event_time'] = pd.to_datetime(train_df['event_time'])
    recent_df = train_df[train_df['event_time'] >= train_df['event_time'].max() - pd.Timedelta(days=30)]

    # 모델 학습
    recommender = CARSRecommender(feature_eng)
    cars_train_df = recommender.prepare_training_data(recent_df, sample_size=sample_size)
    recommender.train(cars_train_df)

    # 추천 생성 (Ground Truth 사용자만)
    print("추천 생성...")
    users = np.array(list(ground_truth.keys()))
    last_event_time = train_df['event_time'].max()

    rec_df = recommender.generate_recommendations(
        users=users,
        event_time=last_event_time,
        top_k=10
    )

    # 평가
    metrics = evaluate_recommendations(rec_df, ground_truth)

    print(f"NDCG@10: {metrics['NDCG@10']:.4f}")
    print(f"Hit@10: {metrics['Hit@10']:.4f}")

    return metrics


def evaluate_als_offline(train_df: pd.DataFrame, ground_truth: dict):
    """ALS 모델 오프라인 평가 (배치 최적화)"""
    print("\n[ALS Model]")

    from implicit.als import AlternatingLeastSquares
    from scipy.sparse import csr_matrix

    # User/Item 인코딩
    users = train_df['user_id'].unique()
    items = train_df['item_id'].unique()

    user_to_idx = {u: i for i, u in enumerate(users)}
    item_to_idx = {i: idx for idx, i in enumerate(items)}
    idx_to_item = {idx: i for i, idx in item_to_idx.items()}

    # Interaction Matrix
    user_indices = train_df['user_id'].map(user_to_idx).values
    item_indices = train_df['item_id'].map(item_to_idx).values

    interaction_matrix = csr_matrix(
        (np.ones(len(train_df)), (user_indices, item_indices)),
        shape=(len(users), len(items))
    )

    # ALS 학습
    print("ALS 학습...")
    model = AlternatingLeastSquares(
        factors=64,
        regularization=0.01,
        alpha=10,
        iterations=20,
        random_state=42
    )
    model.fit(interaction_matrix)

    # 추천 생성 (배치 처리)
    print("추천 생성 (배치)...")
    results = []

    # Cold-start 인기 아이템 미리 계산
    popular_items = train_df.groupby('item_id').size().nlargest(10).index.tolist()

    # Known/Unknown 사용자 분리
    gt_users = list(ground_truth.keys())
    known_users = [u for u in gt_users if u in user_to_idx]
    unknown_users = [u for u in gt_users if u not in user_to_idx]

    print(f"  Known users: {len(known_users)}, Unknown (cold-start): {len(unknown_users)}")

    # Known 사용자: 배치 추천
    if known_users:
        known_user_indices = np.array([user_to_idx[u] for u in known_users])
        # 배치 추천 (전체 한 번에)
        item_ids_batch, scores_batch = model.recommend(
            known_user_indices,
            interaction_matrix[known_user_indices],
            N=10,
            filter_already_liked_items=False
        )

        for i, user_id in enumerate(tqdm(known_users, desc="Processing Known Users")):
            for item_idx in item_ids_batch[i]:
                results.append({'user_id': user_id, 'item_id': idx_to_item[item_idx]})

    # Unknown 사용자: 인기 아이템 추천
    for user_id in unknown_users:
        for item_id in popular_items:
            results.append({'user_id': user_id, 'item_id': item_id})

    rec_df = pd.DataFrame(results)

    # 평가
    metrics = evaluate_recommendations(rec_df, ground_truth)

    print(f"NDCG@10: {metrics['NDCG@10']:.4f}")
    print(f"Hit@10: {metrics['Hit@10']:.4f}")

    return metrics


def main():
    print("=" * 60)
    print("Offline Evaluation (Fast Mode)")
    print("=" * 60)

    data_path = '../data/train.parquet'

    # 데이터 분할
    train_df, test_df, ground_truth = load_and_split_data(data_path, test_days=7)

    # 평가할 사용자 샘플링 (빠른 평가를 위해 10000명으로 축소)
    sample_users = np.random.choice(list(ground_truth.keys()),
                                     min(10000, len(ground_truth)),
                                     replace=False)
    sampled_gt = {u: ground_truth[u] for u in sample_users}
    print(f"\n평가 사용자 수: {len(sampled_gt):,} (샘플링)")

    results = {}

    # 1. Popularity Baseline
    results['Popularity'] = evaluate_popularity_baseline(train_df, sampled_gt)

    # 2. ALS (빠른 버전)
    results['ALS'] = evaluate_als_offline(train_df, sampled_gt)

    # 결과 요약
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"{'Model':<15} {'NDCG@10':<10} {'Hit@10':<10}")
    print("-" * 35)
    for model_name, metrics in results.items():
        print(f"{model_name:<15} {metrics['NDCG@10']:.4f}     {metrics['Hit@10']:.4f}")


if __name__ == "__main__":
    main()

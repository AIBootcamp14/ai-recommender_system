"""
Phase 3: 가중치 그리드 서치
- ALS, Content, CoView 가중치 최적화
- Hybrid 앙상블 최적 조합 탐색
"""
import argparse
import os
import numpy as np
import pandas as pd
from scipy import sparse
from implicit.als import AlternatingLeastSquares
from tqdm import tqdm
import random
from collections import defaultdict


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


def create_sparse_matrix(df, n_users, n_items, weight_map=None):
    """Sparse matrix 생성 (event_type 가중치)"""
    df = df.copy()

    if weight_map:
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


def build_coview_matrix(df, n_items, min_coview=2):
    """Co-view matrix 생성 (세션 기반)"""
    # 세션 내 함께 본 아이템 쌍
    df_sorted = df.sort_values(['user_idx', 'event_time'])

    coview_counts = defaultdict(int)

    for user_idx, group in df_sorted.groupby('user_idx'):
        items = group['item_idx'].tolist()
        unique_items = list(set(items))

        for i in range(len(unique_items)):
            for j in range(i + 1, len(unique_items)):
                coview_counts[(unique_items[i], unique_items[j])] += 1
                coview_counts[(unique_items[j], unique_items[i])] += 1

    # Sparse matrix 생성
    rows, cols, data = [], [], []
    for (i, j), count in coview_counts.items():
        if count >= min_coview:
            rows.append(i)
            cols.append(j)
            data.append(count)

    coview_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)

    # L1 정규화
    row_sums = np.array(coview_matrix.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1
    coview_matrix = sparse.diags(1 / row_sums) @ coview_matrix

    return coview_matrix


def build_content_features(df, item2idx):
    """Content 기반 특성 생성"""
    item_category = {}
    item_brand = {}

    for _, row in df.drop_duplicates('item_idx').iterrows():
        item_idx = row['item_idx']
        item_category[item_idx] = row.get('category_code', '')
        item_brand[item_idx] = row.get('brand', '')

    return item_category, item_brand


def get_content_score(user_history, candidate_idx, item_category, item_brand, df):
    """Content 점수 계산"""
    if len(user_history) == 0:
        return 0.0

    # 후보 아이템 정보
    cand_cat = item_category.get(candidate_idx, '')
    cand_brand = item_brand.get(candidate_idx, '')

    # 사용자 선호 카테고리/브랜드 빈도
    user_cats = [item_category.get(i, '') for i in user_history]
    user_brands = [item_brand.get(i, '') for i in user_history]

    cat_match = sum(1 for c in user_cats if c == cand_cat and c != '') / max(len(user_cats), 1)
    brand_match = sum(1 for b in user_brands if b == cand_brand and b != '') / max(len(user_brands), 1)

    return cat_match * 0.6 + brand_match * 0.4


def get_coview_score(user_history, candidate_idx, coview_matrix):
    """Co-view 점수 계산"""
    if len(user_history) == 0:
        return 0.0

    scores = [coview_matrix[item, candidate_idx] for item in user_history
              if item < coview_matrix.shape[0]]

    return np.mean(scores) if scores else 0.0


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


def evaluate_hybrid(model, train_sparse, test_df, train_df, n_users,
                    coview_matrix, item_category, item_brand,
                    als_weight, content_weight, coview_weight,
                    k=10, sample_size=3000, n_candidates=100):
    """Hybrid 모델 평가"""
    test_items = test_df.groupby('user_idx')['item_idx'].apply(set).to_dict()
    user_history = train_df.groupby('user_idx')['item_idx'].apply(list).to_dict()

    all_users = list(test_items.keys())
    if len(all_users) > sample_size:
        sampled_users = random.sample(all_users, sample_size)
    else:
        sampled_users = all_users

    ndcg_scores = []

    for user_idx in tqdm(sampled_users, desc=f"Eval (ALS={als_weight:.1f})", leave=False):
        if user_idx >= n_users:
            continue

        actual_items = test_items[user_idx]
        history = user_history.get(user_idx, [])

        try:
            # ALS 후보 생성
            candidates, als_scores = model.recommend(
                user_idx,
                train_sparse[user_idx],
                N=n_candidates,
                filter_already_liked_items=False
            )

            # 점수 정규화
            als_scores_norm = (als_scores - als_scores.min()) / (als_scores.max() - als_scores.min() + 1e-8)

            # Hybrid 점수 계산
            final_scores = []
            for i, (cand, als_s) in enumerate(zip(candidates, als_scores_norm)):
                content_s = get_content_score(history, cand, item_category, item_brand, train_df)
                coview_s = get_coview_score(history, cand, coview_matrix)

                final = als_weight * als_s + content_weight * content_s + coview_weight * coview_s
                final_scores.append((cand, final))

            # 최종 정렬
            final_scores.sort(key=lambda x: -x[1])
            recommended = [x[0] for x in final_scores[:k]]

            ndcg_scores.append(ndcg_at_k(actual_items, recommended, k))

        except Exception as e:
            continue

    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def main():
    parser = argparse.ArgumentParser(description='가중치 그리드 서치')
    parser.add_argument('--dir_path', type=str, default='../data/')
    parser.add_argument('--data_file', type=str, default='train.parquet')
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--sample_size', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("Phase 3: 가중치 그리드 서치")
    print("=" * 60)

    # 데이터 로드
    print("\n[1/4] 데이터 로드 중...")
    train_df, user2idx, idx2user, item2idx, idx2item = load_data(args.dir_path, args.data_file)
    n_users = len(user2idx)
    n_items = len(item2idx)
    print(f"  Users: {n_users:,}, Items: {n_items:,}")

    # Train/Test 분할
    train, test = time_based_split(train_df, args.test_ratio)

    # 최적 event_type 가중치
    weight_map = {'view': 1, 'cart': 10, 'purchase': 20}
    train_sparse = create_sparse_matrix(train, n_users, n_items, weight_map)

    # ALS 학습
    print("\n[2/4] ALS 모델 학습...")
    als_params = {
        'factors': 32,
        'regularization': 0.0215,
        'alpha': 7,
        'iterations': 15,
        'use_gpu': False,
        'random_state': 42
    }
    model = AlternatingLeastSquares(**als_params)
    model.fit(train_sparse)

    # Co-view matrix 생성
    print("\n[3/4] Co-view matrix 생성...")
    coview_matrix = build_coview_matrix(train, n_items, min_coview=2)
    print(f"  Co-view pairs: {coview_matrix.nnz:,}")

    # Content 특성
    item_category, item_brand = build_content_features(train_df, item2idx)

    # 그리드 서치
    print("\n[4/4] 가중치 그리드 서치 시작...")
    print(f"  Sample size: {args.sample_size:,}")

    # 가중치 조합 (합=1 유지)
    weight_combinations = [
        (1.0, 0.0, 0.0),    # Pure ALS
        (0.9, 0.05, 0.05),  # ALS 강화
        (0.9, 0.1, 0.0),    # ALS + Content
        (0.9, 0.0, 0.1),    # ALS + CoView
        (0.8, 0.1, 0.1),    # 균형
        (0.8, 0.15, 0.05),  # Content 강화
        (0.8, 0.05, 0.15),  # CoView 강화
        (0.7, 0.2, 0.1),    # Content 더 강화
        (0.7, 0.1, 0.2),    # CoView 더 강화
        (0.7, 0.15, 0.15),  # 균형 (낮은 ALS)
        (0.6, 0.2, 0.2),    # 앙상블 강화
        (0.85, 0.1, 0.05),  # 약간 Content
        (0.85, 0.05, 0.1),  # 약간 CoView
    ]

    results = []

    for als_w, content_w, coview_w in weight_combinations:
        ndcg = evaluate_hybrid(
            model, train_sparse, test, train,
            n_users, coview_matrix, item_category, item_brand,
            als_w, content_w, coview_w,
            k=10, sample_size=args.sample_size
        )

        results.append({
            'als_weight': als_w,
            'content_weight': content_w,
            'coview_weight': coview_w,
            'ndcg': ndcg
        })

        print(f"  ALS={als_w:.2f}, Content={content_w:.2f}, CoView={coview_w:.2f} → NDCG={ndcg:.4f}")

    # 결과 요약
    print("\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('ndcg', ascending=False)
    print(results_df.to_string(index=False))

    # 최고 점수
    best = results_df.iloc[0]
    print(f"\n최고 NDCG: ALS={best['als_weight']:.2f}, Content={best['content_weight']:.2f}, "
          f"CoView={best['coview_weight']:.2f} → {best['ndcg']:.4f}")

    # CSV 저장
    results_df.to_csv('weight_grid_results.csv', index=False)
    print("\n결과 저장: weight_grid_results.csv")


if __name__ == '__main__':
    main()

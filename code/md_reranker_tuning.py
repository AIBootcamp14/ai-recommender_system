"""
MD Reranker 가중치 튜닝
Grid Search로 최적 boost/penalty 값 탐색
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import argparse
import itertools


def build_user_context(user_history):
    """유저의 구매 패턴 분석"""
    categories = user_history['category_code'].dropna().tolist()
    main_categories = list(dict.fromkeys(categories[-10:]))[:3]
    last_category = categories[-1] if categories else None
    brands = user_history['brand'].dropna().value_counts().head(3).index.tolist()

    prices = user_history['price'].dropna()
    if len(prices) > 0:
        median_price = prices.median()
        price_low = median_price * 0.5
        price_high = median_price * 1.5
    else:
        price_low, price_high = 0, float('inf')

    purchased = user_history[user_history['event_type'] == 'purchase']['category_code'].dropna().unique().tolist()

    main_category_l1 = set()
    for cat in main_categories:
        if cat and '.' in cat:
            main_category_l1.add(cat.split('.')[0])

    return {
        'main_categories': main_categories,
        'main_category_l1': main_category_l1,
        'last_category': last_category,
        'favorite_brands': brands,
        'price_low': price_low,
        'price_high': price_high,
        'purchased_categories': purchased
    }


def md_rerank_score(item_meta, user_context, baseline_rank, weights):
    """가중치 파라미터화된 MD 점수"""
    base_score = 1.0 / baseline_rank
    boost = 0
    penalty = 0

    item_cat = item_meta.get('category', 'unknown')
    item_brand = item_meta.get('brand', 'unknown')
    item_price = item_meta.get('price', 0)

    # 1. 카테고리 정확 매칭
    if item_cat in user_context['main_categories']:
        boost += weights['cat_exact']
    elif item_cat and '.' in item_cat:
        item_cat_l1 = item_cat.split('.')[0]
        if item_cat_l1 in user_context['main_category_l1']:
            boost += weights['cat_l1']

    # 2. 최근 관심 카테고리
    if item_cat == user_context['last_category']:
        boost += weights['last_cat']

    # 3. 브랜드 충성도
    if item_brand in user_context['favorite_brands']:
        boost += weights['brand']

    # 4. 가격 범위
    if user_context['price_low'] <= item_price <= user_context['price_high']:
        boost += weights['price_in']
    else:
        penalty += weights['price_out']

    # 5. 이미 구매한 카테고리
    if item_cat in user_context['purchased_categories']:
        penalty += weights['purchased']

    return base_score + boost - penalty


def rerank_candidates(candidates, user_context, item_info, weights):
    """후보 재정렬"""
    scored = []
    for rank, item_id in enumerate(candidates, 1):
        item_meta = item_info.get(item_id, {})
        score = md_rerank_score(item_meta, user_context, rank, weights)
        scored.append((item_id, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [item_id for item_id, score in scored]


def run_reranker(user_candidates, user_history_dict, item_info, weights):
    """전체 유저 재정렬 실행"""
    final_output = []
    for user_id in user_candidates.keys():
        original_candidates = user_candidates[user_id][:10]

        if user_id in user_history_dict:
            user_history = user_history_dict[user_id]
            user_context = build_user_context(user_history)
            reranked = rerank_candidates(original_candidates, user_context, item_info, weights)
        else:
            reranked = original_candidates

        for item_id in reranked:
            final_output.append({'user_id': user_id, 'item_id': item_id})

    return pd.DataFrame(final_output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--input', type=str, default='../out/output_ensemble_6_4.csv')
    parser.add_argument('--output_dir', type=str, default='../out/')
    args = parser.parse_args()

    print("="*60)
    print("MD Reranker - Grid Search Tuning")
    print("="*60)

    # Load data
    print("\n[1/3] Loading data...")
    train_df = pd.read_parquet(f"{args.data_dir}/train.parquet")
    baseline_df = pd.read_csv(args.input)

    # Build item metadata
    print("[2/3] Building metadata...")
    item_meta = train_df.groupby('item_id').agg({
        'category_code': 'first',
        'brand': 'first',
        'price': 'median'
    }).reset_index()

    item_info = {
        row['item_id']: {
            'category': row['category_code'] if pd.notna(row['category_code']) else 'unknown',
            'brand': row['brand'] if pd.notna(row['brand']) else 'unknown',
            'price': row['price'] if pd.notna(row['price']) else 0
        }
        for _, row in item_meta.iterrows()
    }

    user_candidates = defaultdict(list)
    for _, row in baseline_df.iterrows():
        user_candidates[row['user_id']].append(row['item_id'])

    user_history_dict = {uid: group for uid, group in train_df.groupby('user_id')}

    # Grid Search
    print("[3/3] Running Grid Search...")

    # 탐색할 가중치 조합
    param_grid = {
        'cat_exact': [0.3, 0.5, 0.7, 1.0],      # 현재 0.5
        'cat_l1': [0.1, 0.2, 0.3],               # 현재 0.2
        'last_cat': [0.2, 0.3, 0.5],             # 현재 0.3
        'brand': [0.1, 0.2, 0.3],                # 현재 0.2
        'price_in': [0.05, 0.1, 0.2],            # 현재 0.1
        'price_out': [0.1, 0.2, 0.3],            # 현재 0.2
        'purchased': [0.2, 0.3, 0.5],            # 현재 0.3
    }

    # 주요 파라미터만 먼저 탐색 (cat_exact, last_cat, purchased)
    key_params = {
        'cat_exact': [0.3, 0.5, 0.7, 1.0],
        'last_cat': [0.2, 0.3, 0.5],
        'purchased': [0.2, 0.3, 0.5],
    }

    # 나머지는 현재 값으로 고정
    fixed_params = {
        'cat_l1': 0.2,
        'brand': 0.2,
        'price_in': 0.1,
        'price_out': 0.2,
    }

    combinations = list(itertools.product(*key_params.values()))
    print(f"   Testing {len(combinations)} combinations...")

    results = []

    for i, (cat_exact, last_cat, purchased) in enumerate(tqdm(combinations, desc="Grid Search")):
        weights = {
            'cat_exact': cat_exact,
            'cat_l1': fixed_params['cat_l1'],
            'last_cat': last_cat,
            'brand': fixed_params['brand'],
            'price_in': fixed_params['price_in'],
            'price_out': fixed_params['price_out'],
            'purchased': purchased,
        }

        # 결과 저장
        out_df = run_reranker(user_candidates, user_history_dict, item_info, weights)

        config_name = f"cat{cat_exact}_last{last_cat}_purch{purchased}"
        output_path = f"{args.output_dir}/output_md_tune_{config_name}.csv"
        out_df.to_csv(output_path, index=False)

        results.append({
            'config': config_name,
            'cat_exact': cat_exact,
            'last_cat': last_cat,
            'purchased': purchased,
            'output': output_path
        })

    # 결과 요약
    print("\n" + "="*60)
    print("Grid Search Complete!")
    print(f"Generated {len(results)} submission files")
    print("\nFiles to submit:")
    for r in results[:5]:
        print(f"   {r['output']}")
    print("   ...")
    print("="*60)

    # 결과 저장
    pd.DataFrame(results).to_csv(f"{args.output_dir}/md_tuning_configs.csv", index=False)
    print(f"\nConfig list saved to: {args.output_dir}/md_tuning_configs.csv")


if __name__ == '__main__':
    main()

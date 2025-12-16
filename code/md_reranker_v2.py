"""
MD Reranker v2 - 가중치 튜닝 버전
명령줄에서 가중치 조정 가능
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import argparse


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--input', type=str, default='../out/output_ensemble_6_4.csv')
    parser.add_argument('--output', type=str, default='../out/output_md_v2.csv')

    # 가중치 파라미터
    parser.add_argument('--cat_exact', type=float, default=0.5)
    parser.add_argument('--cat_l1', type=float, default=0.2)
    parser.add_argument('--last_cat', type=float, default=0.3)
    parser.add_argument('--brand', type=float, default=0.2)
    parser.add_argument('--price_in', type=float, default=0.1)
    parser.add_argument('--price_out', type=float, default=0.2)
    parser.add_argument('--purchased', type=float, default=0.3)

    args = parser.parse_args()

    weights = {
        'cat_exact': args.cat_exact,
        'cat_l1': args.cat_l1,
        'last_cat': args.last_cat,
        'brand': args.brand,
        'price_in': args.price_in,
        'price_out': args.price_out,
        'purchased': args.purchased,
    }

    print("="*60)
    print("MD Reranker v2")
    print("="*60)
    print(f"\nWeights: {weights}")

    # Load data
    print("\n[1/4] Loading training data...")
    train_df = pd.read_parquet(f"{args.data_dir}/train.parquet")

    print("[2/4] Loading baseline submission...")
    baseline_df = pd.read_csv(args.input)

    # Build item metadata
    print("[3/4] Building item metadata...")
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

    # Rerank
    print("[4/4] Reranking...")
    final_output = []
    reranked_count = 0

    for user_id in tqdm(user_candidates.keys(), desc="Reranking"):
        original_candidates = user_candidates[user_id][:10]

        if user_id in user_history_dict:
            user_history = user_history_dict[user_id]
            user_context = build_user_context(user_history)
            reranked = rerank_candidates(original_candidates, user_context, item_info, weights)
            if reranked != original_candidates:
                reranked_count += 1
        else:
            reranked = original_candidates

        for item_id in reranked:
            final_output.append({'user_id': user_id, 'item_id': item_id})

    # Save
    out_df = pd.DataFrame(final_output)
    out_df.to_csv(args.output, index=False)

    print("\n" + "="*60)
    print(f"Reranked: {reranked_count:,} / {len(user_candidates):,} users")
    print(f"Saved to: {args.output}")
    print("="*60)


if __name__ == '__main__':
    main()

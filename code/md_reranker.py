"""
MD (Merchandising Director) 관점 재정렬
기존 모델 점수를 유지하면서 MD 로직으로 미세 조정

- 비용: $0
- 시간: ~5분 (전체 유저)
- LLM 없이 규칙 기반
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import argparse


def build_user_context(user_history):
    """유저의 구매 패턴 분석 (MD 관점)"""

    # 카테고리 선호도 (최근 행동 가중)
    categories = user_history['category_code'].dropna().tolist()
    main_categories = list(dict.fromkeys(categories[-10:]))[:3]  # 최근 10개 중 상위 3개

    # 마지막 본 카테고리
    last_category = categories[-1] if categories else None

    # 브랜드 충성도
    brands = user_history['brand'].dropna().value_counts().head(3).index.tolist()

    # 가격대 (중앙값 기준 ±50%)
    prices = user_history['price'].dropna()
    if len(prices) > 0:
        median_price = prices.median()
        price_low = median_price * 0.5
        price_high = median_price * 1.5
    else:
        price_low, price_high = 0, float('inf')

    # 이미 구매한 카테고리 (대체재 방지)
    purchased = user_history[user_history['event_type'] == 'purchase']['category_code'].dropna().unique().tolist()

    # 카테고리 depth-1 (대분류)
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


def md_rerank_score(item_meta, user_context, baseline_rank):
    """
    MD 관점 재정렬 점수

    baseline_rank: 기존 순위 (1이 가장 높음)
    반환: 최종 점수 (높을수록 좋음)
    """
    # 기존 점수 (순위의 역수)
    base_score = 1.0 / baseline_rank

    boost = 0
    penalty = 0

    item_cat = item_meta.get('category', 'unknown')
    item_brand = item_meta.get('brand', 'unknown')
    item_price = item_meta.get('price', 0)

    # 1. 카테고리 매칭 (가장 중요)
    if item_cat in user_context['main_categories']:
        boost += 0.5  # 정확 매칭
    elif item_cat and '.' in item_cat:
        item_cat_l1 = item_cat.split('.')[0]
        if item_cat_l1 in user_context['main_category_l1']:
            boost += 0.2  # 대분류 매칭

    # 2. 최근 관심 카테고리와 동일
    if item_cat == user_context['last_category']:
        boost += 0.3

    # 3. 브랜드 충성도
    if item_brand in user_context['favorite_brands']:
        boost += 0.2

    # 4. 가격 범위 체크
    if user_context['price_low'] <= item_price <= user_context['price_high']:
        boost += 0.1
    else:
        penalty += 0.2  # 가격 범위 밖

    # 5. 이미 구매한 카테고리 페널티 (대체재 방지)
    if item_cat in user_context['purchased_categories']:
        penalty += 0.3

    return base_score + boost - penalty


def rerank_candidates(candidates, user_context, item_info):
    """
    후보 아이템들을 MD 로직으로 재정렬

    candidates: [item_id, item_id, ...] (기존 순위대로)
    반환: 재정렬된 [item_id, ...]
    """
    scored = []
    for rank, item_id in enumerate(candidates, 1):
        item_meta = item_info.get(item_id, {})
        score = md_rerank_score(item_meta, user_context, rank)
        scored.append((item_id, score))

    # 점수 높은 순으로 정렬
    scored.sort(key=lambda x: x[1], reverse=True)

    return [item_id for item_id, score in scored]


def main():
    parser = argparse.ArgumentParser(description='MD 관점 재정렬')
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--input', type=str, default='../out/output_score_ensemble.csv',
                        help='기존 최고 성능 제출 파일')
    parser.add_argument('--output', type=str, default='../out/output_md_reranked.csv')
    args = parser.parse_args()

    print("="*60)
    print("MD (Merchandising Director) Reranker")
    print("="*60)

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
    print(f"   Built metadata for {len(item_info):,} items")

    # Group candidates by user
    print("[4/4] Reranking with MD logic...")
    user_candidates = defaultdict(list)
    for _, row in baseline_df.iterrows():
        user_candidates[row['user_id']].append(row['item_id'])

    # Build user history index for fast lookup
    user_history_dict = {uid: group for uid, group in train_df.groupby('user_id')}

    # Rerank
    final_output = []
    reranked_count = 0
    unchanged_count = 0

    for user_id in tqdm(user_candidates.keys(), desc="Reranking"):
        original_candidates = user_candidates[user_id][:10]

        # Get user history
        if user_id in user_history_dict:
            user_history = user_history_dict[user_id]
            user_context = build_user_context(user_history)

            # Rerank
            reranked = rerank_candidates(original_candidates, user_context, item_info)

            # Check if order changed
            if reranked != original_candidates:
                reranked_count += 1
            else:
                unchanged_count += 1
        else:
            # No history - keep original
            reranked = original_candidates
            unchanged_count += 1

        for item_id in reranked:
            final_output.append({'user_id': user_id, 'item_id': item_id})

    # Save
    out_df = pd.DataFrame(final_output)
    out_df.to_csv(args.output, index=False)

    print("\n" + "="*60)
    print("Results:")
    print(f"   Total users: {len(user_candidates):,}")
    print(f"   Reranked: {reranked_count:,} ({100*reranked_count/len(user_candidates):.1f}%)")
    print(f"   Unchanged: {unchanged_count:,} ({100*unchanged_count/len(user_candidates):.1f}%)")
    print(f"   Saved to: {args.output}")
    print("="*60)


if __name__ == '__main__':
    main()

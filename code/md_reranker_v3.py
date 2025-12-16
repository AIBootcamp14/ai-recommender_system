"""
MD Reranker v3 - 고급 규칙 추가
- 보완재 추천 (Complementary Items)
- 다양성 보장 (Diversity)
- 반복 구매 패턴 (Repeat Purchase)
- 스타일 매칭 (Style Matching)
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
import argparse


# 보완재 관계 정의 (신발/의류 도메인)
COMPLEMENTARY_MAP = {
    # 신발 → 의류
    'apparel.shoes': ['apparel.costume'],
    'apparel.shoes.keds': ['apparel.costume', 'apparel.shoes.slipons'],
    'apparel.shoes.slipons': ['apparel.costume', 'apparel.shoes.keds'],
    'apparel.shoes.boots': ['apparel.costume'],
    # 의류 → 신발
    'apparel.costume': ['apparel.shoes', 'apparel.shoes.keds'],
}

# 유사 스타일 그룹 (같은 그룹 내 다양성 제한)
STYLE_GROUPS = {
    'casual_shoes': ['apparel.shoes.keds', 'apparel.shoes.slipons', 'apparel.shoes.sneakers'],
    'formal_shoes': ['apparel.shoes.boots', 'apparel.shoes.loafers'],
    'athletic': ['apparel.shoes.sport'],
}


def build_user_context_v3(user_history):
    """유저의 구매 패턴 분석 (v3 확장)"""
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

    # 구매 이력 분석
    purchases = user_history[user_history['event_type'] == 'purchase']
    purchased_categories = purchases['category_code'].dropna().tolist()
    purchased_items = purchases['item_id'].unique().tolist()

    # 카테고리별 구매 횟수 (반복 구매 패턴)
    category_purchase_count = Counter(purchased_categories)

    # 대분류 추출
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
        'purchased_categories': list(set(purchased_categories)),
        'purchased_items': purchased_items,
        'category_purchase_count': category_purchase_count,
        'total_purchases': len(purchased_items),
    }


def get_complementary_categories(category):
    """보완재 카테고리 반환"""
    if category in COMPLEMENTARY_MAP:
        return COMPLEMENTARY_MAP[category]
    # L1 카테고리로 매칭 시도
    if category and '.' in category:
        l1 = '.'.join(category.split('.')[:2])
        if l1 in COMPLEMENTARY_MAP:
            return COMPLEMENTARY_MAP[l1]
    return []


def get_style_group(category):
    """카테고리의 스타일 그룹 반환"""
    for group_name, cats in STYLE_GROUPS.items():
        if category in cats:
            return group_name
    return None


def md_rerank_score_v3(item_meta, user_context, baseline_rank, weights, selected_categories):
    """
    v3 MD 점수 계산
    - 기존 로직 + 보완재/다양성/반복구매
    """
    base_score = 1.0 / baseline_rank
    boost = 0
    penalty = 0

    item_cat = item_meta.get('category', 'unknown')
    item_brand = item_meta.get('brand', 'unknown')
    item_price = item_meta.get('price', 0)

    # === 기존 규칙 ===

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

    # === 새로운 규칙 (v3) ===

    # 5. 보완재 추천 부스트
    last_cat = user_context['last_category']
    if last_cat:
        complementary = get_complementary_categories(last_cat)
        if item_cat in complementary:
            boost += weights['complementary']

    # 6. 반복 구매 패턴 (같은 카테고리 재구매 유도)
    cat_count = user_context['category_purchase_count'].get(item_cat, 0)
    if cat_count >= 2:
        boost += weights['repeat_purchase']  # 2번 이상 구매한 카테고리
    elif cat_count == 1:
        boost += weights['repeat_purchase'] * 0.5  # 1번 구매한 카테고리

    # 7. 다양성 페널티 (이미 선택된 같은 카테고리가 많으면 페널티)
    same_cat_count = selected_categories.get(item_cat, 0)
    if same_cat_count >= 3:
        penalty += weights['diversity'] * 2  # 3개 이상이면 강한 페널티
    elif same_cat_count >= 2:
        penalty += weights['diversity']  # 2개면 약한 페널티

    # 8. 스타일 그룹 다양성
    item_style = get_style_group(item_cat)
    if item_style:
        style_count = sum(1 for cat in selected_categories if get_style_group(cat) == item_style)
        if style_count >= 4:
            penalty += weights['style_diversity']

    return base_score + boost - penalty


def rerank_candidates_v3(candidates, user_context, item_info, weights):
    """v3 후보 재정렬 (순차적으로 다양성 고려)"""
    scored = []
    selected_categories = Counter()

    for rank, item_id in enumerate(candidates, 1):
        item_meta = item_info.get(item_id, {})
        score = md_rerank_score_v3(item_meta, user_context, rank, weights, selected_categories)
        scored.append((item_id, score, item_meta.get('category', 'unknown')))

    # 점수순 정렬
    scored.sort(key=lambda x: x[1], reverse=True)

    # 다양성을 위한 재정렬 (Greedy Selection)
    final_items = []
    final_categories = Counter()

    for item_id, score, cat in scored:
        # 이미 3개 이상 같은 카테고리가 있으면 스킵 (다른 아이템 우선)
        if final_categories[cat] >= 3 and len(final_items) < 8:
            continue
        final_items.append(item_id)
        final_categories[cat] += 1
        if len(final_items) >= 10:
            break

    # 10개 미만이면 나머지 채우기
    if len(final_items) < 10:
        for item_id, score, cat in scored:
            if item_id not in final_items:
                final_items.append(item_id)
                if len(final_items) >= 10:
                    break

    return final_items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--input', type=str, default='../out/output_ensemble_6_4.csv')
    parser.add_argument('--output', type=str, default='../out/output_md_v3.csv')

    # 기존 가중치
    parser.add_argument('--cat_exact', type=float, default=0.5)
    parser.add_argument('--cat_l1', type=float, default=0.2)
    parser.add_argument('--last_cat', type=float, default=0.3)
    parser.add_argument('--brand', type=float, default=0.2)
    parser.add_argument('--price_in', type=float, default=0.1)
    parser.add_argument('--price_out', type=float, default=0.2)

    # v3 새로운 가중치
    parser.add_argument('--complementary', type=float, default=0.15)
    parser.add_argument('--repeat_purchase', type=float, default=0.2)
    parser.add_argument('--diversity', type=float, default=0.1)
    parser.add_argument('--style_diversity', type=float, default=0.05)

    args = parser.parse_args()

    weights = {
        'cat_exact': args.cat_exact,
        'cat_l1': args.cat_l1,
        'last_cat': args.last_cat,
        'brand': args.brand,
        'price_in': args.price_in,
        'price_out': args.price_out,
        'complementary': args.complementary,
        'repeat_purchase': args.repeat_purchase,
        'diversity': args.diversity,
        'style_diversity': args.style_diversity,
    }

    print("=" * 60)
    print("MD Reranker v3 - Advanced Rules")
    print("=" * 60)
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
    print("[4/4] Reranking with v3 rules...")
    final_output = []
    reranked_count = 0
    diversity_applied = 0

    for user_id in tqdm(user_candidates.keys(), desc="Reranking"):
        original_candidates = user_candidates[user_id][:10]

        if user_id in user_history_dict:
            user_history = user_history_dict[user_id]
            user_context = build_user_context_v3(user_history)
            reranked = rerank_candidates_v3(original_candidates, user_context, item_info, weights)

            if reranked != original_candidates:
                reranked_count += 1
                # 다양성 적용 체크
                orig_cats = [item_info.get(i, {}).get('category') for i in original_candidates]
                new_cats = [item_info.get(i, {}).get('category') for i in reranked]
                if len(set(new_cats)) > len(set(orig_cats)):
                    diversity_applied += 1
        else:
            reranked = original_candidates

        for item_id in reranked:
            final_output.append({'user_id': user_id, 'item_id': item_id})

    # Save
    out_df = pd.DataFrame(final_output)
    out_df.to_csv(args.output, index=False)

    print("\n" + "=" * 60)
    print(f"Reranked: {reranked_count:,} / {len(user_candidates):,} users")
    print(f"Diversity improved: {diversity_applied:,} users")
    print(f"Saved to: {args.output}")
    print("=" * 60)


if __name__ == '__main__':
    main()

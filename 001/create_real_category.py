# create_real_category.py
# LLM을 사용하여 브랜드 기반 실제 카테고리 생성
# 2단계 접근: 1) 카테고리 체계 제안 받기 2) 전체 브랜드 분류

import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# .env 파일 로드
load_dotenv()

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 경로 설정
DATA_DIR = "../../data"
TRAIN_PATH = os.path.join(DATA_DIR, "train.parquet")
OUTPUT_PATH = os.path.join(DATA_DIR, "new_train.parquet")
CATEGORY_CACHE_PATH = "./brand_category_cache.json"
CATEGORY_SCHEMA_PATH = "./category_schema.json"

# 설정
MODEL = "gpt-4o-mini"  # 비용 효율적
BATCH_SIZE = 100  # 한 번에 분류할 브랜드 수


def load_brand_stats():
    """브랜드별 통계 로드"""
    print("[1/5] Loading brand statistics...")
    df = pd.read_parquet(TRAIN_PATH)

    brand_stats = df.groupby('brand').agg({
        'price': ['mean', 'median', 'min', 'max'],
        'item_id': 'nunique',
        'user_id': 'nunique'
    }).reset_index()
    brand_stats.columns = ['brand', 'avg_price', 'median_price', 'min_price', 'max_price', 'unique_items', 'unique_users']

    # 가격 tier 추가
    brand_stats['price_tier'] = pd.cut(
        brand_stats['avg_price'],
        bins=[0, 20, 50, 100, 200, 500, float('inf')],
        labels=['very_low(<20)', 'low(20-50)', 'mid(50-100)', 'mid-high(100-200)', 'high(200-500)', 'premium(500+)']
    )

    print(f"  Total brands: {len(brand_stats):,}")
    print(f"  Price range: {brand_stats['avg_price'].min():.2f} ~ {brand_stats['avg_price'].max():.2f}")

    return df, brand_stats


def step1_propose_category_schema(brand_stats, sample_size=200):
    """1단계: LLM에게 카테고리 체계 제안받기"""
    print("\n[2/5] Step 1: Proposing category schema...")

    # 이미 캐시가 있으면 로드
    if os.path.exists(CATEGORY_SCHEMA_PATH):
        with open(CATEGORY_SCHEMA_PATH, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        print(f"  Loaded cached schema: {len(schema['categories'])} categories")
        return schema

    # 다양한 가격대에서 샘플링
    samples = []
    for tier in brand_stats['price_tier'].unique():
        tier_brands = brand_stats[brand_stats['price_tier'] == tier]
        n_sample = min(len(tier_brands), sample_size // 6)
        sampled = tier_brands.sample(n=n_sample, random_state=42)
        for _, row in sampled.iterrows():
            samples.append(f"{row['brand']} (avg ${row['avg_price']:.0f}, {row['unique_items']} items)")

    sample_text = "\n".join(samples[:sample_size])

    prompt = f"""You are an e-commerce category expert. I have an e-commerce dataset with {len(brand_stats)} brands.
Here's a sample of brands with their average prices and item counts:

{sample_text}

Based on these brands, please propose a hierarchical category schema that would work well for product recommendations.

Requirements:
1. Create 15-25 main categories (not too broad, not too narrow)
2. Each main category can have 2-5 subcategories if needed
3. Categories should be mutually exclusive
4. Consider both product type AND price positioning
5. Include a catch-all "Other" category

Please respond in JSON format:
{{
    "categories": [
        {{
            "id": "sports_footwear",
            "name": "Sports & Athletic Footwear",
            "description": "Athletic shoes, running shoes, training shoes",
            "subcategories": ["running", "basketball", "training", "soccer"],
            "typical_brands": ["nike", "adidas", "puma"],
            "price_range": "mid to high"
        }},
        ...
    ]
}}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={"type": "json_object"}
    )

    schema = json.loads(response.choices[0].message.content)

    # 캐시 저장
    with open(CATEGORY_SCHEMA_PATH, 'w', encoding='utf-8') as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)

    print(f"  Proposed {len(schema['categories'])} categories")
    for cat in schema['categories']:
        print(f"    - {cat['id']}: {cat['name']}")

    return schema


def step2_classify_brands(brand_stats, schema):
    """2단계: 전체 브랜드를 제안된 체계로 분류"""
    print("\n[3/5] Step 2: Classifying all brands...")

    # 캐시 로드
    if os.path.exists(CATEGORY_CACHE_PATH):
        with open(CATEGORY_CACHE_PATH, 'r', encoding='utf-8') as f:
            brand_categories = json.load(f)
        print(f"  Loaded {len(brand_categories)} cached classifications")
    else:
        brand_categories = {}

    # 카테고리 ID 목록
    category_ids = [cat['id'] for cat in schema['categories']]
    category_descriptions = "\n".join([
        f"- {cat['id']}: {cat['name']} - {cat['description']}"
        for cat in schema['categories']
    ])

    # 미분류 브랜드 필터링
    unclassified = brand_stats[~brand_stats['brand'].isin(brand_categories.keys())]
    print(f"  Brands to classify: {len(unclassified)}")

    if len(unclassified) == 0:
        return brand_categories

    # 배치 처리
    batches = [unclassified.iloc[i:i+BATCH_SIZE] for i in range(0, len(unclassified), BATCH_SIZE)]

    for batch_idx, batch in enumerate(batches):
        print(f"  Processing batch {batch_idx + 1}/{len(batches)} ({len(batch)} brands)...")

        brand_list = "\n".join([
            f"{row['brand']} | avg_price: ${row['avg_price']:.0f} | items: {row['unique_items']} | users: {row['unique_users']}"
            for _, row in batch.iterrows()
        ])

        prompt = f"""Classify these brands into the provided categories.

Available categories:
{category_descriptions}

Brands to classify (format: brand | avg_price | items | users):
{brand_list}

For each brand, respond with ONLY the category_id. If unsure, use "other".

Respond in JSON format:
{{
    "classifications": {{
        "brand_name": "category_id",
        ...
    }}
}}
"""

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            classifications = result.get('classifications', {})

            # 결과 검증 및 저장
            for brand, category in classifications.items():
                if category in category_ids:
                    brand_categories[brand] = category
                else:
                    brand_categories[brand] = "other"

            # 중간 저장
            with open(CATEGORY_CACHE_PATH, 'w', encoding='utf-8') as f:
                json.dump(brand_categories, f, indent=2, ensure_ascii=False)

            print(f"    Classified {len(classifications)} brands")

        except Exception as e:
            print(f"    Error in batch {batch_idx + 1}: {e}")
            time.sleep(5)
            continue

        # Rate limiting
        time.sleep(1)

    return brand_categories


def create_new_train(df, brand_categories, schema):
    """new_train.parquet 생성"""
    print("\n[4/5] Creating new_train.parquet...")

    # 카테고리 이름 매핑
    category_name_map = {cat['id']: cat['name'] for cat in schema['categories']}
    category_name_map['other'] = 'Other'

    # 브랜드 -> 카테고리 매핑
    df['real_category'] = df['brand'].map(brand_categories)

    # 미분류 브랜드는 'other'로
    df['real_category'] = df['real_category'].fillna('other')

    # 카테고리 이름으로 변환 (옵션)
    # df['real_category_name'] = df['real_category'].map(category_name_map)

    # 통계 출력
    print(f"  Total rows: {len(df):,}")
    print(f"  Category distribution:")
    cat_dist = df['real_category'].value_counts()
    for cat, count in cat_dist.head(15).items():
        pct = count / len(df) * 100
        print(f"    {cat:30s}: {count:>10,} ({pct:5.1f}%)")
    if len(cat_dist) > 15:
        print(f"    ... and {len(cat_dist) - 15} more categories")

    # 저장
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\n  Saved to {OUTPUT_PATH}")

    return df


def print_summary(df, brand_categories, schema):
    """최종 요약"""
    print("\n[5/5] Summary")
    print("=" * 60)

    print(f"\nTotal brands classified: {len(brand_categories)}")
    print(f"Total categories: {len(schema['categories'])}")
    print(f"Output file: {OUTPUT_PATH}")

    # 카테고리별 브랜드 수
    cat_brand_count = {}
    for brand, cat in brand_categories.items():
        cat_brand_count[cat] = cat_brand_count.get(cat, 0) + 1

    print("\nBrands per category:")
    for cat, count in sorted(cat_brand_count.items(), key=lambda x: -x[1]):
        print(f"  {cat:30s}: {count:4d} brands")

    # 샘플 확인
    print("\nSample classifications:")
    sample_brands = ['nike', 'adidas', 'gucci', 'zara', 'samsung', 'apple', 'ikea', 'garnier']
    for brand in sample_brands:
        if brand in brand_categories:
            print(f"  {brand} -> {brand_categories[brand]}")


def main():
    print("=" * 60)
    print("Brand Category Creation using LLM")
    print("=" * 60)

    # 1. 데이터 로드
    df, brand_stats = load_brand_stats()

    # 2. 카테고리 체계 제안받기
    schema = step1_propose_category_schema(brand_stats)

    # 3. 전체 브랜드 분류
    brand_categories = step2_classify_brands(brand_stats, schema)

    # 4. new_train.parquet 생성
    df = create_new_train(df, brand_categories, schema)

    # 5. 요약
    print_summary(df, brand_categories, schema)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()

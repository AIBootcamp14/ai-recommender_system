"""
LLM-based Contextual Filter using Solar API (Upstage)
Best submission (0.1374) + LLM semantic coherence check
"""

import pandas as pd
import numpy as np
import os
import requests
from dotenv import load_dotenv
from collections import defaultdict
from tqdm import tqdm
import argparse
import time

# Load environment variables
load_dotenv()

class SolarContextFilter:
    def __init__(self, api_key):
        self.api_key = api_key
        self.endpoint = "https://api.upstage.ai/v1/solar/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
    def call_solar(self, prompt, max_tokens=200, user_category=None):
        """Solar API 호출 with MD Expert Role"""
        
        # MD 역할 맞춤형 시스템 메시지
        if user_category and user_category != 'unknown':
            # 카테고리별 전문 MD
            system_msg = f"""You are a 30-year veteran Merchandising Director (MD) specialized in {user_category} products.

Your expertise:
- Deep understanding of customer purchase psychology in {user_category}
- Knowledge of complementary vs substitute products
- Awareness of seasonal trends and brand positioning
- Ability to spot "out-of-context" recommendations instantly

Task: Review the recommended items and identify ANY that seem inconsistent with the customer's clear interest pattern."""
        else:
            # 일반 베테랑 MD
            system_msg = """You are a 30-year veteran Chief Merchandising Officer (CMO) of a major e-commerce platform.

Your expertise:
- Cross-category product knowledge (electronics, fashion, home, sports, etc.)
- Customer behavior pattern recognition
- Understanding of purchase journey (browsing → comparison → decision)
- Ability to distinguish between "exploratory browsing" vs "purchase intent"

Task: Review the recommended items and identify ANY that don't align with the customer's demonstrated interests."""
        
        payload = {
            "model": "solar-1-mini-chat",
            "messages": [
                {
                    "role": "system",
                    "content": system_msg
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.3  # 낮은 temperature로 일관성 확보
        }
        
        try:
            response = requests.post(
                self.endpoint,
                headers=self.headers,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            print(f"Solar API Error: {e}")
            return None
    
    def analyze_user_context(self, user_history):
        """유저 행동 패턴 분석"""
        categories = user_history['category_code'].value_counts().head(3).index.tolist()
        brands = user_history['brand'].value_counts().head(3).index.tolist()
        avg_price = user_history['price'].mean()
        
        context = {
            'main_categories': [c for c in categories if pd.notna(c)],
            'favorite_brands': [b for b in brands if pd.notna(b)],
            'avg_price': avg_price
        }
        
        return context
    
    def rerank_candidates(self, user_context, candidates, item_info):
        """
        LLM을 사용하여 구매 확률 순으로 재정렬 (NDCG 최적화)

        user_context: 유저의 관심사 요약
        candidates: [(item_id, score), ...]
        item_info: {item_id: {'category': ..., 'brand': ..., 'price': ...}, ...}
        """
        # Get user's main category for specialized MD role
        user_main_category = user_context['main_categories'][0] if user_context['main_categories'] else 'general'
        sub_category = user_main_category.split('.')[-1] if '.' in user_main_category else user_main_category

        # 가격대 범위 계산
        avg_price = user_context['avg_price']
        price_low = avg_price * 0.5
        price_high = avg_price * 1.5

        # 추천 아이템 목록 포맷팅
        candidates_desc = ""
        for i, (item_id, score) in enumerate(candidates[:10], 1):
            meta = item_info.get(item_id, {})
            cat = meta.get('category', 'unknown')
            sub_cat = cat.split('.')[-1] if cat and '.' in cat else cat
            brand = meta.get('brand', 'unknown')
            price = meta.get('price', 0)
            candidates_desc += f"{i}. [{sub_cat}] {brand} - ${price:.0f}\n"

        # NDCG 최적화 프롬프트 - 순서가 핵심!
        prompt = f"""You are a 30-year veteran MD. Your goal: MAXIMIZE purchase conversion.

## Customer Profile:
- Focus: {sub_category}
- Budget: ${price_low:.0f} - ${price_high:.0f}
- Preferred brands: {', '.join(user_context['favorite_brands'][:3]) if user_context['favorite_brands'] else 'Various'}

## Purchase Priority Rules (STRICT ORDER):
1. EXACT category match + preferred brand + price in range → TOP
2. EXACT category match + price in range → HIGH
3. Related category (complementary) + price in range → MEDIUM
4. Price out of range OR unrelated category → LOW

## Current Recommendations:
{candidates_desc}

## Task:
Reorder items 1-10 by PURCHASE PROBABILITY (highest first).
Return ONLY numbers in order, comma-separated.

Example: 3,1,5,2,4,6,7,8,9,10

Your ranking:"""

        # Call Solar API
        response = self.call_solar(prompt, max_tokens=50, user_category=user_main_category)

        if response is None:
            return candidates[:10]

        # Parse reranking
        new_order = self._parse_ranking(response)

        # Apply reranking
        try:
            reranked = []
            seen = set()
            for idx in new_order:
                if 1 <= idx <= len(candidates) and idx not in seen:
                    reranked.append(candidates[idx - 1])
                    seen.add(idx)
            # 누락된 아이템 추가
            for i, item in enumerate(candidates[:10], 1):
                if i not in seen:
                    reranked.append(item)
            return reranked[:10]
        except:
            return candidates[:10]

    def _parse_ranking(self, response):
        """LLM 응답에서 순위 파싱"""
        import re
        # "3,1,5,2,4,6,7,8,9,10" 또는 "3, 1, 5, 2, 4, 6, 7, 8, 9, 10" 형식
        numbers = re.findall(r'\d+', response)
        return [int(n) for n in numbers[:10]]

    def filter_candidates(self, user_context, candidates, item_info):
        """
        LLM을 사용하여 맥락에 안 맞는 아이템 필터링 (Legacy)
        """
        # 새로운 rerank 방식 사용
        return self.rerank_candidates(user_context, candidates, item_info)
    
    def _parse_outliers(self, response):
        """LLM 응답에서 이질적 아이템 번호 추출"""
        if "None" in response or "none" in response:
            return set()
        
        # "3, 7" 같은 형식 파싱
        import re
        numbers = re.findall(r'\d+', response)
        return set(int(n) for n in numbers)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--input', type=str, default='../out/output_ensemble_6_4.csv')
    parser.add_argument('--output', type=str, default='../out/output_llm_filtered.csv')
    parser.add_argument('--sample_users', type=int, default=100,
                        help='Number of users to filter (for cost control)')
    args = parser.parse_args()
    
    # Load API key
    api_key = os.getenv('UPSTAGE_API_KEY') or os.getenv('SOLAR_API_KEY')
    if not api_key or api_key == 'your_solar_api_key_here':
        print("ERROR: Please set SOLAR_API_KEY in .env file")
        return
    
    print("="*60)
    print("LLM Contextual Filter (Solar API)")
    print("="*60)
    
    # Initialize filter
    llm_filter = SolarContextFilter(api_key)
    
    # Load data
    print("Loading training data...")
    train_df = pd.read_parquet(os.path.join(args.data_dir, 'train.parquet'))
    
    print("Loading baseline submission...")
    baseline_df = pd.read_csv(args.input)
    
    # Build item info (optimized - groupby instead of loop)
    print("Building item metadata...")
    item_meta = train_df.groupby('item_id').agg({
        'category_code': 'first',
        'brand': 'first',
        'price': 'median'
    }).reset_index()
    item_info = {
        row['item_id']: {
            'category': row['category_code'] if pd.notna(row['category_code']) else 'unknown',
            'brand': row['brand'] if pd.notna(row['brand']) else 'unknown',
            'price': row['price']
        }
        for _, row in item_meta.iterrows()
    }
    print(f"Built metadata for {len(item_info)} items")
    
    # Group candidates by user
    user_candidates = defaultdict(list)
    for _, row in baseline_df.iterrows():
        user_candidates[row['user_id']].append(row['item_id'])
    
    # Sample users (for cost control)
    sampled_users = list(user_candidates.keys())[:args.sample_users]
    
    print(f"Filtering {len(sampled_users)} users with Solar API...")
    
    final_output = []
    
    for user_id in tqdm(sampled_users):
        # Get user history
        user_history = train_df[train_df['user_id'] == user_id]
        
        if len(user_history) == 0:
            # No history → use baseline
            for item in user_candidates[user_id][:10]:
                final_output.append({'user_id': user_id, 'item_id': item})
            continue
        
        # Analyze context
        user_context = llm_filter.analyze_user_context(user_history)
        
        # Get candidates with scores (inverse rank)
        candidates = [(item, 1.0 / (i+1)) for i, item in enumerate(user_candidates[user_id])]
        
        # LLM filter
        filtered = llm_filter.filter_candidates(user_context, candidates, item_info)
        
        # Output
        for item_id, score in filtered:
            final_output.append({'user_id': user_id, 'item_id': item_id})
        
        # Rate limiting (Solar API)
        time.sleep(0.1)
    
    # For remaining users, use baseline
    print("Adding remaining users (baseline)...")
    for user_id in user_candidates.keys():
        if user_id not in sampled_users:
            for item in user_candidates[user_id][:10]:
                final_output.append({'user_id': user_id, 'item_id': item})
    
    # Save
    out_df = pd.DataFrame(final_output)
    out_df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")
    print(f"Filtered {len(sampled_users)} users, used baseline for {len(user_candidates) - len(sampled_users)} users")

if __name__ == '__main__':
    main()

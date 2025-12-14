"""
SolarPro LLM for Recommendation Enhancement
- 카테고리 연관 지식 생성
- 아이템 임베딩 생성
- 추천 이유 생성

Upstage SolarPro API 사용
"""
import os
import json
import argparse
import pandas as pd
import numpy as np
from openai import OpenAI
from typing import List, Dict, Optional
from tqdm import tqdm
import pickle
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


class SolarProRecommender:
    """SolarPro 기반 추천 증강 시스템"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("UPSTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("UPSTAGE_API_KEY 환경변수를 설정하세요")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.upstage.ai/v1/solar"
        )
        self.model = "solar-pro"

    # ==================== 카테고리 연관 지식 ====================

    def generate_category_associations(self, categories: List[str]) -> Dict:
        """카테고리 간 보완재/대체재 관계 생성"""
        prompt = f"""You are an e-commerce expert. Given these product categories:
{json.dumps(categories, indent=2)}

For each category, identify:
1. Complementary categories (products often bought together)
2. Substitute categories (alternative products)

Return JSON format:
{{
  "category_name": {{
    "complementary": ["cat1", "cat2"],
    "substitute": ["cat3"]
  }}
}}

Focus on realistic shopping behavior. Only use categories from the provided list."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000
        )

        try:
            result = response.choices[0].message.content
            # JSON 추출
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0]
            elif "```" in result:
                result = result.split("```")[1].split("```")[0]
            return json.loads(result)
        except:
            return {"error": result}

    # ==================== 아이템 설명 생성 ====================

    def generate_item_description(self, item_info: Dict) -> str:
        """아이템 설명 생성 (임베딩용)"""
        prompt = f"""Based on the following product information:
- Category: {item_info.get('category', 'unknown')}
- Brand: {item_info.get('brand', 'unknown')}
- Frequently viewed with: {item_info.get('co_viewed', [])}

Write a concise 2-sentence description focusing on:
1. What type of customer would buy this
2. What complementary products they might need

Keep it factual and shopping-focused."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=150
        )

        return response.choices[0].message.content.strip()

    def batch_generate_descriptions(self, items: List[Dict], output_path: str = None) -> Dict[str, str]:
        """배치로 아이템 설명 생성"""
        descriptions = {}

        for item in tqdm(items, desc="Generating descriptions"):
            item_id = item['item_id']
            try:
                desc = self.generate_item_description(item)
                descriptions[item_id] = desc
            except Exception as e:
                descriptions[item_id] = f"Product in {item.get('category', 'unknown')} category"
                print(f"Error for {item_id}: {e}")

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(descriptions, f, indent=2)
            print(f"Saved to {output_path}")

        return descriptions

    # ==================== 임베딩 생성 ====================

    def generate_embedding(self, text: str) -> List[float]:
        """텍스트 임베딩 생성 (Solar Embedding)"""
        response = self.client.embeddings.create(
            model="embedding-passage",  # Upstage Solar Embedding for passages
            input=text
        )
        return response.data[0].embedding

    def batch_generate_embeddings(self, texts: Dict[str, str], output_path: str = None) -> Dict[str, List[float]]:
        """배치로 임베딩 생성"""
        embeddings = {}

        for item_id, text in tqdm(texts.items(), desc="Generating embeddings"):
            try:
                emb = self.generate_embedding(text)
                embeddings[item_id] = emb
            except Exception as e:
                print(f"Error for {item_id}: {e}")

        if output_path:
            with open(output_path, 'wb') as f:
                pickle.dump(embeddings, f)
            print(f"Saved embeddings to {output_path}")

        return embeddings

    # ==================== 추천 이유 생성 ====================

    def generate_recommendation_reason(self, user_history: List[Dict], recommended_item: Dict) -> str:
        """추천 이유 생성"""
        history_summary = ", ".join([
            f"{h['category']} ({h['brand']})" for h in user_history[:5]
        ])

        prompt = f"""A customer has viewed these products: {history_summary}

We are recommending: {recommended_item['category']} by {recommended_item['brand']}

Write a brief, personalized recommendation reason (1 sentence) explaining why this product fits their interests."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100
        )

        return response.choices[0].message.content.strip()

    # ==================== 구매 의도 분석 ====================

    def analyze_purchase_intent(self, session_items: List[Dict]) -> Dict:
        """세션 내 구매 의도 분석"""
        items_desc = "\n".join([
            f"- {item['category']} ({item['brand']})" for item in session_items
        ])

        prompt = f"""Analyze this shopping session:
{items_desc}

Predict:
1. purchase_likelihood: high/medium/low
2. likely_purchase_category: most likely category to purchase
3. complementary_needs: what else might they need

Return JSON format."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )

        try:
            result = response.choices[0].message.content
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0]
            elif "```" in result:
                result = result.split("```")[1].split("```")[0]
            return json.loads(result)
        except:
            return {"raw_response": result}


# ==================== 유틸리티 함수 ====================

def load_category_data(data_path: str) -> List[str]:
    """데이터에서 카테고리 목록 추출"""
    df = pd.read_parquet(data_path)
    categories = df['category_code'].dropna().unique().tolist()
    return sorted(categories)


def load_top_items(data_path: str, top_k: int = 1000) -> List[Dict]:
    """상위 인기 아이템 정보 추출"""
    df = pd.read_parquet(data_path)

    # 상위 아이템
    item_counts = df.groupby('item_id').agg({
        'category_code': 'first',
        'brand': 'first',
        'user_id': 'count'
    }).reset_index()
    item_counts.columns = ['item_id', 'category', 'brand', 'view_count']
    item_counts = item_counts.sort_values('view_count', ascending=False).head(top_k)

    return item_counts.to_dict('records')


def compute_similarity(emb1: List[float], emb2: List[float]) -> float:
    """코사인 유사도 계산"""
    emb1 = np.array(emb1)
    emb2 = np.array(emb2)
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


# ==================== 메인 ====================

def main():
    parser = argparse.ArgumentParser(description='SolarPro Recommender')
    parser.add_argument('--data_path', type=str, default='../data/train.parquet')
    parser.add_argument('--output_dir', type=str, default='./llm_output')
    parser.add_argument('--mode', type=str, choices=['categories', 'descriptions', 'embeddings', 'demo'],
                        default='demo', help='실행 모드')
    parser.add_argument('--top_k', type=int, default=100, help='상위 아이템 수')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("SolarPro Recommender")
    print("=" * 60)

    recommender = SolarProRecommender()

    if args.mode == 'categories':
        # 카테고리 연관 지식 생성
        print("\n[1] Loading categories...")
        categories = load_category_data(args.data_path)
        print(f"  Found {len(categories)} categories")

        print("\n[2] Generating category associations...")
        associations = recommender.generate_category_associations(categories)

        output_path = os.path.join(args.output_dir, 'category_associations.json')
        with open(output_path, 'w') as f:
            json.dump(associations, f, indent=2)
        print(f"  Saved to {output_path}")

    elif args.mode == 'descriptions':
        # 아이템 설명 생성
        print("\n[1] Loading top items...")
        items = load_top_items(args.data_path, args.top_k)
        print(f"  Found {len(items)} items")

        print("\n[2] Generating descriptions...")
        output_path = os.path.join(args.output_dir, 'item_descriptions.json')
        descriptions = recommender.batch_generate_descriptions(items, output_path)

    elif args.mode == 'embeddings':
        # 임베딩 생성 (설명 파일 필요)
        desc_path = os.path.join(args.output_dir, 'item_descriptions.json')
        if not os.path.exists(desc_path):
            print(f"Error: {desc_path} not found. Run with --mode descriptions first.")
            return

        print("\n[1] Loading descriptions...")
        with open(desc_path) as f:
            descriptions = json.load(f)
        print(f"  Loaded {len(descriptions)} descriptions")

        print("\n[2] Generating embeddings...")
        output_path = os.path.join(args.output_dir, 'item_embeddings.pkl')
        embeddings = recommender.batch_generate_embeddings(descriptions, output_path)

    elif args.mode == 'demo':
        # 데모 모드
        print("\n[Demo] Category Association")
        print("-" * 40)

        sample_categories = [
            "apparel.shoes",
            "apparel.shoes.sandals",
            "apparel.shoes.keds",
            "accessories.bag",
            "electronics.smartphone"
        ]

        associations = recommender.generate_category_associations(sample_categories)
        print(json.dumps(associations, indent=2))

        print("\n[Demo] Item Description")
        print("-" * 40)

        sample_item = {
            'category': 'apparel.shoes.sandals',
            'brand': 'nike',
            'co_viewed': ['socks', 'shoe care']
        }

        description = recommender.generate_item_description(sample_item)
        print(f"Item: {sample_item}")
        print(f"Description: {description}")

        print("\n[Demo] Recommendation Reason")
        print("-" * 40)

        user_history = [
            {'category': 'apparel.shoes.sandals', 'brand': 'nike'},
            {'category': 'apparel.shoes.keds', 'brand': 'adidas'},
            {'category': 'accessories.bag', 'brand': 'samsonite'}
        ]

        recommended = {'category': 'apparel.shoes', 'brand': 'puma'}

        reason = recommender.generate_recommendation_reason(user_history, recommended)
        print(f"User history: {user_history}")
        print(f"Recommended: {recommended}")
        print(f"Reason: {reason}")


if __name__ == '__main__':
    main()

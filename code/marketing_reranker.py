"""
Marketing-Aware Reranker
물류 제외, 순수 마케팅 전략만 적용하여 GMV 최적화

전략:
1. 고객 세분화 (신규/VIP)
2. Up-Sell (가격대 상향)
3. 브랜드 전략
4. 카테고리별 마진 추정
"""

import pandas as pd
import numpy as np
import argparse
import os
from tqdm import tqdm
from collections import defaultdict

# 카테고리별 예상 마진율 (업계 평균 기준)
CATEGORY_MARGIN = {
    'electronics': 0.15,      # 전자제품: 마진 낮음
    'apparel': 0.45,          # 의류: 마진 높음
    'accessories': 0.50,      # 악세서리: 마진 매우 높음
    'appliances': 0.25,       # 가전: 마진 중간
    'computers': 0.18,        # 컴퓨터: 마진 낮음
    'construction': 0.30,     # 건축/인테리어
    'furniture': 0.40,        # 가구
    'auto': 0.20,            # 자동차용품
    'default': 0.30          # 기본값
}

# 프리미엄 브랜드 리스트 (예시)
PREMIUM_BRANDS = ['samsung', 'apple', 'lg', 'sony', 'nike', 'adidas']

class MarketingReranker:
    def __init__(self, train_df):
        self.train_df = train_df
        
        # 유저별 구매 이력 분석
        self.user_stats = self._build_user_stats()
        
        # 아이템별 통계
        self.item_stats = self._build_item_stats()
        
    def _build_user_stats(self):
        """유저별 통계 구축"""
        print("Building user statistics...")
        stats = {}
        
        user_groups = self.train_df.groupby('user_id')
        
        for user_id, group in tqdm(user_groups):
            purchases = group[group['event_type'] == 'purchase']
            
            stats[user_id] = {
                'purchase_count': len(purchases),
                'total_spent': purchases['price'].sum() if len(purchases) > 0 else 0,
                'avg_price': purchases['price'].mean() if len(purchases) > 0 else self.train_df['price'].median(),
                'favorite_brands': group['brand'].value_counts().head(3).index.tolist(),
                'favorite_categories': group['category_code'].value_counts().head(3).index.tolist(),
            }
            
        return stats
    
    def _build_item_stats(self):
        """아이템별 통계 구축"""
        print("Building item statistics...")
        stats = {}
        
        item_groups = self.train_df.groupby('item_id')
        
        for item_id, group in tqdm(item_groups):
            category = group['category_code'].iloc[0] if not group['category_code'].isna().all() else 'unknown'
            brand = group['brand'].iloc[0] if not group['brand'].isna().all() else 'unknown'
            price = group['price'].median()
            
            # 카테고리에서 상위 레벨 추출
            main_category = category.split('.')[0] if pd.notna(category) and category != 'unknown' else 'default'
            
            stats[item_id] = {
                'price': price,
                'category': category,
                'main_category': main_category,
                'brand': brand,
                'popularity': len(group),  # 조회수
                'estimated_margin': CATEGORY_MARGIN.get(main_category, CATEGORY_MARGIN['default']),
                'is_premium_brand': brand.lower() in PREMIUM_BRANDS if pd.notna(brand) else False
            }
            
        return stats
    
    def rerank(self, user_id, candidates, weights=None):
        """
        마케팅 목표를 반영한 Re-ranking
        
        candidates: [(item_id, base_score), ...]
        weights: {'new_user': 1.0, 'vip': 1.0, 'upsell': 1.0, 'margin': 1.0, 'brand': 1.0}
        """
        if weights is None:
            weights = {
                'new_user': 1.5,
                'vip': 1.3,
                'upsell': 1.4,
                'margin': 1.2,
                'brand': 1.2
            }
        
        user_stat = self.user_stats.get(user_id, {
            'purchase_count': 0,
            'avg_price': self.train_df['price'].median(),
            'favorite_brands': [],
            'favorite_categories': []
        })
        
        reranked = []
        
        for item_id, base_score in candidates:
            item_stat = self.item_stats.get(item_id)
            
            if item_stat is None:
                # Unknown item, use base score
                reranked.append((item_id, base_score))
                continue
            
            marketing_score = base_score
            
            # 1. 신규 고객 전환 전략
            if user_stat['purchase_count'] == 0:
                # 저가 상품 우선 (진입 장벽 낮춤)
                if item_stat['price'] < user_stat['avg_price'] * 0.7:
                    marketing_score *= weights['new_user']
                    
                # 인기 상품 우선 (신뢰도 확보)
                if item_stat['popularity'] > 100:
                    marketing_score *= 1.2
                    
            # 2. VIP 고객 유지 전략
            elif user_stat['purchase_count'] >= 10:
                # 프리미엄 브랜드 우선
                if item_stat['is_premium_brand']:
                    marketing_score *= weights['vip']
                    
                # 선호 카테고리 우선
                if item_stat['main_category'] in user_stat['favorite_categories']:
                    marketing_score *= 1.3
                    
            # 3. Up-Sell 전략 (객단가 증대)
            price_ratio = item_stat['price'] / user_stat['avg_price']
            if 1.2 <= price_ratio <= 1.6:
                # 평소보다 20~60% 비싼 상품 추천 (Up-Sell)
                marketing_score *= weights['upsell']
                
            # 4. 마진 최적화
            if item_stat['estimated_margin'] >= 0.40:
                # 고마진 상품 우선
                marketing_score *= weights['margin']
                
            # 5. 브랜드 전략
            if item_stat['brand'] in user_stat['favorite_brands']:
                # 선호 브랜드
                marketing_score *= weights['brand']
                
            reranked.append((item_id, marketing_score))
        
        # 정렬 및 Top 10 반환
        reranked.sort(key=lambda x: -x[1])
        return reranked[:10]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--input', type=str, default='../out/output_ensemble_6_4.csv', 
                        help='Best baseline submission (0.1374)')
    parser.add_argument('--output', type=str, default='../out/output_marketing_rerank.csv')
    args = parser.parse_args()
    
    print("="*60)
    print("Marketing-Aware Reranker")
    print("="*60)
    
    # Load training data for user/item stats
    print("Loading training data...")
    train_df = pd.read_parquet(os.path.join(args.data_dir, 'train.parquet'))
    
    # Initialize reranker
    reranker = MarketingReranker(train_df)
    
    # Load baseline submission
    print(f"Loading baseline submission: {args.input}")
    baseline_df = pd.read_csv(args.input)
    
    # Group by user
    user_candidates = defaultdict(list)
    
    for _, row in baseline_df.iterrows():
        user_id = row['user_id']
        item_id = row['item_id']
        # Assume rank-based score (1st = 1.0, 2nd = 0.9, ...)
        # We don't have actual scores, so use inverse rank as proxy
        user_candidates[user_id].append(item_id)
    
    # Rerank
    print("Reranking with marketing strategies...")
    final_output = []
    
    for user_id in tqdm(user_candidates.keys()):
        items = user_candidates[user_id]
        
        # Create score (inverse rank)
        candidates = [(item, 1.0 / (i+1)) for i, item in enumerate(items)]
        
        # Rerank
        reranked = reranker.rerank(user_id, candidates)
        
        # Output
        for item_id, score in reranked:
            final_output.append({'user_id': user_id, 'item_id': item_id})
    
    # Save
    out_df = pd.DataFrame(final_output)
    out_df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")

if __name__ == '__main__':
    main()

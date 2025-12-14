"""
ALS Time Decay Recommender
- User Segmentation (Heavy/Medium/Light/Cold)
- Popularity Boost
- Time Decay Weighting (New!)
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from tqdm import tqdm
import argparse
import os
from datetime import datetime


class ALSTimeDecayRecommender:
    """ALS + User Segmentation + Popularity Boost + Time Decay"""

    def __init__(self, als_params: dict, popularity_boost: float = 0.1, decay_rate: float = 0.01):
        self.als_params = als_params
        self.popularity_boost = popularity_boost
        self.decay_rate = decay_rate

        self.als_model = None
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}

        self.user_interaction_count = {}  # user_id -> 상호작용 수
        self.item_popularity = {}  # item_id -> 정규화된 인기도
        self.global_popular_items = []

    def fit(self, train_df: pd.DataFrame):
        """모델 학습"""
        print("\n" + "="*60)
        print("ALS Time Decay Recommender - Training")
        print("="*60)

        # 1. 매핑 생성
        print("\n[1/4] Building mappings...")
        users = train_df['user_id'].unique()
        items = train_df['item_id'].unique()

        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.idx_to_user = {i: u for u, i in self.user_to_idx.items()}
        self.item_to_idx = {item: i for i, item in enumerate(items)}
        self.idx_to_item = {i: item for item, i in self.item_to_idx.items()}

        print(f"  Users: {len(users):,}, Items: {len(items):,}")

        # 2. 사용자별 상호작용 수 계산
        print("\n[2/4] Computing user activity levels...")
        user_counts = train_df.groupby('user_id').size()
        self.user_interaction_count = user_counts.to_dict()

        # 사용자 세그먼트 통계
        heavy = sum(1 for c in self.user_interaction_count.values() if c >= 50)
        medium = sum(1 for c in self.user_interaction_count.values() if 10 <= c < 50)
        light = sum(1 for c in self.user_interaction_count.values() if 0 < c < 10)
        print(f"  Heavy users (50+): {heavy:,}")
        print(f"  Medium users (10-50): {medium:,}")
        print(f"  Light users (<10): {light:,}")

        # 3. 아이템 인기도 계산
        print("\n[3/4] Computing item popularity...")

        # 구매 기반 인기도
        purchase_counts = train_df[train_df['event_type'] == 'purchase']['item_id'].value_counts()
        self.global_popular_items = purchase_counts.head(100).index.tolist()

        # 전체 인기도 (정규화)
        item_counts = train_df['item_id'].value_counts()
        max_count = item_counts.max()
        self.item_popularity = (item_counts / max_count).to_dict()

        print(f"  Popular items computed: {len(self.item_popularity):,}")

        # 4. ALS 학습 (Time Decay 적용)
        print("\n[4/4] Training ALS model with Time Decay...")
        print(f"  Decay Rate: {self.decay_rate}")

        # 데이터 복사 및 전처리
        train_df = train_df.copy()
        
        # 날짜 처리
        if train_df['event_time'].dtype == 'object':
             train_df['event_time'] = pd.to_datetime(train_df['event_time'])
        elif pd.api.types.is_numeric_dtype(train_df['event_time']):
             # timestamp인 경우 처리 (혹시 모를 대비)
             train_df['event_time'] = pd.to_datetime(train_df['event_time'], unit='s')

        max_date = train_df['event_time'].max()
        print(f"  Data range: {train_df['event_time'].min()} ~ {max_date}")

        # 경과 일수 계산
        # Timedelta를 일(float) 단위로 변환
        train_df['days_elapsed'] = (max_date - train_df['event_time']).dt.total_seconds() / (24 * 3600)
        
        # Time Decay Weight 계산: exp(-lambda * days)
        # 예: decay_rate=0.01, 30일 경과 -> exp(-0.3) ≈ 0.74
        train_df['time_weight'] = np.exp(-self.decay_rate * train_df['days_elapsed'])

        # Event Type 가중치
        weight_map = {'view': 1, 'cart': 10, 'purchase': 20}
        train_df['type_weight'] = train_df['event_type'].map(weight_map).fillna(1)

        # 최종 가중치 결합
        train_df['weight'] = train_df['type_weight'] * train_df['time_weight']

        # Index 매핑
        train_df['user_idx'] = train_df['user_id'].map(self.user_to_idx)
        train_df['item_idx'] = train_df['item_id'].map(self.item_to_idx)

        # Matrix 생성
        user_item_counts = train_df.groupby(['user_idx', 'item_idx'])['weight'].sum().reset_index(name='count')

        self.interaction_matrix = csr_matrix(
            (user_item_counts['count'].values,
             (user_item_counts['user_idx'].values, user_item_counts['item_idx'].values)),
            shape=(len(self.user_to_idx), len(self.item_to_idx))
        )

        print(f"  Interaction matrix: {self.interaction_matrix.nnz:,} unique pairs")
        print(f"  ALS params: {self.als_params}")

        self.als_model = AlternatingLeastSquares(**self.als_params)
        self.als_model.fit(self.interaction_matrix)

        print("\nTraining complete!")

    def recommend(self, user_id: str, top_k: int = 10) -> list:
        """추천 생성 (User Segmentation + Popularity Boost)"""

        # Cold-start
        if user_id not in self.user_to_idx:
            return self.global_popular_items[:top_k]

        user_idx = self.user_to_idx[user_id]
        user_activity = self.user_interaction_count.get(user_id, 0)

        # ALS 추천
        try:
            item_ids, als_scores = self.als_model.recommend(
                user_idx,
                self.interaction_matrix[user_idx],
                N=100,
                filter_already_liked_items=False
            )
        except:
            return self.global_popular_items[:top_k]

        # ALS 점수 정규화
        if als_scores.max() > als_scores.min():
            als_scores_norm = (als_scores - als_scores.min()) / (als_scores.max() - als_scores.min())
        else:
            als_scores_norm = np.ones_like(als_scores)

        # User Segmentation에 따른 인기도 가중치
        if user_activity >= 50:  # Heavy user
            pop_weight = 0.0  # 순수 개인화
        elif user_activity >= 10:  # Medium user
            pop_weight = self.popularity_boost  # 약간의 인기도
        else:  # Light user
            pop_weight = self.popularity_boost * 2  # 더 높은 인기도 가중치

        # 최종 점수 계산
        final_scores = []
        for i, (item_idx, als_score) in enumerate(zip(item_ids, als_scores_norm)):
            item_id = self.idx_to_item[item_idx]
            pop_score = self.item_popularity.get(item_id, 0)

            # 가중 결합
            final = (1 - pop_weight) * als_score + pop_weight * pop_score
            final_scores.append((item_id, final))

        # 정렬 및 Top-K
        final_scores.sort(key=lambda x: -x[1])
        return [item_id for item_id, _ in final_scores[:top_k]]

    def generate_submission(self, test_df: pd.DataFrame, output_path: str, top_k: int = 10):
        """제출 파일 생성"""
        print("\nGenerating recommendations...")

        results = []
        test_users = test_df['user_id'].unique()

        for user_id in tqdm(test_users, desc="Recommending"):
            recs = self.recommend(user_id, top_k=top_k)
            for item_id in recs:
                results.append({
                    'user_id': user_id,
                    'item_id': item_id
                })

        submission = pd.DataFrame(results)
        submission.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")
        print(f"  Total rows: {len(submission):,}")

        return submission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--output_dir', type=str, default='../out/')
    parser.add_argument('--output', type=str, default='output_als_time.csv')
    parser.add_argument('--factors', type=int, default=64)
    parser.add_argument('--regularization', type=float, default=0.01)
    parser.add_argument('--alpha', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--popularity_boost', type=float, default=0.1)
    parser.add_argument('--decay_rate', type=float, default=0.01, help='Time decay rate (lambda)')
    args = parser.parse_args()

    als_params = {
        'factors': args.factors,
        'regularization': args.regularization,
        'alpha': args.alpha,
        'iterations': args.iterations,
        'random_state': 42
    }

    print("="*60)
    print("ALS Time Decay Recommender")
    print("="*60)
    print(f"\nALS Params: factors={args.factors}, reg={args.regularization}, "
          f"alpha={args.alpha}, iter={args.iterations}")
    print(f"Popularity Boost: {args.popularity_boost}")
    print(f"Time Decay Rate: {args.decay_rate}")

    # 데이터 로드
    print("\n[1/3] Loading data...")
    train_df = pd.read_parquet(os.path.join(args.data_dir, 'train.parquet'))

    sample_sub = pd.read_csv(os.path.join(args.data_dir, 'sample_submission.csv'))
    test_users = sample_sub['user_id'].unique()
    test_df = pd.DataFrame({'user_id': test_users})

    print(f"  Train: {len(train_df):,} interactions")
    print(f"  Test: {len(test_users):,} users")

    # 모델 학습
    print("\n[2/3] Training model...")
    model = ALSTimeDecayRecommender(
        als_params=als_params, 
        popularity_boost=args.popularity_boost,
        decay_rate=args.decay_rate
    )
    model.fit(train_df)

    # 제출 파일 생성
    print("\n[3/3] Generating submission...")
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output)
    model.generate_submission(test_df, output_path)

    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == '__main__':
    main()

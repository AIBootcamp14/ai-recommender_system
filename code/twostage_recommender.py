"""
Two-Stage Recommender System
Stage 1: ALS로 후보 100개 생성
Stage 2: LightGBM으로 재순위 (구매 확률 예측)
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from implicit.als import AlternatingLeastSquares
from collections import defaultdict
from itertools import combinations
from tqdm import tqdm
import argparse
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split


class TwoStageRecommender:
    """Two-Stage Recommender: ALS + LightGBM Reranking"""

    def __init__(self, als_params: dict):
        self.als_params = als_params
        self.als_model = None
        self.lgb_model = None

        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}

        # 아이템 속성
        self.item_category = {}
        self.item_brand = {}
        self.item_price = {}
        self.item_popularity = {}

        # 사용자 통계
        self.user_activity = {}
        self.user_category_pref = {}
        self.user_brand_pref = {}
        self.user_avg_price = {}

        # 전역 통계
        self.category_purchase_rate = {}
        self.brand_purchase_rate = {}
        self.global_popular_items = []

    def fit(self, train_df: pd.DataFrame):
        """모델 학습"""
        print("\n" + "="*60)
        print("Two-Stage Recommender - Training")
        print("="*60)

        # 1. 매핑 생성
        print("\n[1/6] Building mappings...")
        users = train_df['user_id'].unique()
        items = train_df['item_id'].unique()

        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.idx_to_user = {i: u for u, i in self.user_to_idx.items()}
        self.item_to_idx = {item: i for i, item in enumerate(items)}
        self.idx_to_item = {i: item for item, i in self.item_to_idx.items()}

        print(f"  Users: {len(users):,}, Items: {len(items):,}")

        # 2. 아이템 속성 추출
        print("\n[2/6] Extracting item features...")
        self._build_item_features(train_df)

        # 3. 사용자 통계 추출
        print("\n[3/6] Building user statistics...")
        self._build_user_features(train_df)

        # 4. 전역 통계 추출
        print("\n[4/6] Building global statistics...")
        self._build_global_stats(train_df)

        # 5. ALS 학습
        print("\n[5/6] Training ALS model...")
        self._train_als(train_df)

        # 6. LightGBM 학습
        print("\n[6/6] Training LightGBM reranker...")
        self._train_lgb(train_df)

        print("\nTraining complete!")

    def _build_item_features(self, train_df: pd.DataFrame):
        """아이템 특성 추출"""
        item_attrs = train_df.groupby('item_id').agg({
            'category_code': 'first',
            'brand': 'first',
            'price': 'median',
            'user_id': 'nunique'  # 인기도
        }).reset_index()

        for _, row in item_attrs.iterrows():
            item_id = row['item_id']
            self.item_category[item_id] = row['category_code']
            self.item_brand[item_id] = row['brand']
            self.item_price[item_id] = row['price']
            self.item_popularity[item_id] = row['user_id']

        # 인기도 정규화
        max_pop = max(self.item_popularity.values())
        for item_id in self.item_popularity:
            self.item_popularity[item_id] /= max_pop

        print(f"  Items with features: {len(self.item_category):,}")

    def _build_user_features(self, train_df: pd.DataFrame):
        """사용자 특성 추출"""
        user_groups = train_df.groupby('user_id')

        for user_id, group in tqdm(user_groups, desc="  Building user profiles"):
            # 활동량
            self.user_activity[user_id] = len(group)

            # 카테고리 선호도
            cat_counts = group['category_code'].value_counts(normalize=True)
            self.user_category_pref[user_id] = cat_counts.to_dict()

            # 브랜드 선호도
            brand_counts = group['brand'].value_counts(normalize=True)
            self.user_brand_pref[user_id] = brand_counts.to_dict()

            # 평균 가격
            self.user_avg_price[user_id] = group['price'].mean()

        print(f"  User profiles: {len(self.user_activity):,}")

    def _build_global_stats(self, train_df: pd.DataFrame):
        """전역 통계 추출"""
        # 카테고리별 구매율
        cat_stats = train_df.groupby('category_code').apply(
            lambda x: (x['event_type'] == 'purchase').mean()
        )
        self.category_purchase_rate = cat_stats.to_dict()

        # 브랜드별 구매율
        brand_stats = train_df.groupby('brand').apply(
            lambda x: (x['event_type'] == 'purchase').mean()
        )
        self.brand_purchase_rate = brand_stats.to_dict()

        # 인기 아이템 (구매 기준)
        purchase_counts = train_df[train_df['event_type'] == 'purchase']['item_id'].value_counts()
        self.global_popular_items = purchase_counts.head(100).index.tolist()

        print(f"  Category purchase rates: {len(self.category_purchase_rate)}")
        print(f"  Brand purchase rates: {len(self.brand_purchase_rate)}")

    def _train_als(self, train_df: pd.DataFrame):
        """ALS 모델 학습"""
        # event_type 가중치 적용
        weight_map = {'view': 1, 'cart': 10, 'purchase': 20}
        train_df = train_df.copy()
        train_df['user_idx'] = train_df['user_id'].map(self.user_to_idx)
        train_df['item_idx'] = train_df['item_id'].map(self.item_to_idx)
        train_df['weight'] = train_df['event_type'].map(weight_map).fillna(1)

        user_item_counts = train_df.groupby(['user_idx', 'item_idx'])['weight'].sum().reset_index(name='count')

        self.interaction_matrix = csr_matrix(
            (user_item_counts['count'].values,
             (user_item_counts['user_idx'].values, user_item_counts['item_idx'].values)),
            shape=(len(self.user_to_idx), len(self.item_to_idx))
        )

        print(f"  Interaction matrix: {self.interaction_matrix.nnz:,} unique pairs")

        self.als_model = AlternatingLeastSquares(**self.als_params)
        self.als_model.fit(self.interaction_matrix)

    def _create_features(self, user_id: str, item_id: str, als_score: float, als_rank: int) -> dict:
        """LightGBM용 피처 생성"""
        features = {}

        # ALS 관련
        features['als_score'] = als_score
        features['als_rank'] = als_rank

        # 아이템 특성
        features['item_popularity'] = self.item_popularity.get(item_id, 0)
        features['item_price'] = self.item_price.get(item_id, 0)

        # 사용자 특성
        features['user_activity'] = self.user_activity.get(user_id, 0)
        features['user_avg_price'] = self.user_avg_price.get(user_id, 0)

        # 가격 차이
        user_price = self.user_avg_price.get(user_id, 0)
        item_price = self.item_price.get(item_id, 0)
        features['price_diff'] = abs(user_price - item_price) if user_price > 0 else 0
        features['price_ratio'] = item_price / user_price if user_price > 0 else 1

        # 카테고리 매칭
        item_cat = self.item_category.get(item_id)
        user_cat_pref = self.user_category_pref.get(user_id, {})
        features['category_match'] = user_cat_pref.get(item_cat, 0) if item_cat else 0

        # 브랜드 매칭
        item_brand = self.item_brand.get(item_id)
        user_brand_pref = self.user_brand_pref.get(user_id, {})
        features['brand_match'] = user_brand_pref.get(item_brand, 0) if item_brand else 0

        # 카테고리/브랜드 구매율
        features['category_purchase_rate'] = self.category_purchase_rate.get(item_cat, 0) if item_cat else 0
        features['brand_purchase_rate'] = self.brand_purchase_rate.get(item_brand, 0) if item_brand else 0

        # 활동량 구간 (Light/Medium/Heavy)
        activity = self.user_activity.get(user_id, 0)
        features['is_light_user'] = 1 if activity < 10 else 0
        features['is_heavy_user'] = 1 if activity >= 50 else 0

        return features

    def _train_lgb(self, train_df: pd.DataFrame):
        """LightGBM 모델 학습"""
        print("  Creating training data for LightGBM...")

        # 구매 이벤트 추출 (positive samples)
        purchase_df = train_df[train_df['event_type'] == 'purchase'][['user_id', 'item_id']].drop_duplicates()
        purchase_set = set(zip(purchase_df['user_id'], purchase_df['item_id']))

        print(f"  Positive samples (purchases): {len(purchase_set):,}")

        # 학습 데이터 생성 (샘플링)
        training_data = []
        sample_users = np.random.choice(
            list(self.user_to_idx.keys()),
            size=min(50000, len(self.user_to_idx)),
            replace=False
        )

        for user_id in tqdm(sample_users, desc="  Generating training samples"):
            if user_id not in self.user_to_idx:
                continue

            user_idx = self.user_to_idx[user_id]

            # ALS 후보 생성
            try:
                item_ids, als_scores = self.als_model.recommend(
                    user_idx,
                    self.interaction_matrix[user_idx],
                    N=50,
                    filter_already_liked_items=False
                )
            except:
                continue

            for rank, (item_idx, als_score) in enumerate(zip(item_ids, als_scores)):
                item_id = self.idx_to_item[item_idx]

                # 레이블: 구매 여부
                label = 1 if (user_id, item_id) in purchase_set else 0

                # 피처 생성
                features = self._create_features(user_id, item_id, als_score, rank)
                features['label'] = label

                training_data.append(features)

        # DataFrame 변환
        train_lgb_df = pd.DataFrame(training_data)

        if len(train_lgb_df) == 0:
            print("  Warning: No training data generated, skipping LightGBM")
            return

        print(f"  LightGBM training samples: {len(train_lgb_df):,}")
        print(f"  Positive ratio: {train_lgb_df['label'].mean():.4f}")

        # Train/Val split
        feature_cols = [c for c in train_lgb_df.columns if c != 'label']
        X = train_lgb_df[feature_cols]
        y = train_lgb_df['label']

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # LightGBM 학습
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }

        self.lgb_model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(50)]
        )

        self.feature_cols = feature_cols

        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.lgb_model.feature_importance()
        }).sort_values('importance', ascending=False)

        print("\n  Top features:")
        for _, row in importance.head(5).iterrows():
            print(f"    {row['feature']}: {row['importance']}")

    def recommend(self, user_id: str, top_k: int = 10) -> list:
        """추천 생성"""
        if user_id not in self.user_to_idx:
            # Cold-start: 인기 아이템 반환
            return self.global_popular_items[:top_k]

        user_idx = self.user_to_idx[user_id]

        # Stage 1: ALS 후보 생성
        n_candidates = 100
        try:
            item_ids, als_scores = self.als_model.recommend(
                user_idx,
                self.interaction_matrix[user_idx],
                N=n_candidates,
                filter_already_liked_items=False
            )
        except:
            return self.global_popular_items[:top_k]

        # Stage 2: LightGBM Reranking
        if self.lgb_model is None:
            # LightGBM 없으면 ALS 결과 그대로 반환
            return [self.idx_to_item[idx] for idx in item_ids[:top_k]]

        candidates = []
        for rank, (item_idx, als_score) in enumerate(zip(item_ids, als_scores)):
            item_id = self.idx_to_item[item_idx]
            features = self._create_features(user_id, item_id, als_score, rank)
            candidates.append((item_id, features))

        # LightGBM 예측
        X_pred = pd.DataFrame([c[1] for c in candidates])[self.feature_cols]
        lgb_scores = self.lgb_model.predict(X_pred)

        # 재정렬
        scored_items = [(candidates[i][0], lgb_scores[i]) for i in range(len(candidates))]
        scored_items.sort(key=lambda x: -x[1])

        return [item_id for item_id, _ in scored_items[:top_k]]

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
    parser.add_argument('--output', type=str, default='output_twostage.csv')
    args = parser.parse_args()

    # ALS 최적 파라미터
    als_params = {
        'factors': 32,
        'regularization': 0.0215,
        'alpha': 7,
        'iterations': 15,
        'random_state': 42
    }

    print("="*60)
    print("Two-Stage Recommender (ALS + LightGBM)")
    print("="*60)

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
    model = TwoStageRecommender(als_params=als_params)
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

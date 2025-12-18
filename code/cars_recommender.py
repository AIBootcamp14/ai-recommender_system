"""
CARS (Context-Aware Recommender System) - Vectorized Version
- Temporal Context: 시간대, 요일, 계절
- User Context: 선호 시간대, 가격대, 카테고리
- Model: LightGBM for CTR prediction (Vectorized)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm
import os

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Error: lightgbm not installed")


class CARSFeatureEngineering:
    """Context-Aware Feature Engineering (Vectorized)"""

    def __init__(self):
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.category_encoder = LabelEncoder()
        self.brand_encoder = LabelEncoder()

        self.user_context = None
        self.item_context = None
        self.popular_items = []

    def fit(self, df: pd.DataFrame):
        """학습 데이터로 인코더 및 Context 학습"""
        print("=" * 60)
        print("CARS Feature Engineering - Fit (Vectorized)")
        print("=" * 60)

        # 결측치 처리
        df = df.copy()
        df['category_code'] = df['category_code'].fillna('unknown')
        df['brand'] = df['brand'].fillna('unknown')

        # 1. 기본 인코딩
        print("\n[1/4] 기본 인코딩...")
        self.user_encoder.fit(df['user_id'].astype(str))
        self.item_encoder.fit(df['item_id'].astype(str))
        self.category_encoder.fit(df['category_code'].astype(str))
        self.brand_encoder.fit(df['brand'].astype(str))

        # 2. Temporal Features 추가
        print("[2/4] Temporal Features 추출...")
        df['event_time'] = pd.to_datetime(df['event_time'])
        df['hour'] = df['event_time'].dt.hour
        df['day_of_week'] = df['event_time'].dt.dayofweek

        # 3. User Context 집계
        print("[3/4] User Context 집계...")
        self._build_user_context(df)

        # 4. Item Context 집계
        print("[4/4] Item Context 집계...")
        self._build_item_context(df)

        # Popular Items
        item_counts = df.groupby('item_id').size().sort_values(ascending=False)
        self.popular_items = item_counts.head(500).index.tolist()

        print(f"\n완료: {len(self.user_context)} users, {len(self.item_context)} items")

    def _build_user_context(self, df: pd.DataFrame):
        """사용자별 Context 집계 (Vectorized)"""
        # 기본 통계
        user_stats = df.groupby('user_id').agg(
            avg_hour=('hour', 'mean'),
            avg_price=('price', 'mean'),
            std_price=('price', 'std'),
            item_diversity=('item_id', 'nunique'),
            interaction_count=('item_id', 'count')
        ).reset_index()

        user_stats['std_price'] = user_stats['std_price'].fillna(0)

        # 주 활동 요일
        mode_dow = df.groupby('user_id')['day_of_week'].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 3)
        user_stats = user_stats.merge(mode_dow.rename('mode_dow').reset_index(), on='user_id', how='left')

        # event_type 비율
        event_counts = df.groupby(['user_id', 'event_type']).size().unstack(fill_value=0)
        total = event_counts.sum(axis=1)
        if 'cart' in event_counts.columns:
            event_counts['cart_rate'] = event_counts['cart'] / total
        else:
            event_counts['cart_rate'] = 0
        if 'purchase' in event_counts.columns:
            event_counts['purchase_rate'] = event_counts['purchase'] / total
        else:
            event_counts['purchase_rate'] = 0

        user_stats = user_stats.merge(
            event_counts[['cart_rate', 'purchase_rate']].reset_index(),
            on='user_id', how='left'
        )
        user_stats['cart_rate'] = user_stats['cart_rate'].fillna(0)
        user_stats['purchase_rate'] = user_stats['purchase_rate'].fillna(0)

        self.user_context = user_stats.set_index('user_id')

    def _build_item_context(self, df: pd.DataFrame):
        """아이템별 Context 집계 (Vectorized)"""
        item_stats = df.groupby('item_id').agg(
            price=('price', 'mean'),
            popularity=('user_id', 'nunique')
        ).reset_index()

        # 인기도 정규화
        max_pop = item_stats['popularity'].max()
        item_stats['popularity_norm'] = item_stats['popularity'] / max_pop

        self.item_context = item_stats.set_index('item_id')

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터프레임 전체에 Context Feature 추가 (Vectorized)"""
        df = df.copy()

        # 결측치 처리
        df['category_code'] = df['category_code'].fillna('unknown')
        df['brand'] = df['brand'].fillna('unknown')

        # 1. Temporal Context
        df['event_time'] = pd.to_datetime(df['event_time'])
        df['hour'] = df['event_time'].dt.hour
        df['day_of_week'] = df['event_time'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['time_slot'] = pd.cut(df['hour'], bins=[0, 6, 12, 18, 24],
                                  labels=[0, 1, 2, 3], include_lowest=True).astype(int)
        df['is_peak_hour'] = ((df['hour'] >= 14) & (df['hour'] <= 18)).astype(int)
        df['month'] = df['event_time'].dt.month

        # 2. User Context (Join)
        df = df.merge(
            self.user_context.add_prefix('user_'),
            left_on='user_id', right_index=True, how='left'
        )

        # Cold-start user 기본값
        df['user_avg_hour'] = df['user_avg_hour'].fillna(12)
        df['user_mode_dow'] = df['user_mode_dow'].fillna(3)
        df['user_avg_price'] = df['user_avg_price'].fillna(79)
        df['user_std_price'] = df['user_std_price'].fillna(50)
        df['user_item_diversity'] = df['user_item_diversity'].fillna(1)
        df['user_interaction_count'] = df['user_interaction_count'].fillna(0)
        df['user_cart_rate'] = df['user_cart_rate'].fillna(0)
        df['user_purchase_rate'] = df['user_purchase_rate'].fillna(0)

        # 3. Item Context (Join)
        df = df.merge(
            self.item_context.add_prefix('item_'),
            left_on='item_id', right_index=True, how='left'
        )

        # Cold-start item 기본값
        df['item_price'] = df['item_price'].fillna(79)
        df['item_popularity'] = df['item_popularity'].fillna(1)
        df['item_popularity_norm'] = df['item_popularity_norm'].fillna(0)

        # 4. Encoding
        df['user_idx'] = self.user_encoder.transform(df['user_id'].astype(str).values)

        # Item encoding with unknown handling
        item_mapping = {item: idx for idx, item in enumerate(self.item_encoder.classes_)}
        df['item_idx'] = df['item_id'].map(item_mapping).fillna(0).astype(int)

        # Category/Brand encoding
        cat_mapping = {cat: idx for idx, cat in enumerate(self.category_encoder.classes_)}
        df['category_idx'] = df['category_code'].astype(str).map(cat_mapping).fillna(0).astype(int)

        brand_mapping = {brand: idx for idx, brand in enumerate(self.brand_encoder.classes_)}
        df['brand_idx'] = df['brand'].astype(str).map(brand_mapping).fillna(0).astype(int)

        return df


class CARSRecommender:
    """Context-Aware Recommender using LightGBM (Vectorized)"""

    def __init__(self, feature_eng: CARSFeatureEngineering):
        self.feature_eng = feature_eng
        self.model = None
        self.feature_names = [
            'hour', 'day_of_week', 'is_weekend', 'time_slot', 'is_peak_hour', 'month',
            'user_avg_hour', 'user_mode_dow', 'user_avg_price', 'user_std_price',
            'user_item_diversity', 'user_interaction_count', 'user_cart_rate', 'user_purchase_rate',
            'item_price', 'item_popularity_norm',
            'user_idx', 'item_idx', 'category_idx', 'brand_idx'
        ]

    def prepare_training_data(self, df: pd.DataFrame, sample_size: int = 500000,
                               negative_ratio: int = 2) -> pd.DataFrame:
        """학습 데이터 준비 (Vectorized Negative Sampling)"""
        print("\n" + "=" * 60)
        print("CARS Training Data Preparation (Vectorized)")
        print("=" * 60)

        # Transform features
        print("\n[1/4] Feature Transform...")
        df_transformed = self.feature_eng.transform(df)

        # Sample positive data
        print(f"[2/4] Sampling {sample_size:,} positive samples...")
        if len(df_transformed) > sample_size:
            pos_df = df_transformed.sample(sample_size, random_state=42)
        else:
            pos_df = df_transformed
        pos_df['label'] = 1

        # Negative Sampling (Vectorized)
        print(f"[3/4] Generating negative samples (ratio={negative_ratio})...")
        n_neg = len(pos_df) * negative_ratio
        all_items = np.array(self.feature_eng.popular_items[:1000])  # 인기 아이템만

        # 랜덤 user-item 조합 생성
        neg_users = pos_df['user_id'].sample(n_neg, replace=True).values
        neg_items = np.random.choice(all_items, size=n_neg)

        neg_df = pd.DataFrame({
            'user_id': neg_users,
            'item_id': neg_items,
            'event_time': pos_df['event_time'].sample(n_neg, replace=True).values,
            'category_code': 'unknown',
            'brand': 'unknown',
            'price': 79
        })

        neg_df = self.feature_eng.transform(neg_df)
        neg_df['label'] = 0

        # 합치기
        print("[4/4] Merging data...")
        feature_cols = self.feature_names + ['label']
        train_df = pd.concat([
            pos_df[feature_cols],
            neg_df[feature_cols]
        ], ignore_index=True)

        # Shuffle
        train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"\n완료: {len(train_df):,} samples (pos={len(pos_df):,}, neg={len(neg_df):,})")

        return train_df

    def train(self, train_df: pd.DataFrame):
        """LightGBM 모델 학습"""
        print("\n" + "=" * 60)
        print("CARS Model Training (LightGBM)")
        print("=" * 60)

        X = train_df[self.feature_names]
        y = train_df['label']

        print(f"\nFeatures ({len(self.feature_names)}): {self.feature_names}")
        print(f"Train size: {len(X):,}")

        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 64,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_jobs': -1,
        }

        train_data = lgb.Dataset(X, label=y)

        print("\n학습 중...")
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            callbacks=[lgb.log_evaluation(50)]
        )

        # Feature Importance
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importance()
        }).sort_values('importance', ascending=False)

        print("\n[Feature Importance]")
        print(importance.head(10).to_string(index=False))

    def generate_recommendations(self, users: np.ndarray, event_time,
                                  top_k: int = 10) -> pd.DataFrame:
        """모든 사용자에 대해 추천 생성 (Fully Vectorized)"""
        print("\n" + "=" * 60)
        print("CARS Recommendation Generation (Optimized)")
        print("=" * 60)

        # 인기 아이템 Top 100만 사용 (속도 향상)
        candidate_items = np.array(self.feature_eng.popular_items[:100])
        n_candidates = len(candidate_items)

        print(f"Users: {len(users):,}, Candidates: {n_candidates}")

        results = []

        # 배치 처리 (더 큰 배치)
        batch_size = 50000
        n_batches = (len(users) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(n_batches), desc="Generating"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(users))
            batch_users = users[start_idx:end_idx]
            n_batch = len(batch_users)

            # User x Item 조합 생성
            user_repeat = np.repeat(batch_users, n_candidates)
            item_tile = np.tile(candidate_items, n_batch)

            # 예측 데이터 생성 (최소한의 컬럼만)
            pred_df = pd.DataFrame({
                'user_id': user_repeat,
                'item_id': item_tile,
                'event_time': event_time,
                'category_code': 'unknown',
                'brand': 'unknown',
                'price': 79
            })

            # Transform
            pred_df = self.feature_eng.transform(pred_df)

            # 예측
            scores = self.model.predict(pred_df[self.feature_names])

            # Reshape으로 User별 점수 매트릭스 생성
            score_matrix = scores.reshape(n_batch, n_candidates)

            # Argsort로 Top-K 인덱스 추출 (Vectorized)
            top_k_indices = np.argsort(-score_matrix, axis=1)[:, :top_k]

            # 결과 저장 (Vectorized)
            for i, user_id in enumerate(batch_users):
                top_items = candidate_items[top_k_indices[i]]
                for item_id in top_items:
                    results.append({'user_id': user_id, 'item_id': item_id})

        return pd.DataFrame(results)


def generate_cars_recommendations(data_dir: str, output_path: str,
                                   sample_size: int = 300000):
    """CARS 추천 생성 파이프라인"""
    print("=" * 60)
    print("CARS (Context-Aware Recommender System) v2")
    print("=" * 60)

    # 1. 데이터 로드
    print("\n[1/6] 데이터 로드...")
    train_df = pd.read_parquet(os.path.join(data_dir, 'train.parquet'))
    sample_sub = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))

    print(f"Train: {len(train_df):,} rows")
    print(f"Users to predict: {len(sample_sub):,}")

    # 2. Feature Engineering
    print("\n[2/6] Feature Engineering...")
    feature_eng = CARSFeatureEngineering()
    feature_eng.fit(train_df)

    # 3. 학습 데이터 준비 (최근 30일 사용)
    print("\n[3/6] 학습 데이터 준비...")
    train_df['event_time'] = pd.to_datetime(train_df['event_time'])
    recent_df = train_df[train_df['event_time'] >= train_df['event_time'].max() - pd.Timedelta(days=30)]
    print(f"Recent 30 days: {len(recent_df):,} rows")

    recommender = CARSRecommender(feature_eng)
    cars_train_df = recommender.prepare_training_data(recent_df, sample_size=sample_size)

    # 4. 모델 학습
    print("\n[4/6] 모델 학습...")
    recommender.train(cars_train_df)

    # 5. 추천 생성
    print("\n[5/6] 추천 생성...")
    users = sample_sub['user_id'].unique()
    last_event_time = train_df['event_time'].max()

    result_df = recommender.generate_recommendations(
        users=users,
        event_time=last_event_time,
        top_k=10
    )

    # 6. 저장
    print("\n[6/6] 결과 저장...")
    result_df.to_csv(output_path, index=False)

    print(f"\n저장 완료: {output_path}")
    print(f"총 {len(result_df):,} rows")

    return result_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='CARS Recommender v2')
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--output', type=str, default='../out/output_cars_v1.csv')
    parser.add_argument('--sample_size', type=int, default=300000)

    args = parser.parse_args()

    generate_cars_recommendations(
        data_dir=args.data_dir,
        output_path=args.output,
        sample_size=args.sample_size
    )

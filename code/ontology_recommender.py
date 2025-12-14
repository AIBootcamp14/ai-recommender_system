"""
온톨로지 기반 추천 시스템
- LLM 없이 100% 커버리지 달성
- 카테고리/브랜드/가격 속성 전파
- Co-view 그래프 기반 유사도
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
import pickle


class OntologyRecommender:
    """온톨로지 기반 다중 신호 추천기"""

    def __init__(self, als_params: dict, content_weight: float = 0.1,
                 coview_weight: float = 0.1):
        self.als_params = als_params
        self.content_weight = content_weight
        self.coview_weight = coview_weight
        self.als_weight = 1.0 - content_weight - coview_weight

        self.als_model = None
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}

        # 온톨로지 속성
        self.item_category = {}      # item_id -> category
        self.item_brand = {}         # item_id -> brand
        self.item_price_tier = {}    # item_id -> price_tier (0-4)
        self.category_to_idx = {}    # category -> idx
        self.brand_to_idx = {}       # brand -> idx

        # Co-view 그래프
        self.coview_matrix = None    # item x item sparse matrix

        # 사용자 프로파일
        self.user_category_pref = {} # user_id -> category 분포
        self.user_brand_pref = {}    # user_id -> brand 분포
        self.user_price_pref = {}    # user_id -> 평균 price_tier

    def build_ontology(self, train_df: pd.DataFrame):
        """온톨로지 속성 추출"""
        print("\n[Ontology] Building item attributes...")

        # 아이템별 속성 추출
        item_attrs = train_df.groupby('item_id').agg({
            'category_code': 'first',
            'brand': 'first',
            'price': 'median'
        }).reset_index()

        # 카테고리 매핑
        categories = item_attrs['category_code'].unique()
        self.category_to_idx = {c: i for i, c in enumerate(categories)}

        # 브랜드 매핑
        brands = item_attrs['brand'].unique()
        self.brand_to_idx = {b: i for i, b in enumerate(brands)}

        # 가격 구간화 (5등급)
        price_quantiles = item_attrs['price'].quantile([0.2, 0.4, 0.6, 0.8]).values

        for _, row in item_attrs.iterrows():
            item_id = row['item_id']
            self.item_category[item_id] = row['category_code']
            self.item_brand[item_id] = row['brand']

            # 가격 등급
            price = row['price']
            tier = np.searchsorted(price_quantiles, price)
            self.item_price_tier[item_id] = tier

        print(f"  Categories: {len(self.category_to_idx)}")
        print(f"  Brands: {len(self.brand_to_idx)}")
        print(f"  Items with attributes: {len(self.item_category)}")

    def build_coview_graph(self, train_df: pd.DataFrame, min_coview: int = 2):
        """Co-view 그래프 구축"""
        print("\n[Ontology] Building co-view graph...")

        # 세션별 아이템
        sessions = train_df.groupby('user_session')['item_id'].apply(list)
        multi_sessions = sessions[sessions.apply(len) >= 2]

        print(f"  Multi-item sessions: {len(multi_sessions):,}")

        # Co-view 카운트
        coview_counts = defaultdict(int)

        for items in tqdm(multi_sessions, desc="  Counting co-views"):
            unique_items = list(set(items))[:20]  # 세션당 최대 20개
            for i1, i2 in combinations(unique_items, 2):
                if i1 in self.item_to_idx and i2 in self.item_to_idx:
                    idx1, idx2 = self.item_to_idx[i1], self.item_to_idx[i2]
                    coview_counts[(idx1, idx2)] += 1
                    coview_counts[(idx2, idx1)] += 1

        # Sparse matrix 생성
        n_items = len(self.item_to_idx)
        self.coview_matrix = lil_matrix((n_items, n_items), dtype=np.float32)

        filtered_count = 0
        for (i, j), count in coview_counts.items():
            if count >= min_coview:
                self.coview_matrix[i, j] = count
                filtered_count += 1

        self.coview_matrix = self.coview_matrix.tocsr()

        # 행별 정규화 (L1)
        row_sums = np.array(self.coview_matrix.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        self.coview_matrix = self.coview_matrix.multiply(1.0 / row_sums[:, np.newaxis]).tocsr()

        print(f"  Co-view pairs (min={min_coview}): {filtered_count:,}")

    def build_user_profiles(self, train_df: pd.DataFrame):
        """사용자 프로파일 구축"""
        print("\n[Ontology] Building user profiles...")

        user_groups = train_df.groupby('user_id')

        for user_id, group in tqdm(user_groups, desc="  Profiling users"):
            # 카테고리 선호도 (빈도 기반)
            cat_counts = group['category_code'].value_counts(normalize=True)
            self.user_category_pref[user_id] = cat_counts.to_dict()

            # 브랜드 선호도
            brand_counts = group['brand'].value_counts(normalize=True)
            self.user_brand_pref[user_id] = brand_counts.to_dict()

            # 평균 가격대
            prices = group['price'].values
            price_quantiles = np.array([20, 40, 60, 80])  # 대략적인 구간
            tiers = np.searchsorted(price_quantiles, prices)
            self.user_price_pref[user_id] = np.mean(tiers)

        print(f"  User profiles: {len(self.user_category_pref):,}")

    def fit(self, train_df: pd.DataFrame):
        """모델 학습"""
        print("\n" + "="*60)
        print("Ontology Recommender - Training")
        print("="*60)

        # 1. 매핑 생성
        print("\n[1/5] Building mappings...")
        users = train_df['user_id'].unique()
        items = train_df['item_id'].unique()

        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.idx_to_user = {i: u for u, i in self.user_to_idx.items()}
        self.item_to_idx = {item: i for i, item in enumerate(items)}
        self.idx_to_item = {i: item for item, i in self.item_to_idx.items()}

        print(f"  Users: {len(users):,}, Items: {len(items):,}")

        # 2. 온톨로지 구축
        print("\n[2/5] Building ontology...")
        self.build_ontology(train_df)

        # 3. 사용자 프로파일
        print("\n[3/5] Building user profiles...")
        self.build_user_profiles(train_df)

        # 4. Co-view 그래프
        print("\n[4/5] Building co-view graph...")
        self.build_coview_graph(train_df)

        # 5. ALS 학습
        print("\n[5/5] Training ALS model...")

        # event_type 가중치 적용 (최적값: view=1, cart=10, purchase=20)
        weight_map = {'view': 1, 'cart': 10, 'purchase': 20}
        train_df['user_idx'] = train_df['user_id'].map(self.user_to_idx)
        train_df['item_idx'] = train_df['item_id'].map(self.item_to_idx)
        train_df['weight'] = train_df['event_type'].map(weight_map).fillna(1)
        user_item_counts = train_df.groupby(['user_idx', 'item_idx'])['weight'].sum().reset_index(name='count')

        interaction_matrix = csr_matrix(
            (user_item_counts['count'].values,
             (user_item_counts['user_idx'].values, user_item_counts['item_idx'].values)),
            shape=(len(users), len(items))
        )
        print(f"  Interaction matrix: {interaction_matrix.nnz:,} unique pairs")

        self.als_model = AlternatingLeastSquares(**self.als_params)
        self.als_model.fit(interaction_matrix)

        self.interaction_matrix = interaction_matrix
        print("  Training complete!")

    def get_content_score(self, user_id: str, item_id: str) -> float:
        """콘텐츠 기반 점수 (온톨로지 속성 매칭)"""
        score = 0.0

        # 카테고리 매칭
        item_cat = self.item_category.get(item_id)
        user_cat_pref = self.user_category_pref.get(user_id, {})
        if item_cat and item_cat in user_cat_pref:
            score += user_cat_pref[item_cat] * 0.5

        # 브랜드 매칭
        item_brand = self.item_brand.get(item_id)
        user_brand_pref = self.user_brand_pref.get(user_id, {})
        if item_brand and item_brand in user_brand_pref:
            score += user_brand_pref[item_brand] * 0.3

        # 가격대 매칭 (차이가 작을수록 높은 점수)
        item_tier = self.item_price_tier.get(item_id, 2)
        user_tier = self.user_price_pref.get(user_id, 2)
        tier_diff = abs(item_tier - user_tier)
        score += (1 - tier_diff / 4) * 0.2

        return score

    def get_coview_score(self, user_id: str, candidate_idx: int) -> float:
        """Co-view 기반 점수"""
        if user_id not in self.user_to_idx:
            return 0.0

        user_idx = self.user_to_idx[user_id]

        # 사용자가 본 아이템들
        user_items = self.interaction_matrix[user_idx].indices

        if len(user_items) == 0:
            return 0.0

        # 후보 아이템과 사용자 히스토리 간의 co-view 점수
        coview_scores = self.coview_matrix[user_items, candidate_idx].toarray().flatten()

        return np.mean(coview_scores) if len(coview_scores) > 0 else 0.0

    def recommend(self, user_id: str, top_k: int = 10) -> list:
        """추천 생성"""
        if user_id not in self.user_to_idx:
            # Cold-start: 인기 아이템 반환
            return self._get_popular_items(top_k)

        user_idx = self.user_to_idx[user_id]

        # ALS 후보 생성 (Top 100)
        n_candidates = 100
        item_ids, als_scores = self.als_model.recommend(
            user_idx,
            self.interaction_matrix[user_idx],
            N=n_candidates,
            filter_already_liked_items=False
        )

        # ALS 점수 정규화
        if als_scores.max() > als_scores.min():
            als_scores_norm = (als_scores - als_scores.min()) / (als_scores.max() - als_scores.min())
        else:
            als_scores_norm = np.ones_like(als_scores)

        # 다중 신호 융합
        final_scores = []

        for i, (item_idx, als_score) in enumerate(zip(item_ids, als_scores_norm)):
            item_id = self.idx_to_item[item_idx]

            # 콘텐츠 점수
            content_score = self.get_content_score(user_id, item_id)

            # Co-view 점수
            coview_score = self.get_coview_score(user_id, item_idx)

            # 가중 융합
            final = (
                self.als_weight * als_score +
                self.content_weight * content_score +
                self.coview_weight * coview_score
            )

            final_scores.append((item_id, final))

        # 정렬 및 Top-K
        final_scores.sort(key=lambda x: -x[1])
        return [item_id for item_id, _ in final_scores[:top_k]]

    def _get_popular_items(self, top_k: int) -> list:
        """인기 아이템 반환"""
        item_counts = np.array(self.interaction_matrix.sum(axis=0)).flatten()
        top_indices = np.argsort(-item_counts)[:top_k]
        return [self.idx_to_item[i] for i in top_indices]

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
    parser.add_argument('--output', type=str, default='output_ontology.csv')
    parser.add_argument('--content_weight', type=float, default=0.1)
    parser.add_argument('--coview_weight', type=float, default=0.1)
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
    print("Ontology-based Recommender (No LLM)")
    print("="*60)
    print(f"\nWeights: ALS={1-args.content_weight-args.coview_weight:.1f}, "
          f"Content={args.content_weight:.1f}, CoView={args.coview_weight:.1f}")

    # 데이터 로드
    print("\n[1/3] Loading data...")
    train_df = pd.read_parquet(os.path.join(args.data_dir, 'train.parquet'))

    # sample_submission.csv에서 테스트 유저 추출
    sample_sub = pd.read_csv(os.path.join(args.data_dir, 'sample_submission.csv'))
    test_users = sample_sub['user_id'].unique()
    test_df = pd.DataFrame({'user_id': test_users})

    print(f"  Train: {len(train_df):,} interactions")
    print(f"  Test: {len(test_users):,} users")

    # 모델 학습
    print("\n[2/3] Training model...")
    model = OntologyRecommender(
        als_params=als_params,
        content_weight=args.content_weight,
        coview_weight=args.coview_weight
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

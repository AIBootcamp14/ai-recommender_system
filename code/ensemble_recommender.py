"""
ALS + LLM Embedding Ensemble Recommender
- ALS 기반 후보 생성
- LLM 임베딩 기반 리랭킹
- 카테고리 연관도 부스팅
"""
import os
import argparse
import pickle
import json
import numpy as np
import pandas as pd
from scipy import sparse
from implicit.als import AlternatingLeastSquares
from tqdm import tqdm


class EnsembleRecommender:
    """ALS + LLM 임베딩 앙상블 추천기"""

    def __init__(self, als_params: dict = None, llm_weight: float = 0.2, category_boost: float = 0.1):
        self.als_params = als_params or {
            'factors': 32,
            'regularization': 0.0215,
            'alpha': 7,
            'iterations': 15
        }
        self.llm_weight = llm_weight
        self.category_boost = category_boost

        self.model = None
        self.item_embeddings = None
        self.category_associations = None
        self.item_to_category = {}
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_item = {}

    def load_llm_data(self, embeddings_path: str, associations_path: str = None):
        """LLM 생성 데이터 로드"""
        # 임베딩 로드
        if os.path.exists(embeddings_path):
            with open(embeddings_path, 'rb') as f:
                self.item_embeddings = pickle.load(f)
            print(f"  Loaded {len(self.item_embeddings)} item embeddings")
        else:
            print(f"  Warning: {embeddings_path} not found")

        # 카테고리 연관 지식 로드
        if associations_path and os.path.exists(associations_path):
            with open(associations_path, 'r') as f:
                self.category_associations = json.load(f)
            print(f"  Loaded {len(self.category_associations)} category associations")

    def fit(self, df: pd.DataFrame):
        """모델 학습"""
        print("\n[1/3] Building mappings...")

        # 매핑 생성
        users = df['user_id'].unique()
        items = df['item_id'].unique()

        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.item_to_idx = {it: i for i, it in enumerate(items)}
        self.idx_to_item = {i: it for it, i in self.item_to_idx.items()}

        # 아이템-카테고리 매핑
        item_cat = df[['item_id', 'category_code']].drop_duplicates('item_id')
        self.item_to_category = dict(zip(item_cat['item_id'], item_cat['category_code'].fillna('unknown')))

        n_users = len(users)
        n_items = len(items)
        print(f"  Users: {n_users:,}, Items: {n_items:,}")

        print("\n[2/3] Creating sparse matrix...")

        # Sparse matrix 생성
        rows = df['user_id'].map(self.user_to_idx).values
        cols = df['item_id'].map(self.item_to_idx).values
        data = np.ones(len(df))

        user_item = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(n_users, n_items)
        )

        # Confidence 적용
        user_item_conf = (user_item * self.als_params['alpha']).astype('double')

        print("\n[3/3] Training ALS model...")

        self.model = AlternatingLeastSquares(
            factors=self.als_params['factors'],
            regularization=self.als_params['regularization'],
            iterations=self.als_params['iterations'],
            random_state=42
        )
        self.model.fit(user_item_conf)

        # 사용자별 상호작용 저장
        self.user_item_matrix = user_item

        print("  Training complete!")

    def _compute_llm_similarity(self, user_history_items: list, candidate_items: list) -> dict:
        """LLM 임베딩 기반 유사도 계산"""
        if not self.item_embeddings:
            return {}

        # 사용자 히스토리 임베딩 평균
        history_embeddings = []
        for item_id in user_history_items:
            if item_id in self.item_embeddings:
                history_embeddings.append(self.item_embeddings[item_id])

        if not history_embeddings:
            return {}

        user_embedding = np.mean(history_embeddings, axis=0)
        user_embedding = user_embedding / np.linalg.norm(user_embedding)

        # 후보 아이템과 유사도 계산
        similarities = {}
        for item_id in candidate_items:
            if item_id in self.item_embeddings:
                item_emb = np.array(self.item_embeddings[item_id])
                item_emb = item_emb / np.linalg.norm(item_emb)
                sim = np.dot(user_embedding, item_emb)
                similarities[item_id] = sim

        return similarities

    def _compute_category_boost(self, user_history_items: list, candidate_items: list) -> dict:
        """카테고리 연관도 부스팅"""
        if not self.category_associations:
            return {}

        # 사용자 히스토리 카테고리
        user_categories = set()
        for item_id in user_history_items:
            if item_id in self.item_to_category:
                user_categories.add(self.item_to_category[item_id])

        # 보완재 카테고리 추출
        complementary_cats = set()
        for cat in user_categories:
            if cat in self.category_associations:
                complementary_cats.update(
                    self.category_associations[cat].get('complementary', [])
                )

        # 후보 아이템 부스트 계산
        boosts = {}
        for item_id in candidate_items:
            item_cat = self.item_to_category.get(item_id, '')
            if item_cat in complementary_cats:
                boosts[item_id] = 1.0
            elif item_cat in user_categories:
                boosts[item_id] = 0.5  # 같은 카테고리는 절반 부스트

        return boosts

    def recommend(self, user_id: str, top_k: int = 10, als_candidates: int = 100) -> list:
        """앙상블 추천"""
        if user_id not in self.user_to_idx:
            # Cold-start: 인기 아이템 반환
            return self._get_popular_items(top_k)

        user_idx = self.user_to_idx[user_id]

        # 사용자가 이미 본 아이템
        seen_items = set(self.user_item_matrix[user_idx].indices)
        seen_item_ids = [self.idx_to_item[i] for i in seen_items]

        # Stage 1: ALS 후보 생성
        user_vec = self.model.user_factors[user_idx]
        scores = self.model.item_factors.dot(user_vec)

        # 본 아이템 제외
        scores[list(seen_items)] = -np.inf

        # 상위 후보
        top_indices = np.argsort(scores)[::-1][:als_candidates]
        candidates = [(self.idx_to_item[i], scores[i]) for i in top_indices]

        # Stage 2: LLM 유사도 계산
        candidate_ids = [c[0] for c in candidates]
        llm_sims = self._compute_llm_similarity(seen_item_ids, candidate_ids)

        # Stage 3: 카테고리 부스트
        cat_boosts = self._compute_category_boost(seen_item_ids, candidate_ids)

        # Stage 4: 최종 점수 계산
        final_scores = []
        for item_id, als_score in candidates:
            # 정규화
            als_norm = (als_score - scores[scores > -np.inf].min()) / \
                       (scores[scores > -np.inf].max() - scores[scores > -np.inf].min() + 1e-8)

            llm_sim = llm_sims.get(item_id, 0)
            cat_boost = cat_boosts.get(item_id, 0)

            # 앙상블 점수
            final = (1 - self.llm_weight) * als_norm + \
                    self.llm_weight * llm_sim + \
                    self.category_boost * cat_boost

            final_scores.append((item_id, final))

        # 정렬 및 반환
        final_scores.sort(key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in final_scores[:top_k]]

    def _get_popular_items(self, top_k: int) -> list:
        """인기 아이템 반환"""
        item_popularity = np.array(self.user_item_matrix.sum(axis=0)).flatten()
        top_indices = np.argsort(item_popularity)[::-1][:top_k]
        return [self.idx_to_item[i] for i in top_indices]

    def generate_submission(self, test_df: pd.DataFrame, output_path: str, top_k: int = 10):
        """제출 파일 생성 (Long format: 각 행에 user_id, item_id 1개씩)"""
        print("\nGenerating recommendations...")

        results = []
        test_users = test_df['user_id'].unique()

        for user_id in tqdm(test_users, desc="Recommending"):
            recs = self.recommend(user_id, top_k=top_k)
            # Long format: 각 추천 아이템을 별도 행으로
            for item_id in recs:
                results.append({
                    'user_id': user_id,
                    'item_id': item_id
                })

        submission = pd.DataFrame(results)
        submission.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")
        print(f"  Total rows: {len(submission):,} ({len(test_users):,} users × {top_k} items)")

        return submission


def main():
    parser = argparse.ArgumentParser(description='Ensemble Recommender')
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--llm_dir', type=str, default='./llm/llm_output/')
    parser.add_argument('--output_dir', type=str, default='../out/')
    parser.add_argument('--output', type=str, default='output_ensemble.csv')
    parser.add_argument('--llm_weight', type=float, default=0.2, help='LLM embedding weight')
    parser.add_argument('--category_boost', type=float, default=0.1, help='Category association boost')
    parser.add_argument('--als_candidates', type=int, default=100, help='ALS candidate pool size')
    parser.add_argument('--top_k', type=int, default=10)

    # ALS 파라미터
    parser.add_argument('--num_factor', type=int, default=32)
    parser.add_argument('--regularization', type=float, default=0.0215)
    parser.add_argument('--alpha', type=int, default=7)
    parser.add_argument('--iterations', type=int, default=15)

    args = parser.parse_args()

    print("=" * 60)
    print("ALS + LLM Ensemble Recommender")
    print("=" * 60)

    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output)

    # 데이터 로드
    print("\n[1/5] Loading data...")
    train_df = pd.read_parquet(os.path.join(args.data_dir, 'train.parquet'))

    # sample_submission에서 테스트 사용자 추출
    sample_sub = pd.read_csv(os.path.join(args.data_dir, 'sample_submission.csv'))
    test_users = sample_sub['user_id'].unique()
    test_df = pd.DataFrame({'user_id': test_users})

    print(f"  Train: {len(train_df):,} interactions")
    print(f"  Test: {len(test_users):,} users")

    # 앙상블 추천기 초기화
    print("\n[2/5] Initializing ensemble recommender...")
    als_params = {
        'factors': args.num_factor,
        'regularization': args.regularization,
        'alpha': args.alpha,
        'iterations': args.iterations
    }
    print(f"  ALS params: {als_params}")
    print(f"  LLM weight: {args.llm_weight}")
    print(f"  Category boost: {args.category_boost}")

    recommender = EnsembleRecommender(
        als_params=als_params,
        llm_weight=args.llm_weight,
        category_boost=args.category_boost
    )

    # LLM 데이터 로드
    print("\n[3/5] Loading LLM data...")
    recommender.load_llm_data(
        embeddings_path=os.path.join(args.llm_dir, 'item_embeddings.pkl'),
        associations_path=os.path.join(args.llm_dir, 'category_associations.json')
    )

    # 모델 학습
    print("\n[4/5] Training model...")
    recommender.fit(train_df)

    # 추천 생성
    print("\n[5/5] Generating recommendations...")
    recommender.generate_submission(
        test_df,
        output_path=output_path,
        top_k=args.top_k
    )

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()

"""
Item2Vec 기반 추천 시스템
- 세션을 문장, 아이템을 단어로 취급
- Word2Vec(Skip-gram)으로 아이템 임베딩 학습
- ALS + Item2Vec 하이브리드
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from gensim.models import Word2Vec
from collections import defaultdict
from tqdm import tqdm
import argparse
import os


class Item2VecRecommender:
    """Item2Vec + ALS 하이브리드 추천기"""

    def __init__(self, als_params: dict, i2v_weight: float = 0.3):
        self.als_params = als_params
        self.i2v_weight = i2v_weight
        self.als_weight = 1.0 - i2v_weight

        self.als_model = None
        self.i2v_model = None

        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}

        self.user_items = {}  # user_id -> list of item_ids
        self.item_popularity = {}
        self.global_popular_items = []

    def fit(self, train_df: pd.DataFrame):
        """모델 학습"""
        print("\n" + "="*60)
        print("Item2Vec Recommender - Training")
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

        # 2. 사용자별 아이템 히스토리 구축
        print("\n[2/5] Building user histories...")
        user_groups = train_df.groupby('user_id')['item_id'].apply(list)
        self.user_items = user_groups.to_dict()
        print(f"  User histories: {len(self.user_items):,}")

        # 3. 세션 시퀀스 추출 (Item2Vec 학습용)
        print("\n[3/5] Extracting session sequences...")
        sessions = train_df.groupby('user_session')['item_id'].apply(list)
        # 2개 이상 아이템이 있는 세션만 사용
        session_sequences = [list(s) for s in sessions if len(s) >= 2]
        print(f"  Valid sessions: {len(session_sequences):,}")

        # 아이템을 문자열로 변환 (Word2Vec 요구사항)
        session_sequences_str = [[str(item) for item in seq] for seq in session_sequences]

        # 4. Item2Vec 학습
        print("\n[4/5] Training Item2Vec...")
        self.i2v_model = Word2Vec(
            sentences=session_sequences_str,
            vector_size=64,
            window=5,
            min_count=2,
            workers=4,
            sg=1,  # Skip-gram
            epochs=10
        )
        print(f"  Item2Vec vocabulary: {len(self.i2v_model.wv):,}")

        # 5. ALS 학습
        print("\n[5/5] Training ALS model...")
        self._train_als(train_df)

        # 인기 아이템 계산
        self._compute_popularity(train_df)

        print("\nTraining complete!")

    def _train_als(self, train_df: pd.DataFrame):
        """ALS 모델 학습"""
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

    def _compute_popularity(self, train_df: pd.DataFrame):
        """인기도 계산"""
        # 구매 기준 인기도
        purchase_counts = train_df[train_df['event_type'] == 'purchase']['item_id'].value_counts()
        self.global_popular_items = purchase_counts.head(100).index.tolist()

        # 전체 인기도
        item_counts = train_df['item_id'].value_counts()
        max_count = item_counts.max()
        self.item_popularity = (item_counts / max_count).to_dict()

    def get_i2v_recommendations(self, user_id: str, n_candidates: int = 100) -> list:
        """Item2Vec 기반 추천"""
        if user_id not in self.user_items:
            return []

        user_history = self.user_items[user_id]

        # 사용자가 본 아이템 중 Item2Vec에 있는 것만
        valid_items = [str(item) for item in user_history if str(item) in self.i2v_model.wv]

        if not valid_items:
            return []

        # 최근 아이템에 가중치 부여 (최근 5개 더 중요)
        recent_items = valid_items[-5:] if len(valid_items) > 5 else valid_items

        # 유사 아이템 찾기
        try:
            similar_items = self.i2v_model.wv.most_similar(
                positive=recent_items,
                topn=n_candidates + len(user_history)
            )
        except:
            return []

        # 이미 본 아이템 제외
        user_history_set = set(str(item) for item in user_history)
        recommendations = []

        for item_str, score in similar_items:
            if item_str not in user_history_set:
                try:
                    item_id = int(item_str)
                    if item_id in self.item_to_idx:
                        recommendations.append((item_id, score))
                except:
                    continue

            if len(recommendations) >= n_candidates:
                break

        return recommendations

    def recommend(self, user_id: str, top_k: int = 10) -> list:
        """추천 생성"""
        if user_id not in self.user_to_idx:
            # Cold-start: 인기 아이템 반환
            return self.global_popular_items[:top_k]

        user_idx = self.user_to_idx[user_id]

        # ALS 후보 생성
        n_candidates = 100
        try:
            als_item_ids, als_scores = self.als_model.recommend(
                user_idx,
                self.interaction_matrix[user_idx],
                N=n_candidates,
                filter_already_liked_items=False
            )
        except:
            return self.global_popular_items[:top_k]

        # ALS 점수 정규화
        if als_scores.max() > als_scores.min():
            als_scores_norm = (als_scores - als_scores.min()) / (als_scores.max() - als_scores.min())
        else:
            als_scores_norm = np.ones_like(als_scores)

        # ALS 결과를 dict로
        als_scores_dict = {
            self.idx_to_item[als_item_ids[i]]: als_scores_norm[i]
            for i in range(len(als_item_ids))
        }

        # Item2Vec 추천
        i2v_recs = self.get_i2v_recommendations(user_id, n_candidates)

        # Item2Vec 점수 정규화
        i2v_scores_dict = {}
        if i2v_recs:
            i2v_max = max(score for _, score in i2v_recs)
            i2v_min = min(score for _, score in i2v_recs)
            if i2v_max > i2v_min:
                for item_id, score in i2v_recs:
                    i2v_scores_dict[item_id] = (score - i2v_min) / (i2v_max - i2v_min)
            else:
                for item_id, score in i2v_recs:
                    i2v_scores_dict[item_id] = 1.0

        # 모든 후보 아이템 수집
        all_candidates = set(als_scores_dict.keys()) | set(i2v_scores_dict.keys())

        # 최종 점수 계산
        final_scores = []
        for item_id in all_candidates:
            als_score = als_scores_dict.get(item_id, 0)
            i2v_score = i2v_scores_dict.get(item_id, 0)

            final = self.als_weight * als_score + self.i2v_weight * i2v_score
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
    parser.add_argument('--output', type=str, default='output_item2vec.csv')
    parser.add_argument('--i2v_weight', type=float, default=0.3)
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
    print("Item2Vec Recommender (ALS + Item2Vec)")
    print("="*60)
    print(f"\nWeights: ALS={1-args.i2v_weight:.1f}, Item2Vec={args.i2v_weight:.1f}")

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
    model = Item2VecRecommender(als_params=als_params, i2v_weight=args.i2v_weight)
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

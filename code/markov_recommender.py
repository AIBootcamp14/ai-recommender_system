"""
Markov Chain 기반 추천 시스템
- 세션 내 아이템 전이 확률 학습
- ALS + Markov 하이브리드
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from implicit.als import AlternatingLeastSquares
from collections import defaultdict
from tqdm import tqdm
import argparse
import os


class MarkovRecommender:
    """Markov Chain + ALS 하이브리드 추천기"""

    def __init__(self, als_params: dict, markov_weight: float = 0.3):
        self.als_params = als_params
        self.markov_weight = markov_weight
        self.als_weight = 1.0 - markov_weight

        self.als_model = None
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}

        # Markov transition matrix
        self.transition_matrix = None  # item_idx -> item_idx 전이 확률

        self.user_items = {}  # user_id -> list of item_ids (최근 순)
        self.global_popular_items = []

    def build_transition_matrix(self, train_df: pd.DataFrame):
        """세션 기반 Markov 전이 행렬 구축"""
        print("\n[Markov] Building transition matrix...")

        n_items = len(self.item_to_idx)

        # 전이 카운트
        transition_counts = defaultdict(lambda: defaultdict(int))

        # 세션별 시퀀스 추출
        sessions = train_df.groupby('user_session')['item_id'].apply(list)
        valid_sessions = sessions[sessions.apply(len) >= 2]

        print(f"  Valid sessions: {len(valid_sessions):,}")

        for items in tqdm(valid_sessions, desc="  Counting transitions"):
            for i in range(len(items) - 1):
                curr_item = items[i]
                next_item = items[i + 1]

                if curr_item in self.item_to_idx and next_item in self.item_to_idx:
                    curr_idx = self.item_to_idx[curr_item]
                    next_idx = self.item_to_idx[next_item]
                    transition_counts[curr_idx][next_idx] += 1

        # Sparse matrix로 변환
        print("  Building sparse matrix...")
        self.transition_matrix = lil_matrix((n_items, n_items), dtype=np.float32)

        for curr_idx, next_items in transition_counts.items():
            total = sum(next_items.values())
            for next_idx, count in next_items.items():
                self.transition_matrix[curr_idx, next_idx] = count / total

        self.transition_matrix = self.transition_matrix.tocsr()

        # 통계
        nnz = self.transition_matrix.nnz
        print(f"  Transition pairs: {nnz:,}")
        print(f"  Avg transitions per item: {nnz / n_items:.1f}")

    def fit(self, train_df: pd.DataFrame):
        """모델 학습"""
        print("\n" + "="*60)
        print("Markov Recommender - Training")
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

        # 2. 사용자별 아이템 히스토리 (최근 순)
        print("\n[2/4] Building user histories...")
        # timestamp가 있으면 정렬, 없으면 그냥 순서대로
        if 'event_time' in train_df.columns:
            sorted_df = train_df.sort_values(['user_id', 'event_time'])
        else:
            sorted_df = train_df

        user_groups = sorted_df.groupby('user_id')['item_id'].apply(list)
        self.user_items = user_groups.to_dict()
        print(f"  User histories: {len(self.user_items):,}")

        # 3. Markov 전이 행렬 구축
        print("\n[3/4] Building Markov transition matrix...")
        self.build_transition_matrix(train_df)

        # 4. ALS 학습
        print("\n[4/4] Training ALS model...")

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

        # 인기 아이템
        self._compute_popularity(train_df)

        print("\nTraining complete!")

    def _compute_popularity(self, train_df: pd.DataFrame):
        """인기도 계산"""
        purchase_counts = train_df[train_df['event_type'] == 'purchase']['item_id'].value_counts()
        self.global_popular_items = purchase_counts.head(100).index.tolist()

    def get_markov_recommendations(self, user_id: str, n_candidates: int = 100) -> list:
        """Markov 기반 추천 (최근 아이템 기반 전이 확률)"""
        if user_id not in self.user_items:
            return []

        user_history = self.user_items[user_id]
        user_history_set = set(user_history)

        # 최근 아이템들 (최대 5개)
        recent_items = user_history[-5:]

        # 각 최근 아이템에서의 전이 확률 집계
        candidate_scores = defaultdict(float)

        for i, item in enumerate(recent_items):
            if item not in self.item_to_idx:
                continue

            item_idx = self.item_to_idx[item]

            # 전이 확률 가져오기
            row = self.transition_matrix[item_idx]
            if row.nnz == 0:
                continue

            # 최근 아이템일수록 높은 가중치
            recency_weight = (i + 1) / len(recent_items)

            for next_idx in row.indices:
                next_item = self.idx_to_item[next_idx]
                if next_item not in user_history_set:
                    prob = row[0, next_idx]
                    candidate_scores[next_item] += prob * recency_weight

        # 점수순 정렬
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: -x[1])
        return [(item, score) for item, score in sorted_candidates[:n_candidates]]

    def recommend(self, user_id: str, top_k: int = 10) -> list:
        """추천 생성"""
        if user_id not in self.user_to_idx:
            # Cold-start
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

        # Markov 추천
        markov_recs = self.get_markov_recommendations(user_id, n_candidates)

        # Markov 점수 정규화
        markov_scores_dict = {}
        if markov_recs:
            markov_max = max(score for _, score in markov_recs)
            markov_min = min(score for _, score in markov_recs)
            if markov_max > markov_min:
                for item_id, score in markov_recs:
                    markov_scores_dict[item_id] = (score - markov_min) / (markov_max - markov_min)
            else:
                for item_id, score in markov_recs:
                    markov_scores_dict[item_id] = 1.0

        # 모든 후보 수집
        all_candidates = set(als_scores_dict.keys()) | set(markov_scores_dict.keys())

        # 최종 점수 계산
        final_scores = []
        for item_id in all_candidates:
            als_score = als_scores_dict.get(item_id, 0)
            markov_score = markov_scores_dict.get(item_id, 0)

            final = self.als_weight * als_score + self.markov_weight * markov_score
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
    parser.add_argument('--output', type=str, default='output_markov.csv')
    parser.add_argument('--markov_weight', type=float, default=0.3)
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
    print("Markov Chain Recommender (ALS + Markov)")
    print("="*60)
    print(f"\nWeights: ALS={1-args.markov_weight:.1f}, Markov={args.markov_weight:.1f}")

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
    model = MarkovRecommender(als_params=als_params, markov_weight=args.markov_weight)
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

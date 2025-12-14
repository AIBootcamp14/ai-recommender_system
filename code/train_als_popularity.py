"""
ALS + 인기도 폴백 추천
- Cold-start 사용자(52%)에게 인기 아이템 추천
- 기존 사용자에게는 ALS 개인화 추천
"""
import argparse
import os

import pandas as pd
import numpy as np
from scipy import sparse
from implicit.als import AlternatingLeastSquares
from tqdm import tqdm
from utils import set_seed


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="train.parquet", type=str)
    parser.add_argument("--dir_path", default="../data/", type=str)
    parser.add_argument("--output_dir", default="../output/", type=str)
    parser.add_argument("--submission_file", default="sample_submission.csv", type=str)

    # model args
    parser.add_argument("--num_factor", type=int, default=32)
    parser.add_argument("--regularization", type=float, default=0.021466)
    parser.add_argument("--alpha", type=int, default=7)

    # train args
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 50)
    print("ALS + 인기도 폴백 추천")
    print("=" * 50)

    # 1. 데이터 로드
    print("\n[1/5] 데이터 로드...")
    train_df = pd.read_parquet(os.path.join(args.dir_path, args.data_dir))
    submission_df = pd.read_csv(os.path.join(args.dir_path, args.submission_file))

    # 제출 대상 사용자
    submission_users = submission_df['user_id'].unique()
    print(f"  Train 상호작용: {len(train_df):,}")
    print(f"  제출 대상 사용자: {len(submission_users):,}")

    # 2. 인기 아이템 계산
    print("\n[2/5] 인기 아이템 계산...")
    item_popularity = train_df.groupby('item_id').size().sort_values(ascending=False)
    popular_items = item_popularity.head(100).index.tolist()
    print(f"  Top 10 인기 아이템 상호작용 수:")
    for i, item in enumerate(popular_items[:10]):
        print(f"    {i+1}. {item[:20]}... ({item_popularity[item]:,}회)")

    # 3. 인덱싱 및 ALS 학습
    print("\n[3/5] ALS 모델 학습...")
    user2idx = {v: k for k, v in enumerate(train_df['user_id'].unique())}
    idx2user = {k: v for k, v in enumerate(train_df['user_id'].unique())}
    item2idx = {v: k for k, v in enumerate(train_df['item_id'].unique())}
    idx2item = {k: v for k, v in enumerate(train_df['item_id'].unique())}

    train_df['user_idx'] = train_df['user_id'].map(user2idx)
    train_df['item_idx'] = train_df['item_id'].map(item2idx)

    train_df["label"] = 1
    user_item_matrix = train_df.groupby(["user_idx", "item_idx"])["label"].sum().reset_index()

    sparse_user_item = sparse.csr_matrix(
        (user_item_matrix["label"].values,
         (user_item_matrix["user_idx"].values,
          user_item_matrix["item_idx"].values)),
        shape=(len(user2idx), len(item2idx)),
        dtype=np.float32
    )

    model = AlternatingLeastSquares(
        factors=args.num_factor,
        regularization=args.regularization,
        alpha=args.alpha,
        use_gpu=False
    )
    model.fit(sparse_user_item)

    # 4. Cold-start 분석
    print("\n[4/5] Cold-start 분석...")
    known_users = set(train_df['user_id'].unique())
    cold_users = [u for u in submission_users if u not in known_users]
    warm_users = [u for u in submission_users if u in known_users]

    print(f"  Known 사용자 (ALS 추천): {len(warm_users):,} ({len(warm_users)/len(submission_users)*100:.1f}%)")
    print(f"  Cold-start 사용자 (인기도 추천): {len(cold_users):,} ({len(cold_users)/len(submission_users)*100:.1f}%)")

    # 5. 추천 생성
    print("\n[5/5] 추천 생성...")
    results = []

    # 5-1. Warm users: ALS 추천
    print(f"  ALS 추천 생성 중... ({len(warm_users):,} users)")
    warm_user_idx = np.array([user2idx[u] for u in warm_users])
    als_recommendations = model.recommend(
        warm_user_idx,
        sparse_user_item[warm_user_idx],
        N=10,
        filter_already_liked_items=False
    )

    for i, user_id in enumerate(tqdm(warm_users, desc="  Warm users")):
        for item_idx in als_recommendations[0][i]:
            results.append({
                'user_id': user_id,
                'item_id': idx2item[item_idx]
            })

    # 5-2. Cold users: 인기도 추천
    print(f"  인기도 추천 생성 중... ({len(cold_users):,} users)")
    for user_id in tqdm(cold_users, desc="  Cold users"):
        for item_id in popular_items[:10]:
            results.append({
                'user_id': user_id,
                'item_id': item_id
            })

    # 결과 저장
    result_df = pd.DataFrame(results)

    outdir = args.output_dir
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    output_path = os.path.join(outdir, "output_als_popularity.csv")
    result_df.to_csv(output_path, index=False)

    print("\n" + "=" * 50)
    print("완료!")
    print("=" * 50)
    print(f"  출력 파일: {output_path}")
    print(f"  총 행 수: {len(result_df):,}")
    print(f"  사용자 수: {result_df['user_id'].nunique():,}")
    print(f"  사용자당 추천: {len(result_df) // result_df['user_id'].nunique()}")


if __name__ == "__main__":
    main()

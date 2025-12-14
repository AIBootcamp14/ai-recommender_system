"""
ALS with event_type 가중치
- view: 1
- cart: 5
- purchase: 10
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

    # model args (Optuna 최적값)
    parser.add_argument("--num_factor", type=int, default=32)
    parser.add_argument("--regularization", type=float, default=0.021466)
    parser.add_argument("--alpha", type=int, default=7)

    # event_type 가중치
    parser.add_argument("--weight_view", type=float, default=1.0)
    parser.add_argument("--weight_cart", type=float, default=5.0)
    parser.add_argument("--weight_purchase", type=float, default=10.0)

    # train args
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 50)
    print("ALS with event_type 가중치")
    print("=" * 50)
    print(f"  view: {args.weight_view}")
    print(f"  cart: {args.weight_cart}")
    print(f"  purchase: {args.weight_purchase}")

    # 1. 데이터 로드
    print("\n[1/4] 데이터 로드...")
    train_df = pd.read_parquet(os.path.join(args.dir_path, args.data_dir))
    print(f"  총 상호작용: {len(train_df):,}")

    # event_type 분포 확인
    print("\n  Event type 분포:")
    event_counts = train_df['event_type'].value_counts()
    for event, count in event_counts.items():
        print(f"    {event}: {count:,} ({count/len(train_df)*100:.2f}%)")

    # 2. 인덱싱
    print("\n[2/4] 인덱싱...")
    user2idx = {v: k for k, v in enumerate(train_df['user_id'].unique())}
    idx2user = {k: v for k, v in enumerate(train_df['user_id'].unique())}
    item2idx = {v: k for k, v in enumerate(train_df['item_id'].unique())}
    idx2item = {k: v for k, v in enumerate(train_df['item_id'].unique())}

    train_df['user_idx'] = train_df['user_id'].map(user2idx)
    train_df['item_idx'] = train_df['item_id'].map(item2idx)

    # 3. event_type 가중치 적용
    print("\n[3/4] event_type 가중치 적용...")
    weight_map = {
        'view': args.weight_view,
        'cart': args.weight_cart,
        'purchase': args.weight_purchase
    }
    train_df["label"] = train_df["event_type"].map(weight_map)

    # 가중치 적용 후 통계
    print(f"  가중치 적용 후 총 신호: {train_df['label'].sum():,.0f}")
    print(f"  평균 가중치: {train_df['label'].mean():.4f}")

    # user-item matrix 생성
    user_item_matrix = train_df.groupby(["user_idx", "item_idx"])["label"].sum().reset_index()

    sparse_user_item = sparse.csr_matrix(
        (user_item_matrix["label"].values,
         (user_item_matrix["user_idx"].values,
          user_item_matrix["item_idx"].values)),
        shape=(len(user2idx), len(item2idx)),
        dtype=np.float32
    )

    # 4. ALS 학습
    print("\n[4/4] ALS 모델 학습...")
    model = AlternatingLeastSquares(
        factors=args.num_factor,
        regularization=args.regularization,
        alpha=args.alpha,
        use_gpu=False
    )
    model.fit(sparse_user_item)

    # 5. 추천 생성
    print("\n추천 생성...")
    test_users_idx = np.array(train_df['user_idx'].unique())
    test_users_idx_li = [num for num in test_users_idx for _ in range(10)]

    public_outputs = model.recommend(
        test_users_idx,
        sparse_user_item[test_users_idx],
        N=10,
        filter_already_liked_items=False
    )

    recommend_items = public_outputs[0]
    sub_df = pd.DataFrame({
        'user_id': test_users_idx_li,
        'item_id': recommend_items.flatten()
    })
    sub_df['user_id'] = sub_df['user_id'].map(idx2user)
    sub_df['item_id'] = sub_df['item_id'].map(idx2item)

    # 저장
    outdir = args.output_dir
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    output_path = os.path.join(outdir, "output_weighted.csv")
    sub_df.to_csv(output_path, index=False)

    print("\n" + "=" * 50)
    print("완료!")
    print("=" * 50)
    print(f"  출력 파일: {output_path}")
    print(f"  총 행 수: {len(sub_df):,}")


if __name__ == "__main__":
    main()

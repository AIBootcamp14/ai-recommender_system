import argparse
import os

import pandas as pd
import numpy as np
from scipy import sparse
from implicit.als import AlternatingLeastSquares
from tqdm import tqdm
from utils import *


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="train.parquet", type=str)
    parser.add_argument("--dir_path", default="../data/", type=str)
    parser.add_argument("--output_dir", default="../output/", type=str)

    # model args
    parser.add_argument("--num_factor", help="The number of latent factors to compute", type=int,default=32)
    parser.add_argument(
        "--regularization", type=float, default=0.001, help="The regularization factor to use"
    )
    parser.add_argument(
        "--alpha", type=int, default=10, help="governs the baseline confidence in preference observations"
    )

    # train args
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    set_seed(args.seed)

    train_df = pd.read_parquet(os.path.join(args.dir_path, args.data_dir))

    user2idx = {v: k for k, v in enumerate(train_df['user_id'].unique())}
    idx2user = {k: v for k, v in enumerate(train_df['user_id'].unique())}
    item2idx = {v: k for k, v in enumerate(train_df['item_id'].unique())}
    idx2item = {k: v for k, v in enumerate(train_df['item_id'].unique())}

    # Apply the mapping functions to 'user_id' and 'item_id' columns
    train_df['user_idx'] = train_df['user_id'].map(user2idx)
    train_df['item_idx'] = train_df['item_id'].map(item2idx)

    # use the same confidence score for all event_types
    train_df["label"] = 1
    user_item_matrix = train_df.groupby(["user_idx", "item_idx"])["label"].sum().reset_index()

    sparse_user_item = sparse.csr_matrix(
                                        (user_item_matrix["label"].values,
                                        (user_item_matrix["user_idx"].values,
                                        user_item_matrix["item_idx"].values)),
                                        shape=(len(user2idx), len(item2idx)),
                                        dtype=np.float32)
    sparse_user_item = sparse_user_item.tocsr()
    # ref: https://github.com/benfred/implicit/blob/main/examples/movielens.py
    num_factor = args.num_factor
    regularization = args.regularization
    alpha = args.alpha

    model = AlternatingLeastSquares(
        factors=num_factor,
        regularization=regularization,
        alpha=alpha,
        use_gpu=False)
    
    model.fit(sparse_user_item)

    test_users_idx = np.array(train_df['user_idx'].unique())
    test_users_idx_li = [num for num in test_users_idx for _ in range(10)]
    public_outputs = model.recommend(test_users_idx, sparse_user_item[test_users_idx], N=10, filter_already_liked_items=False)

    recommend_items = public_outputs[0]
    sub_df = pd.DataFrame({'user_id' : test_users_idx_li, 'item_id' : recommend_items.flatten()})
    sub_df['user_id'] = sub_df['user_id'].map(idx2user)
    sub_df['item_id'] = sub_df['item_id'].map(idx2item)

    
    outdir = args.output_dir
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    sub_df.to_csv(os.path.join(outdir,"output.csv"), index=False)

if __name__ == "__main__":
    main()

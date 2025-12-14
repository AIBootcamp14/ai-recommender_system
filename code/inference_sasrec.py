from recbole.quick_start.quick_start import load_data_and_model
import numpy as np
import pandas as pd
import os
from collections import defaultdict
from recbole.utils.case_study import full_sort_topk
from tqdm import tqdm
import argparse
import json

from utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset", default="train.parquet", type=str)
    parser.add_argument("--data_dir", default="../data/", type=str)
    parser.add_argument("--output_dir", default="../output/", type=str)
    parser.add_argument("--model_file", default="./saved/SASRec.pth", type=str)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    set_seed(args.seed)

    train = pd.read_parquet(os.path.join(args.data_dir, args.train_dataset))

    train = train.sort_values(by=['user_session','event_time'])

    with open(os.path.join(args.data_dir,'user2idx.json'),"r") as f_user:
        user2idx = json.load(f_user)

    with open(os.path.join(args.data_dir,'item2idx.json'),"r") as f_item:
        item2idx = json.load(f_item)

    idx2user = {k: v for k, v in enumerate(train['user_id'].unique())}
    idx2item = {k: v for k, v in enumerate(train['item_id'].unique())}


    # Apply the mapping functions to 'user_id' and 'item_id' columns
    train['user_idx'] = train['user_id'].map(user2idx)
    train['item_idx'] = train['item_id'].map(item2idx)

    users = defaultdict(list) # defaultdict은 dictionary의 key가 없을때 default 값을 value로 반환
    for u, i in zip(train['user_idx'], train['item_idx']):
        users[u].append(i)

    config, model, dataset, _ , _, test_data = load_data_and_model(
        model_file=args.model_file,
    )
    print('Data and model load compelete')

    popular_top_10 = train.groupby('item_idx').count().rename(columns = {"user_idx": "user_counts"}).sort_values(by=['user_counts', 'item_idx'], ascending=[False, True])[:10].index
    result = []
    
    # short history user에 대해선 popular로 처리
    for uid in tqdm(users):
        if str(uid) in dataset.field2token_id['user_idx']:
            recbole_id = dataset.token2id(dataset.uid_field, str(uid))
            topk_score, topk_iid_list = full_sort_topk([recbole_id], model, test_data, k=10, device=config['device'])
            predicted_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())
            predicted_item_list = predicted_item_list[-1]
            predicted_item_list = list(map(int,predicted_item_list))
        else:
            predicted_item_list = list(popular_top_10)

        for iid in predicted_item_list:
            result.append((idx2user[uid], idx2item[iid]))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    pd.DataFrame(result, columns=["user_id", "item_id"]).to_csv(
        os.path.join(args.output_dir,"output.csv"), index=False
    )

if __name__ == "__main__":
    main()
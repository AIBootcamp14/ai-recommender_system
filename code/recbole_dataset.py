import argparse
import pandas as pd
from collections import defaultdict
import os 
from utils import *
import json



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data", type=str)
    parser.add_argument("--train_dataset", default="train.parquet", type=str)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    set_seed(args.seed)
    train = pd.read_parquet(os.path.join(args.data_dir, args.train_dataset))
    train['event_time'] = pd.to_datetime(train['event_time'], format='%Y-%m-%d %H:%M:%S %Z')
    train = train.sort_values(by=['user_session','event_time'])

    train_df = train[['user_id','item_id','user_session','event_time']]
    train_df['event_time'] = train_df['event_time'].values.astype(float)

    user2idx = {v: k for k, v in enumerate(train_df['user_id'].unique())}
    item2idx = {v: k for k, v in enumerate(train_df['item_id'].unique())}

    with open(os.path.join(args.data_dir,'user2idx.json'),"w") as f_user:
        json.dump(user2idx,f_user)


    with open(os.path.join(args.data_dir,'item2idx.json'),"w") as f_item:
        json.dump(item2idx,f_item)

    # Apply the mapping functions to 'user_id' and 'item_id' columns
    train_df['user_idx'] = train_df['user_id'].map(user2idx)
    train_df['item_idx'] = train_df['item_id'].map(item2idx)


    train_df = train_df.dropna().reset_index(drop=True)
    train_df.rename(columns={'user_idx': 'user_idx:token', 'item_idx': 'item_idx:token', 'event_time': 'event_time:float'},inplace=True)
    outdir = os.path.join(args.data_dir,'SASRec_dataset')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    train_df[['user_idx:token', 'item_idx:token', 'event_time:float']].to_csv(os.path.join(outdir,'SASRec_dataset.inter'), sep='\t',index=None)
    print('Recbole dataset generated')

if __name__ == "__main__":
    main()

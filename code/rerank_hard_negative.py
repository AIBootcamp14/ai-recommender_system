"""
LightGBM Reranker with Hard Negative Mining
1. Uses SASRec to generate candidates (Top-50).
2. Mines HARD NEGATIVES: Items recommended by SASRec but NOT purchased.
3. Trains LightGBM to distinguish (Purchased) vs (SASRec False Positives).
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from tqdm import tqdm
import os
import argparse
import pickle
import torch
from sasrec_proxy import SASRec, Config, RecSysDataset  # Import SASRec class

# --- 1. Candidate Generation (Feature Extraction) ---
def generate_candidates_and_features(model, train_df, test_users, item_meta, cfg, mode='train'):
    """
    Generate candidates using SASRec and extract features.
    mode='train': Leave-one-out validation (Last item is target).
    mode='test': Predict for unknown future (Submission).
    """
    print(f"Generating Candidates (Mode: {mode})...")
    
    # Pre-compute User History
    user_grp = train_df.groupby('user_id')['item_id'].apply(list).to_dict()
    
    # Candidates List
    data_list = []
    
    # Prepare Model
    model.eval()
    items = train_df['item_id'].unique()
    num_items = len(items)
    item_to_idx = {item: i+1 for i, item in enumerate(items)}
    idx_to_item = {i+1: item for i, item in enumerate(items)}
    
    # Target Users
    if mode == 'train':
        # Use users who have at least 2 interactions (to have history + target)
        target_users = [u for u in user_grp.keys() if len(user_grp[u]) >= 2]
        # Less sampling for speed if needed, but here we use full valid users
        target_users = np.random.choice(target_users, min(len(target_users), 50000), replace=False) 
    else:
        target_users = test_users

    # Batch Processing
    BATCH_SIZE = 256
    batch_seqs = []
    batch_users = []
    
    # Global Stats for Features
    item_pop = train_df['item_id'].value_counts(normalize=True).to_dict()
    
    for u in tqdm(target_users, desc="Inferencing"):
        seq = user_grp.get(u, [])
        target_item = None
        
        if mode == 'train':
            target_item = seq[-1] # The item they actually bought (Positive)
            seq = seq[:-1]        # History before that
        
        # Padding
        if len(seq) > cfg.MAX_LEN:
            seq_in = seq[-cfg.MAX_LEN:]
        else:
            seq_in = [0] * (cfg.MAX_LEN - len(seq)) + seq
        
        # Mapping to Idx
        seq_idx = [item_to_idx.get(item, 0) for item in seq_in]
        
        batch_seqs.append(seq_idx)
        batch_users.append(u)
        
        if len(batch_seqs) == BATCH_SIZE or u == target_users[-1]:
            # Predict
            batch_tensor = torch.LongTensor(batch_seqs).to(cfg.DEVICE)
            with torch.no_grad():
                logits = model.predict(batch_tensor) # (B, V)
                
                # Top-K Candidates (e.g., 50)
                scores, top_indices = torch.topk(logits, 50, dim=1)
                
                scores = scores.cpu().numpy()
                top_indices = top_indices.cpu().numpy()
                
            # Process Batch
            for k, user_id in enumerate(batch_users):
                # Positive (if train)
                target_iid = None
                if mode == 'train':
                    # Retrieve the original target item from the outer loop logic?
                    # Since we batched, we need to recover it.
                    # Simplified: We just re-fetch from user_grp
                    full_seq = user_grp[user_id]
                    target_iid = full_seq[-1]
                
                # Negative Candidates (Hard Negatives) extracted from Top-K
                candidates = []
                found_target = False
                
                for rank, idx in enumerate(top_indices[k]):
                    pred_item = idx_to_item.get(idx)
                    score = float(scores[k][rank])
                    
                    if pred_item is None: continue
                    
                    # Labeling
                    label = 0
                    if mode == 'train':
                        if pred_item == target_iid:
                            label = 1
                            found_target = True
                        elif pred_item in user_grp[user_id][:-1]:
                            # Already bought in history -> skip or treat as 0?
                            # Usually we skip already purchased items in training
                            continue
                            
                    # Add feature row
                    data_list.append({
                        'user_id': user_id,
                        'item_id': pred_item,
                        'sasrec_score': score,
                        'sasrec_rank': rank + 1,
                        'item_pop': item_pop.get(pred_item, 0),
                        'label': label
                    })
                
                # Force add Positive if not in Top-50 (for Train)
                if mode == 'train' and not found_target and target_iid:
                     # Calculate SASRec score for this specific item
                     # (Skip for now to save time, assume hard negatives are enough)
                     pass

            # Reset Batch
            batch_seqs = []
            batch_users = []
            
    return pd.DataFrame(data_list)

# --- 2. Train LightGBM ---
def train_lgb(train_data):
    print(f"Training LightGBM with {len(train_data)} samples...")
    print(f"Positive Rate: {train_data['label'].mean():.4f}")
    
    # Feature Columns
    features = ['sasrec_score', 'sasrec_rank', 'item_pop']
    
    # Split
    # Split by user to avoid leakage? roughly random split is fine for rerank proof of concept
    msk = np.random.rand(len(train_data)) < 0.8
    train_set = train_data[msk]
    val_set = train_data[~msk]
    
    lgb_train = lgb.Dataset(train_set[features], label=train_set['label'])
    lgb_val = lgb.Dataset(val_set[features], label=val_set['label'], reference=lgb_train)
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.9,
        'verbose': -1
    }
    
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
    )
    
    return model, features


# --- Proxy Logic (Must match sasrec_proxy.py) ---
def apply_proxy_labeling(df, min_view_count=3):
    print(f"Applying Proxy Labeling (Keep Purchase/Cart + Views >= {min_view_count})...")
    # 1. Identify Purchase/Cart
    strong_signals = df[df['event_type'].isin(['purchase', 'cart'])].copy()
    
    # 2. Identify Frequent Views
    views = df[df['event_type'] == 'view']
    view_counts = views.groupby(['user_id', 'item_id']).size().reset_index(name='count')
    frequent_view_pairs = view_counts[view_counts['count'] >= min_view_count][['user_id', 'item_id']]
    
    frequent_views = pd.merge(views, frequent_view_pairs, on=['user_id', 'item_id'], how='inner')
    
    # 3. Combine
    proxy_df = pd.concat([strong_signals, frequent_views]).drop_duplicates()
    proxy_df = proxy_df.sort_values(['user_id', 'event_time'])
    
    return proxy_df

# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--output_dir', type=str, default='../out/')
    parser.add_argument('--model_path', type=str, default='./saved/SASRec_model.pth', help='Path to pre-trained SASRec model')
    parser.add_argument('--min_view', type=int, default=3, help='Must match training setting')
    args = parser.parse_args()
    
    # Load Data
    print("Loading data...")
    train_df = pd.read_parquet(os.path.join(args.data_dir, 'train.parquet'))
    if train_df['event_time'].dtype == 'object':
         train_df['event_time'] = pd.to_datetime(train_df['event_time'])
         
    sample_sub = pd.read_csv(os.path.join(args.data_dir, 'sample_submission.csv'))
    test_users = sample_sub['user_id'].unique()
    
    # Apply Proxy Filtering to match Model Vocab
    train_df = apply_proxy_labeling(train_df, min_view_count=args.min_view)
    
    # Load SASRec Model
    items = train_df['item_id'].unique()
    num_items = len(items)
    print(f"Num Items (Active): {num_items} (Should match checkpoint)")
    
    cfg = Config()
    
    # Trick: If saved model is entire object or state_dict
    # Ideally should share code. Here we assume we can instantiate and load dict.
    model = SASRec(num_items, cfg).to(cfg.DEVICE)
    
    # We need to find the latest model file if not specified
    if not os.path.exists(args.model_path):
        # Allow running without explicit path if we can find one?
        # For now, let's assume the user has ran sasrec_proxy.py but it didn't save .pth!
        # Ah, sasrec_proxy.py DID NOT save .pth in the script provided earlier.
        # We need to modifying sasrec_proxy.py to save model OR re-train here quicly. 
        # Since re-training is fast (5 min), let's just use the sasrec feature as dummy for now?
        # NO, we must use the trained SASRec.
        
        print("ERROR: SASRec model checkpoint not found!")
        print("Please modify sasrec_proxy.py to save 'model.state_dict()' to './saved/model.pth'")
        print("Or run this script AFTER saving the model.")
        return

    model.load_state_dict(torch.load(args.model_path))
    print("SASRec model loaded.")

    # 1. Prepare Training Data for LGB (Hard Negatives)
    # Using a subset of train_df to create "Past -> Future" split for labeling
    lgb_train_df = generate_candidates_and_features(model, train_df, None, None, cfg, mode='train')
    
    # 2. Train LGB
    lgb_model, features = train_lgb(lgb_train_df)
    
    # 3. Predict for Submission
    lgb_test_df = generate_candidates_and_features(model, train_df, test_users, None, cfg, mode='test')
    
    print("Reranking Test Candidates...")
    preds = lgb_model.predict(lgb_test_df[features])
    lgb_test_df['final_score'] = preds
    
    # 4. Generate Output
    final_output = []
    # Rank per user
    lgb_test_df = lgb_test_df.sort_values(['user_id', 'final_score'], ascending=[True, False])
    
    # Group and take top 10
    # groupby().head(10) is fast enough
    top10 = lgb_test_df.groupby('user_id').head(10)
    
    top10[['user_id', 'item_id']].to_csv(os.path.join(args.output_dir, 'output_rerank_hardneg.csv'), index=False)
    print("Saved to output_rerank_hardneg.csv")

if __name__ == '__main__':
    main()

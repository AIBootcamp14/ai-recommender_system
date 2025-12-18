"""
Temperature Scaling for SASRec
재학습 없이 예측 확률만 보정하여 성능 향상
"""

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import os
from tqdm import tqdm
from sasrec_proxy import SASRec, Config, apply_proxy_labeling

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--output_dir', type=str, default='../out/')
    parser.add_argument('--model_path', type=str, default='../out/sasrec_proxy_best.pth')
    parser.add_argument('--temperature', type=float, default=1.0, 
                        help='Temperature for softmax scaling (0.5, 0.7, 1.0, 1.5, 2.0)')
    args = parser.parse_args()
    
    print(f"Temperature Scaling with T={args.temperature}")
    
    # Load data
    train_df = pd.read_parquet(os.path.join(args.data_dir, 'train.parquet'))
    if train_df['event_time'].dtype == 'object':
        train_df['event_time'] = pd.to_datetime(train_df['event_time'])
    
    # Apply proxy labeling
    train_df = apply_proxy_labeling(train_df, min_view_count=3)
    
    # Build vocab
    items = train_df['item_id'].unique()
    item_to_idx = {item: i+1 for i, item in enumerate(items)}
    idx_to_item = {i+1: item for i, item in enumerate(items)}
    num_items = len(items)
    
    # Load model
    cfg = Config()
    model = SASRec(num_items, cfg).to(cfg.DEVICE)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    print("Model loaded")
    
    # Prepare sequences
    user_seqs = train_df.groupby('user_id')['item_id'].apply(list).to_dict()
    
    sample_sub = pd.read_csv(os.path.join(args.data_dir, 'sample_submission.csv'))
    test_users = sample_sub['user_id'].unique()
    
    pop_items = train_df['item_id'].value_counts().head(10).index.tolist()
    
    final_recs = []
    
    print("Generating predictions with temperature scaling...")
    for u in tqdm(test_users):
        seq = user_seqs.get(u, [])
        
        if len(seq) == 0:
            recs = pop_items
        else:
            # Prepare input
            if len(seq) > cfg.MAX_LEN:
                seq_in = seq[-cfg.MAX_LEN:]
            else:
                seq_in = [0] * (cfg.MAX_LEN - len(seq)) + seq
            
            seq_idx = [item_to_idx.get(item, 0) for item in seq_in]
            seq_tensor = torch.LongTensor([seq_idx]).to(cfg.DEVICE)
            
            with torch.no_grad():
                logits = model.predict(seq_tensor)  # (1, num_items)
                
                # Temperature Scaling
                scaled_logits = logits / args.temperature
                probs = F.softmax(scaled_logits, dim=1)
                
                # Top 10
                _, top_indices = torch.topk(probs, 10, dim=1)
                
            recs = [idx_to_item.get(idx.item(), pop_items[0]) for idx in top_indices[0]]
        
        for item in recs:
            final_recs.append({'user_id': u, 'item_id': item})
    
    # Save
    out_df = pd.DataFrame(final_recs)
    out_path = os.path.join(args.output_dir, f'output_sasrec_temp_{args.temperature}.csv')
    out_df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

if __name__ == '__main__':
    main()

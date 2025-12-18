"""
Hybrid Reranker: SASRec + Co-view
1. Loads pre-trained SASRec model and Co-view matrix.
2. Generates Top-50 candidates per user.
3. Boosts candidate scores if they are frequently co-viewed with user's LAST ITEM.
4. Ranks and outputs Top-10.
"""

import pandas as pd
import numpy as np
import pickle
import torch
import argparse
import os
from tqdm import tqdm
from sasrec_proxy import SASRec, Config, apply_proxy_labeling

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--output_dir', type=str, default='../out/')
    parser.add_argument('--model_path', type=str, default='../out/sasrec_proxy_best.pth')
    parser.add_argument('--coview_path', type=str, default='../out/coview_matrix.pkl')
    parser.add_argument('--alpha', type=float, default=0.7, help='Weight for SASRec Score')
    parser.add_argument('--beta', type=float, default=0.3, help='Weight for Co-view Score')
    args = parser.parse_args()
    
    # 1. Load Data & Assets
    print("Loading Co-view Matrix...")
    with open(args.coview_path, 'rb') as f:
        coview_matrix = pickle.load(f)
        
    print("Loading Train Data...")
    train_df = pd.read_parquet(os.path.join(args.data_dir, 'train.parquet'))
    if train_df['event_time'].dtype == 'object':
         train_df['event_time'] = pd.to_datetime(train_df['event_time'])
         
    # Apply Proxy Logic to match Vocab
    train_df = apply_proxy_labeling(train_df, min_view_count=3)
    
    items = train_df['item_id'].unique()
    item_to_idx = {item: i+1 for i, item in enumerate(items)}
    idx_to_item = {i+1: item for i, item in enumerate(items)}
    num_items = len(items)
    print(f"Vocab Size: {num_items}")

    cfg = Config()
    model = SASRec(num_items, cfg).to(cfg.DEVICE)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    print("SASRec Model Loaded.")
    
    # 2. Prepare User History
    user_grp = train_df.groupby('user_id')['item_id'].apply(list).to_dict()
    sample_sub = pd.read_csv(os.path.join(args.data_dir, 'sample_submission.csv'))
    test_users = sample_sub['user_id'].unique()
    
    # Global Pop for Fallback
    pop_items = train_df['item_id'].value_counts().head(50).index.tolist()
    
    final_recs = []
    
    # 3. Batch Inference + Reranking
    BATCH_SIZE = 256
    user_batch = []
    seq_batch = []
    
    print("Generating & Reranking...")
    for u in tqdm(test_users):
        seq = user_grp.get(u, [])
        last_item = seq[-1] if seq else None
        
        # Prepare Seq
        if len(seq) > cfg.MAX_LEN:
            seq_in = seq[-cfg.MAX_LEN:]
        else:
            seq_in = [0] * (cfg.MAX_LEN - len(seq)) + seq
        
        seq_idx = [item_to_idx.get(item, 0) for item in seq_in]
        
        user_batch.append((u, last_item))
        seq_batch.append(seq_idx)
        
        if len(user_batch) == BATCH_SIZE or u == test_users[-1]:
            # Predict Top-50
            batch_tensor = torch.LongTensor(seq_batch).to(cfg.DEVICE)
            with torch.no_grad():
                logits = model.predict(batch_tensor)
                scores_topk, indices_topk = torch.topk(logits, 50, dim=1)
                
                # Normalize SASRec Scores (0~1) per user roughly?
                # Logits can be large. Softmax is better for prob.
                # probs = torch.softmax(logits, dim=1) # Too heavy for all items
                # Let's softmax only top-k? Or just MinMax on TopK?
                # MinMax on TopK is safer.
                
                scores_np = scores_topk.cpu().numpy()
                indices_np = indices_topk.cpu().numpy()
            
            # Rerank Loop
            for k, (curr_u, curr_last) in enumerate(user_batch):
                cands = indices_np[k]
                base_scores = scores_np[k]
                
                # Min-max scaling for base scores
                s_min, s_max = base_scores.min(), base_scores.max()
                if s_max > s_min:
                    norm_base = (base_scores - s_min) / (s_max - s_min)
                else:
                    norm_base = np.ones_like(base_scores)
                
                reranked_candidates = []
                
                for idx_in_topk, item_idx_model in enumerate(cands):
                    cand_item = idx_to_item.get(item_idx_model)
                    if cand_item is None: continue
                    
                    # 1. Base Score (SASRec)
                    score_sas = norm_base[idx_in_topk]
                    
                    # 2. Co-view Score
                    score_co = 0.0
                    if curr_last is not None and curr_last in coview_matrix:
                        # Check global dict
                         score_co = coview_matrix[curr_last].get(cand_item, 0.0)
                    
                    # 3. Fusion
                    # Co-view scores (Jaccard) are usually small (<0.1). Amplify or Normalize?
                    # Let's simple boost: alpha * SAS + beta * (CoView * 10?)
                    # Experiment: Just weighted sum.
                    final_score = (args.alpha * score_sas) + (args.beta * score_co * 5.0) 
                    # *5.0 is heuristic to match scale of 0~1 vs 0~0.2
                    
                    reranked_candidates.append((cand_item, final_score))
                
                # Sort & Pick Top 10
                reranked_candidates.sort(key=lambda x: -x[1])
                top10 = [x[0] for x in reranked_candidates[:10]]
                
                # Fallback if less than 10
                if len(top10) < 10:
                    for p in pop_items:
                        if p not in top10:
                            top10.append(p)
                        if len(top10) == 10: break
                        
                for item in top10:
                    final_recs.append({'user_id': curr_u, 'item_id': item})

            user_batch = []
            seq_batch = []

    # Output
    out_df = pd.DataFrame(final_recs)
    out_path = os.path.join(args.output_dir, 'output_rerank_coview.csv')
    out_df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

if __name__ == '__main__':
    main()

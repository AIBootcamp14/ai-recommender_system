"""
Weighted Softmax Ensemble (Score-based Fusion)
Merges multiple submission/score files using Score Normalization.
Better than RRF when one model (SASRec) is significantly stronger but needs ALS's global view.

Requires:
- Input CSVs should ideally have scores. If only ranks are present, we convert Rank -> Score (1/rank).
- We assume standard submission format: user_id, item_id.

Strategy:
1. Load submissions.
2. Recover/Assign Scores:
   - If score not available, Score = 1 / (Rank + 1)
3. Normalize Scores per User:
   - Min-Max Scaling per user per model to make ranges comparable.
4. Weighted Sum:
   - Final_Score = w1 * Norm_Score1 + w2 * Norm_Score2
"""

import pandas as pd
import numpy as np
import argparse
import os
from collections import defaultdict
from tqdm import tqdm

def load_submission(file_path):
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)
    if 'session_id' in df.columns:
        df = df.rename(columns={'session_id': 'user_id'})
    return df

def score_fusion(files, weights, top_k=10):
    user_item_scores = defaultdict(lambda: defaultdict(float))
    
    for idx, fpath in enumerate(files):
        w = weights[idx]
        df = load_submission(fpath)
        print(f"Processing Model {idx+1} (w={w})...")
        
        # Check if 'score' column exists, else Infer from Rank
        has_score = 'score' in df.columns
        
        # Group by user to normalize
        # Assuming df is sorted by rank if no score
        current_user = None
        user_scores = []
        rank = 0
        
        # Function to process a user's batch
        def process_batch(u, items, scores):
            if not items: return
            
            # Normalize Scores (Min-Max) to 0.0 ~ 1.0 range
            if len(scores) > 1:
                min_s, max_s = min(scores), max(scores)
                if max_s > min_s:
                    scores = [(s - min_s) / (max_s - min_s) for s in scores]
                else:
                    scores = [1.0] * len(scores) # All same
            else:
                scores = [1.0]
                
            for item, s in zip(items, scores):
                user_item_scores[u][item] += s * w

        # Iterator
        batch_u = None
        batch_items = []
        batch_scores = []
        
        for row in tqdm(df.itertuples(), total=len(df)):
            u = getattr(row, 'user_id')
            item = getattr(row, 'item_id')
            
            if has_score:
                s = getattr(row, 'score')
            else:
                # Implicit Rank Score: 1 / log2(Rank + 2) -> discount slower
                # Rank resets per user. Simplest: Just append based on order.
                # Since we normalize later, exact value matters less than relative order.
                if u != batch_u:
                    rank = 1
                else:
                    rank += 1
                s = 1.0 / rank # Simple reciprocal
            
            if u != batch_u:
                if batch_u is not None:
                    process_batch(batch_u, batch_items, batch_scores)
                batch_u = u
                batch_items = []
                batch_scores = []
                rank = 1
            
            batch_items.append(item)
            batch_scores.append(s)
            
        # Last batch
        if batch_u:
            process_batch(batch_u, batch_items, batch_scores)
            
    return user_item_scores

def generate_output(scores_dict, output_path, top_k=10):
    print("\nSorting and Generating Final CSV...")
    results = []
    
    for u, item_map in tqdm(scores_dict.items()):
        # Sort by final weighted score
        sorted_items = sorted(item_map.items(), key=lambda x: -x[1])[:top_k]
        
        for item, score in sorted_items:
            results.append({'user_id': u, 'item_id': item})
            
    sub_df = pd.DataFrame(results)
    sub_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', nargs='+', required=True)
    parser.add_argument('--weights', nargs='+', type=float, required=True)
    parser.add_argument('--output', type=str, default='../out/output_score_ensemble.csv')
    args = parser.parse_args()
    
    if len(args.files) != len(args.weights):
        raise ValueError("Count mismatch files vs weights")
        
    scores = score_fusion(args.files, args.weights)
    generate_output(scores, args.output)

if __name__ == '__main__':
    main()

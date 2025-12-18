"""
Ensemble Grid Search
여러 가중치 조합을 빠르게 생성
"""

import pandas as pd
import argparse
import os
from tqdm import tqdm
from collections import defaultdict

def ensemble_with_weights(files, weights, output_path):
    """
    파일들을 가중치로 앙상블
    """
    print(f"Ensemble with weights: {weights}")
    
    # Load all files
    dfs = [pd.read_csv(f) for f in files]
    
    # Score aggregation
    final_scores = defaultdict(float)
    
    for i, (df, weight) in enumerate(zip(dfs, weights)):
        print(f"Processing {files[i]} (weight={weight})...")
        
        current_user = None
        rank = 1
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Model {i+1}"):
            u, item = row['user_id'], row['item_id']
            
            if u != current_user:
                current_user = u
                rank = 1
            else:
                rank += 1
            
            # RRF-like scoring
            score = weight * (1.0 / (60 + rank))
            final_scores[(u, item)] += score
    
    # Sort and generate final output
    print("Sorting and generating final CSV...")
    sorted_scores = sorted(final_scores.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))
    
    # Group by users and take top 10
    user_recs = defaultdict(list)
    for (user_id, item_id), score in sorted_scores:
        if len(user_recs[user_id]) < 10:
            user_recs[user_id].append(item_id)
    
    # Flatten
    final_data = []
    for user_id in sorted(user_recs.keys()):
        for item_id in user_recs[user_id]:
            final_data.append({'user_id': user_id, 'item_id': item_id})
    
    # Save
    out_df = pd.DataFrame(final_data)
    out_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='../out/')
    args = parser.parse_args()
    
    # Files
    sasrec_file = os.path.join(args.output_dir, 'output_sasrec_proxy.csv')
    als_file = os.path.join(args.output_dir, 'output_als_decay_0.1.csv')
    
    # Grid search configurations
    configs = [
        (0.65, 0.35, 'output_ensemble_65_35.csv'),
        (0.55, 0.45, 'output_ensemble_55_45.csv'),
        (0.62, 0.38, 'output_ensemble_62_38.csv'),
    ]
    
    print("="*60)
    print("Ensemble Grid Search")
    print("="*60)
    
    for sasrec_w, als_w, output_name in configs:
        output_path = os.path.join(args.output_dir, output_name)
        ensemble_with_weights(
            [sasrec_file, als_file],
            [sasrec_w, als_w],
            output_path
        )
        print()

if __name__ == '__main__':
    main()

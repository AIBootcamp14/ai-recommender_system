"""
Build Co-view Matrix
Calculates item-item co-occurrence probabilities within user sessions.
Uses PMI (Pointwise Mutual Information) or simple normalized co-counts to capture relationships.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import argparse
import os
import pickle
from tqdm import tqdm
import scipy.sparse as sp

def build_coview_matrix(data_dir, output_path, min_cooccur=3):
    print("Loading data...")
    # Load all interactions
    # Ideally should use full train data
    df = pd.read_parquet(os.path.join(data_dir, 'train.parquet'))
    
    # Filter for meaningful sessions? 
    # For co-view, we want "Item A and Item B in same session".
    # User session ID is needed. If not present, use user_id + time gap logic.
    # Assuming 'user_session' column exists based on previous file knowledge.
    # If not, we resort to user_id grouping.
    
    if 'user_session' in df.columns:
        group_col = 'user_session'
    else:
        group_col = 'user_id' # Fallback
        
    print(f"Grouping by {group_col}...")
    
    # Only view/cart/purchase matters? Yes.
    # Let's map items to IDs first to save memory
    items = df['item_id'].unique()
    item_to_idx = {item: i for i, item in enumerate(items)}
    idx_to_item = {i: item for i, item in enumerate(items)}
    n_items = len(items)
    
    # Count item occurrences (Global popularity for normalization)
    item_counts = defaultdict(int)
    
    # Co-occurrence pairs
    # Use sparse matrix: rows=item_i, cols=item_j, val=count
    # To save memory, we can iterate sessions
    
    sessions = df.groupby(group_col)['item_id'].apply(list)
    
    co_counts = defaultdict(int)
    
    print("Counting co-occurrences...")
    for seq in tqdm(sessions):
        unique_items = list(set(seq)) # Don't count self-loops multiple times per session?
        # Actually frequency matters. But simple set is safer for now.
        
        # Map to indices
        indices = [item_to_idx[item] for item in unique_items if item in item_to_idx]
        
        for i in indices:
            item_counts[i] += 1
            for j in indices:
                if i == j: continue
                # Undirected graph: sort to force A-B same as B-A?
                # Or Directed: A then B? 
                # Co-view is usually symmetric context.
                pair = tuple(sorted((i, j)))
                co_counts[pair] += 1

    print("Building Matrix...")
    # Convert to sparse matrix or just dict lookup?
    # Dictionary is easier for lookup later.
    # Normalize: Jaccard = Intersection / Union
    # Union(A, B) = Count(A) + Count(B) - Intersection(A, B)
    
    norm_co_matrix = {}
    filtered_pairs = 0
    total_pairs = len(co_counts)
    
    for (i, j), count in tqdm(co_counts.items()):
        # Filter: Only keep pairs with co-occurrence >= min_cooccur
        if count < min_cooccur:
            filtered_pairs += 1
            continue
            
        cnt_i = item_counts[i]
        cnt_j = item_counts[j]
        union = cnt_i + cnt_j - count
        
        if union > 0:
            score = count / union # Jaccard
        else:
            score = 0
            
        # Store both directions for easier lookup
        item_i = idx_to_item[i]
        item_j = idx_to_item[j]
        
        if item_i not in norm_co_matrix: norm_co_matrix[item_i] = {}
        if item_j not in norm_co_matrix: norm_co_matrix[item_j] = {}
        
        norm_co_matrix[item_i][item_j] = score
        norm_co_matrix[item_j][item_i] = score

    print(f"Filtered {filtered_pairs}/{total_pairs} pairs (below threshold {min_cooccur})")
    print(f"Saving co-view matrix with {len(norm_co_matrix)} items...")
    with open(output_path, 'wb') as f:
        pickle.dump(norm_co_matrix, f)
        
    print(f"Saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--output', type=str, default='../out/coview_matrix.pkl')
    parser.add_argument('--min_cooccur', type=int, default=3, help='Minimum co-occurrence count to include')
    args = parser.parse_args()
    
    build_coview_matrix(args.data_dir, args.output, args.min_cooccur)

if __name__ == '__main__':
    main()

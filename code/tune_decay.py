"""
Time Decay Rate Tuning script
"""
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
import argparse
import os
import time

def evaluate_decay_rate(train_df, test_df, decay_rate, als_params, user_to_idx, item_to_idx, idx_to_item, user_interaction_count, global_popular_items, item_popularity, popularity_boost):
    print(f"\nEvaluating Decay Rate: {decay_rate}")
    
    # --- Weighting ---
    df = train_df.copy()
    max_date = df['event_time'].max()
    df['days_elapsed'] = (max_date - df['event_time']).dt.total_seconds() / (24 * 3600)
    df['time_weight'] = np.exp(-decay_rate * df['days_elapsed'])
    
    weight_map = {'view': 1, 'cart': 10, 'purchase': 20}
    df['type_weight'] = df['event_type'].map(weight_map).fillna(1)
    df['weight'] = df['type_weight'] * df['time_weight']
    
    df['user_idx'] = df['user_id'].map(user_to_idx)
    df['item_idx'] = df['item_id'].map(item_to_idx)
    
    # Matrix
    user_item_counts = df.groupby(['user_idx', 'item_idx'])['weight'].sum().reset_index(name='count')
    interaction_matrix = csr_matrix(
        (user_item_counts['count'].values,
         (user_item_counts['user_idx'].values, user_item_counts['item_idx'].values)),
        shape=(len(user_to_idx), len(item_to_idx))
    )
    
    # Model
    model = AlternatingLeastSquares(**als_params)
    model.fit(interaction_matrix)
    
    # 간단한 검증을 위해 몇 명만 샘플링하거나 전체를 하거나. 
    # 여기서는 시간상 전체 생성을 해서 제출 파일 포맷으로 만듦 (이후 제출해서 점수 확인)
    # 다만 코드로 검증할 ground truth가 없으므로 파일 생성까지만 수행.
    # 사용자가 직접 제출해서 리더보드 점수를 봐야 함.
    
    return model, interaction_matrix

def recommend(user_id, model, interaction_matrix, user_to_idx, idx_to_item, user_interaction_count, global_popular_items, item_popularity, popularity_boost, top_k=10):
    if user_id not in user_to_idx:
        return global_popular_items[:top_k]
    
    user_idx = user_to_idx[user_id]
    user_activity = user_interaction_count.get(user_id, 0)
    
    try:
        item_ids, als_scores = model.recommend(
            user_idx, interaction_matrix[user_idx], N=100, filter_already_liked_items=False
        )
    except:
        return global_popular_items[:top_k]

    if als_scores.max() > als_scores.min():
        als_scores_norm = (als_scores - als_scores.min()) / (als_scores.max() - als_scores.min())
    else:
        als_scores_norm = np.ones_like(als_scores)

    if user_activity >= 50: pop_weight = 0.0
    elif user_activity >= 10: pop_weight = popularity_boost
    else: pop_weight = popularity_boost * 2

    final_scores = []
    for i, (item_idx, als_score) in enumerate(zip(item_ids, als_scores_norm)):
        item_id = idx_to_item[item_idx]
        pop_score = item_popularity.get(item_id, 0)
        final = (1 - pop_weight) * als_score + pop_weight * pop_score
        final_scores.append((item_id, final))

    final_scores.sort(key=lambda x: -x[1])
    return [item_id for item_id, _ in final_scores[:top_k]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--output_dir', type=str, default='../out/')
    args = parser.parse_args()
    
    # Settings
    decay_rates = [0.1, 0.2, 0.5] # 0.05까지 해봄 -> 더 과감하게
    als_params = {'factors': 64, 'regularization': 0.01, 'alpha': 10, 'iterations': 20, 'random_state': 42}
    popularity_boost = 0.1
    
    # Load Data
    print("Loading Data...")
    train_df = pd.read_parquet(os.path.join(args.data_dir, 'train.parquet'))
    if train_df['event_time'].dtype == 'object':
        train_df['event_time'] = pd.to_datetime(train_df['event_time'])
    
    sample_sub = pd.read_csv(os.path.join(args.data_dir, 'sample_submission.csv'))
    test_users = sample_sub['user_id'].unique()
    
    # Pre-compute common resources
    users = train_df['user_id'].unique()
    items = train_df['item_id'].unique()
    user_to_idx = {u: i for i, u in enumerate(users)}
    idx_to_user = {i: u for u, i in user_to_idx.items()}
    item_to_idx = {item: i for i, item in enumerate(items)}
    idx_to_item = {i: item for item, i in item_to_idx.items()}
    
    user_counts = train_df.groupby('user_id').size()
    user_interaction_count = user_counts.to_dict()
    
    purchase_counts = train_df[train_df['event_type'] == 'purchase']['item_id'].value_counts()
    global_popular_items = purchase_counts.head(100).index.tolist()
    
    item_counts = train_df['item_id'].value_counts()
    item_popularity = (item_counts / item_counts.max()).to_dict()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for rate in decay_rates:
        print(f"\n>>> Processing Decay Rate: {rate} <<<")
        model, matrix = evaluate_decay_rate(
            train_df, pd.DataFrame({'user_id': test_users}), rate, als_params,
            user_to_idx, item_to_idx, idx_to_item,
            user_interaction_count, global_popular_items, item_popularity, popularity_boost
        )
        
        # Submission Gen
        results = []
        # 시간 단축을 위해 batch 추천 등 최적화 가능하지만, 일단 loop
        print(f"Generating csv for rate {rate}...")
        for user_id in test_users: # tqdm 생략 or simple print
            recs = recommend(user_id, model, matrix, user_to_idx, idx_to_item, 
                             user_interaction_count, global_popular_items, item_popularity, popularity_boost)
            for item_id in recs:
                results.append({'user_id': user_id, 'item_id': item_id})
        
        sub_df = pd.DataFrame(results)
        filename = f"output_decay_{rate}.csv"
        sub_df.to_csv(os.path.join(args.output_dir, filename), index=False)
        print(f"Saved {filename}")

if __name__ == '__main__':
    main()

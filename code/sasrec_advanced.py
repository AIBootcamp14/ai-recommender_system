"""
SASRec Advanced: High Priority Improvements
1. Epoch 증가 (20 -> 50)
2. Multi-Task Learning (Item + Category Prediction)
3. Session Intensity Proxy Labeling
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import time
from tqdm import tqdm

class Config:
    MAX_LEN = 50
    HIDDEN_UNITS = 64
    NUM_HEADS = 2
    NUM_BLOCKS = 2
    DROPOUT_RATE = 0.2
    LR = 0.001
    BATCH_SIZE = 128
    NUM_EPOCHS = 50  # Increased from 20
    DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

class SASRecMultiTask(nn.Module):
    """SASRec with Multi-Task Learning (Item + Category)"""
    def __init__(self, num_items, num_categories, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Shared Embeddings
        self.item_emb = nn.Embedding(num_items + 1, cfg.HIDDEN_UNITS, padding_idx=0)
        self.pos_emb = nn.Embedding(cfg.MAX_LEN, cfg.HIDDEN_UNITS)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.HIDDEN_UNITS,
            nhead=cfg.NUM_HEADS,
            dim_feedforward=cfg.HIDDEN_UNITS * 4,
            dropout=cfg.DROPOUT_RATE,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.NUM_BLOCKS)
        
        # Multi-Task Heads
        self.item_head = nn.Linear(cfg.HIDDEN_UNITS, num_items + 1)  # Main Task
        self.category_head = nn.Linear(cfg.HIDDEN_UNITS, num_categories + 1)  # Auxiliary Task
        
    def forward(self, seq, task='item'):
        # seq: (B, L)
        positions = torch.arange(seq.size(1), device=seq.device).unsqueeze(0)
        
        # Embeddings
        seq_emb = self.item_emb(seq) + self.pos_emb(positions)
        
        # Mask (padding)
        mask = (seq == 0)
        
        # Encode
        hidden = self.encoder(seq_emb, src_key_padding_mask=mask)
        
        # Task-specific output
        if task == 'item':
            return self.item_head(hidden)  # (B, L, num_items)
        elif task == 'category':
            return self.category_head(hidden)  # (B, L, num_categories)
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def predict(self, seq):
        return self.forward(seq, task='item')[:, -1, :]  # Last position

class RecSysDatasetMultiTask(Dataset):
    def __init__(self, sequences, categories, max_len):
        """
        sequences: List of item sequences
        categories: List of category sequences (aligned with items)
        """
        self.sequences = sequences
        self.categories = categories
        self.max_len = max_len
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        cat = self.categories[idx]
        
        # Prepare input/target for next-item prediction
        if len(seq) > self.max_len:
            input_seq = seq[-(self.max_len+1):-1]
            target_item = seq[-1]
            
            input_cat = cat[-(self.max_len+1):-1]
            target_cat = cat[-1]
        else:
            input_seq = seq[:-1]
            target_item = seq[-1]
            
            input_cat = cat[:-1]
            target_cat = cat[-1]
        
        # Padding
        if len(input_seq) < self.max_len:
            pad_len = self.max_len - len(input_seq)
            input_seq = [0] * pad_len + input_seq
            input_cat = [0] * pad_len + input_cat
            
        return (torch.LongTensor(input_seq), 
                target_item,  # Scalar int
                torch.LongTensor(input_cat), 
                target_cat)  # Scalar int

def apply_proxy_labeling_intensity(df, min_view_count=3):
    """
    Session Intensity Proxy Labeling:
    - Purchase/Cart: Always included
    - View: Include if total count >= min_view OR session revisit >= 2
    """
    print(f"Applying Session Intensity Proxy Labeling...")
    original_len = len(df)
    
    # Strong signals (always keep)
    strong_signals = df[df['event_type'].isin(['purchase', 'cart'])].copy()
    
    # Views: Apply intensity logic
    views = df[df['event_type'] == 'view'].copy()
    
    # Strategy 1: Global view count
    view_counts_global = views.groupby(['user_id', 'item_id']).size().reset_index(name='global_count')
    
    # Strategy 2: Session-level revisits
    view_counts_session = views.groupby(['user_session', 'item_id']).size().reset_index(name='session_count')
    
    # Merge both
    views_with_counts = views.merge(view_counts_global, on=['user_id', 'item_id'], how='left')
    views_with_counts = views_with_counts.merge(view_counts_session, on=['user_session', 'item_id'], how='left')
    
    # Filter: Keep if (global >= min_view) OR (session revisit >= 2)
    frequent_views = views_with_counts[
        (views_with_counts['global_count'] >= min_view_count) | 
        (views_with_counts['session_count'] >= 2)
    ].copy()
    
    # Weight assignment (for future use, though current model doesn't use weights directly)
    frequent_views['proxy_weight'] = frequent_views['session_count'] * frequent_views['global_count']
    
    # Combine
    proxy_df = pd.concat([strong_signals, frequent_views]).drop_duplicates(subset=['user_id', 'item_id', 'event_time'])
    proxy_df = proxy_df.sort_values(['user_id', 'event_time'])
    
    print(f"  Original: {original_len:,}")
    print(f"  Filtered: {len(proxy_df):,} ({len(proxy_df)/original_len:.1%})")
    
    return proxy_df

def train_multitask(model, dataloader, optimizer, item_criterion, cat_criterion, device, alpha=0.7):
    """
    Multi-Task Training
    alpha: weight for item loss (1-alpha for category loss)
    """
    model.train()
    total_loss = 0
    count = 0
    
    for seqs, target_items, cat_seqs, target_cats in tqdm(dataloader, desc="Training"):
        seqs = seqs.to(device)
        target_items = target_items.to(device)
        cat_seqs = cat_seqs.to(device)
        target_cats = target_cats.to(device)
        
        optimizer.zero_grad()
        
        # Item prediction
        item_logits = model(seqs, task='item')[:, -1, :]  # Last position
        item_loss = item_criterion(item_logits, target_items)
        
        # Category prediction
        cat_logits = model(cat_seqs, task='category')[:, -1, :]
        cat_loss = cat_criterion(cat_logits, target_cats)
        
        # Combined loss
        loss = alpha * item_loss + (1 - alpha) * cat_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        count += 1
        
    return total_loss / count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--output_dir', type=str, default='../out/')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--min_view', type=int, default=3)
    args = parser.parse_args()
    
    cfg = Config()
    cfg.NUM_EPOCHS = args.epochs
    
    print("="*60)
    print("SASRec Advanced: Multi-Task + Session Intensity")
    print("="*60)
    print("Loading data...")
    
    train_df = pd.read_parquet(os.path.join(args.data_dir, 'train.parquet'))
    if train_df['event_time'].dtype == 'object':
        train_df['event_time'] = pd.to_datetime(train_df['event_time'])
    
    # Apply Session Intensity Proxy
    train_df = apply_proxy_labeling_intensity(train_df, min_view_count=args.min_view)
    
    # Build vocabularies
    items = train_df['item_id'].unique()
    item_to_idx = {item: i+1 for i, item in enumerate(items)}
    idx_to_item = {i+1: item for i, item in enumerate(items)}
    num_items = len(items)
    
    # Categories
    train_df['category_code'] = train_df['category_code'].fillna('unknown')
    categories = train_df['category_code'].unique()
    cat_to_idx = {cat: i+1 for i, cat in enumerate(categories)}
    idx_to_cat = {i+1: cat for i, cat in enumerate(categories)}
    num_categories = len(categories)
    
    print(f"Num Items: {num_items}, Num Categories: {num_categories}")
    
    # Build sequences
    user_groups = train_df.groupby('user_id')
    
    item_seqs = []
    cat_seqs = []
    
    for user_id, group in tqdm(user_groups, desc="Building sequences"):
        group = group.sort_values('event_time')
        
        item_seq = [item_to_idx.get(item, 0) for item in group['item_id'].tolist()]
        cat_seq = [cat_to_idx.get(cat, 0) for cat in group['category_code'].tolist()]
        
        if len(item_seq) >= 2:  # Need at least 2 for input->target
            item_seqs.append(item_seq)
            cat_seqs.append(cat_seq)
    
    print(f"Training sequences: {len(item_seqs)}")
    
    # Dataset & Dataloader
    dataset = RecSysDatasetMultiTask(item_seqs, cat_seqs, cfg.MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
    
    # Model
    model = SASRecMultiTask(num_items, num_categories, cfg).to(cfg.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    item_criterion = nn.CrossEntropyLoss(ignore_index=0)
    cat_criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Training
    best_loss = float('inf')
    model_save_path = os.path.join(args.output_dir, 'sasrec_advanced_best.pth')
    
    print("\nTraining...")
    for epoch in range(cfg.NUM_EPOCHS):
        start = time.time()
        avg_loss = train_multitask(model, dataloader, optimizer, item_criterion, cat_criterion, cfg.DEVICE)
        elapsed = time.time() - start
        
        print(f"Epoch {epoch+1}/{cfg.NUM_EPOCHS} | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_save_path)
    
    print(f"\nBest model saved to {model_save_path}")
    
    # Load best & Inference
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    
    print("\nGenerating Submission...")
    sample_sub = pd.read_csv(os.path.join(args.data_dir, 'sample_submission.csv'))
    test_users = sample_sub['user_id'].unique()
    
    user_seqs_map = dict(zip(user_groups.groups.keys(), 
                             [[item_to_idx.get(item, 0) for item in g['item_id'].tolist()] 
                              for _, g in user_groups]))
    
    pop_items = train_df['item_id'].value_counts().head(10).index.tolist()
    
    final_recs = []
    
    for u in tqdm(test_users):
        seq = user_seqs_map.get(u, [])
        
        if len(seq) > cfg.MAX_LEN:
            seq_in = seq[-cfg.MAX_LEN:]
        else:
            seq_in = [0] * (cfg.MAX_LEN - len(seq)) + seq
        
        if len(seq) == 0:
            recs = pop_items
        else:
            seq_tensor = torch.LongTensor([seq_in]).to(cfg.DEVICE)
            with torch.no_grad():
                logits = model.predict(seq_tensor)
                _, top_indices = torch.topk(logits, 10, dim=1)
                
            recs = [idx_to_item.get(idx.item(), pop_items[0]) for idx in top_indices[0]]
        
        for item in recs:
            final_recs.append({'user_id': u, 'item_id': item})
    
    out_df = pd.DataFrame(final_recs)
    out_path = os.path.join(args.output_dir, 'output_sasrec_advanced.csv')
    out_df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

if __name__ == '__main__':
    main()

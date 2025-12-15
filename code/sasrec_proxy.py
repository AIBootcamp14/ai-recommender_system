"""
SASRec with Proxy Labeling (Target = Purchase + Cart + Frequent Views)
- Filters out "meaningless views" to focus on strong signals.
- Solves the 0.02% Sparsity problem by augmenting positive labels.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time

# --- Config & Hyperparameters ---
class Config:
    MAX_LEN = 50
    HIDDEN_UNITS = 64
    NUM_BLOCKS = 2
    NUM_HEADS = 2
    DROPOUT_RATE = 0.2
    LR = 0.001
    BATCH_SIZE = 128
    NUM_EPOCHS = 10
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# --- Dataset ---
class RecSysDataset(Dataset):
    def __init__(self, user_seqs, num_items, max_len=50):
        self.user_seqs = user_seqs
        self.num_items = num_items
        self.max_len = max_len

    def __len__(self):
        return len(self.user_seqs)

    def __getitem__(self, index):
        seq = self.user_seqs[index]
        if len(seq) > self.max_len:
            seq = seq[-self.max_len:]
        else:
            seq = [0] * (self.max_len - len(seq)) + seq
        return torch.LongTensor(seq)

# --- Model ---
class SASRec(nn.Module):
    def __init__(self, num_items, cfg):
        super(SASRec, self).__init__()
        self.num_items = num_items
        self.cfg = cfg
        self.item_emb = nn.Embedding(num_items + 1, cfg.HIDDEN_UNITS, padding_idx=0)
        self.pos_emb = nn.Embedding(cfg.MAX_LEN, cfg.HIDDEN_UNITS)
        self.emb_dropout = nn.Dropout(cfg.DROPOUT_RATE)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.HIDDEN_UNITS,
            nhead=cfg.NUM_HEADS,
            dim_feedforward=cfg.HIDDEN_UNITS * 4,
            dropout=cfg.DROPOUT_RATE,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.NUM_BLOCKS)
        self.ln = nn.LayerNorm(cfg.HIDDEN_UNITS)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            stdv = np.sqrt(1. / self.num_items)
            module.weight.data.uniform_(-stdv, stdv)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, item_seq):
        device = item_seq.device
        seq_len = item_seq.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)
        ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(item_seq.size(0), -1)
        padding_mask = (item_seq == 0)
        
        emb = self.item_emb(item_seq) + self.pos_emb(ids)
        emb = self.emb_dropout(emb)
        output = self.transformer_encoder(emb, mask=mask, src_key_padding_mask=padding_mask)
        output = self.ln(output)
        return output

    def predict(self, item_seq):
        output = self.forward(item_seq)
        last_output = output[:, -1, :]
        logits = torch.matmul(last_output, self.item_emb.weight.t())
        logits[:, 0] = -1e9
        return logits

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    count = 0
    for seq in dataloader:
        seq = seq.to(device)
        logits = model(seq)
        logits = torch.matmul(logits, model.item_emb.weight.t())
        
        preds = logits[:, :-1, :].reshape(-1, logits.size(-1))
        targets = seq[:, 1:].reshape(-1)
        
        loss = criterion(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        count += 1
    return total_loss / count



def apply_proxy_labeling(df, min_view_count=3):
    """
    Applies Proxy Labeling Logic:
    1. Keep ALL 'purchase' events.
    2. Keep ALL 'cart' events.
    3. Keep 'view' events ONLY IF the user viewed the item >= min_view_count times.
    """
    print(f"Applying Proxy Labeling (Keep Purchase/Cart + Views >= {min_view_count})...")
    original_len = len(df)
    
    # 1. Identify Purchase/Cart
    strong_signals = df[df['event_type'].isin(['purchase', 'cart'])].copy()
    
    # 2. Identify Frequent Views
    views = df[df['event_type'] == 'view']
    view_counts = views.groupby(['user_id', 'item_id']).size().reset_index(name='count')
    frequent_view_pairs = view_counts[view_counts['count'] >= min_view_count][['user_id', 'item_id']]
    
    # Merge back to get the original view events (with timestamps) for these pairs
    # Note: We keep ALL instances of the view if it meets the criteria, to preserve frequency in sequence?
    # Or just keep them? Usually keeping them all preserves "intensity" implicitly.
    frequent_views = pd.merge(views, frequent_view_pairs, on=['user_id', 'item_id'], how='inner')
    
    # 3. Combine
    proxy_df = pd.concat([strong_signals, frequent_views]).drop_duplicates()
    proxy_df = proxy_df.sort_values(['user_id', 'event_time'])
    
    print(f"  Original Interations: {original_len:,}")
    print(f"  Filtered (Proxy):     {len(proxy_df):,} ({len(proxy_df)/original_len:.1%})")
    
    return proxy_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--output_dir', type=str, default='../out/')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--min_view', type=int, default=3, help='Minimum view count to be considered a proxy label')
    args = parser.parse_args()
    
    cfg = Config()
    cfg.NUM_EPOCHS = args.epochs
    cfg.MAX_LEN = args.max_len
    
    print("="*60)
    print("SASRec with Proxy Labeling")
    print("="*60)
    
    # 1. Load Data
    print("Loading data...")
    train_df = pd.read_parquet(os.path.join(args.data_dir, 'train.parquet'))
    if train_df['event_time'].dtype == 'object':
         train_df['event_time'] = pd.to_datetime(train_df['event_time'])
    
    # 2. Apply Proxy Labeling
    train_df = apply_proxy_labeling(train_df, min_view_count=args.min_view)
    
    # 3. Preprocessing
    items = train_df['item_id'].unique()
    item_to_idx = {item: i+1 for i, item in enumerate(items)}
    idx_to_item = {i+1: item for i, item in enumerate(items)}
    num_items = len(items)
    print(f"Num Items (Active): {num_items}")

    print("Creating User Sequences...")
    user_grp = train_df.groupby('user_id')['item_id'].apply(lambda x: [item_to_idx[i] for i in x]).to_dict()
    all_seqs = list(user_grp.values())
    
    dataset = RecSysDataset(all_seqs, num_items, max_len=cfg.MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
    
    model = SASRec(num_items, cfg).to(cfg.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    

    # Training Loop
    best_loss = float('inf')
    model_save_path = os.path.join(args.output_dir, 'sasrec_proxy_best.pth')
    
    print("\nTraining...")
    for epoch in range(cfg.NUM_EPOCHS):
        start_time = time.time()
        avg_loss = train(model, dataloader, optimizer, criterion, cfg.DEVICE)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{cfg.NUM_EPOCHS} | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")
        
        # Save Best Model mostly based on Loss (or implement validation)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_save_path)
            # print(f"  Saved best model to {model_save_path}")

    print(f"\nTraining Complete! Best model saved to {model_save_path}")
    
    # Reload best model for inference
    model.load_state_dict(torch.load(model_save_path))
    print("Loaded best model for inference.")

    print("\nGenerating Submission...")
    sample_sub = pd.read_csv(os.path.join(args.data_dir, 'sample_submission.csv'))
    test_users = sample_sub['user_id'].unique()
    
    model.eval()
    user_recs_map = {}
    
    # Inference Loop
    # NOTE: Using the PROXY sequences for inference context
    
    # Prepare all sequences (some users might not be in proxy dataset due to filtering)
    # If a user is missing from proxy dataset, we can fall back to Popularity (Global Proxy Popularity)
    
    # Global Popularity in Proxy Dataset
    pop_items = train_df['item_id'].value_counts().head(10).index.tolist()
    
    BATCH_SIZE = 256
    test_seqs_tensor = []
    test_user_ids_list = []
    
    # Process known users
    for u in test_users:
        if u in user_grp:
            seq = user_grp[u]
            if len(seq) > cfg.MAX_LEN:
                seq = seq[-cfg.MAX_LEN:]
            else:
                seq = [0] * (cfg.MAX_LEN - len(seq)) + seq
            test_seqs_tensor.append(seq)
            test_user_ids_list.append(u)
        else:
            # Fallback for empty/cold users immediately
            user_recs_map[u] = pop_items
            
    if test_seqs_tensor:
        test_tensor = torch.LongTensor(test_seqs_tensor).to(cfg.DEVICE)
        num_batches = (len(test_tensor) + BATCH_SIZE - 1) // BATCH_SIZE
        
        with torch.no_grad():
            for i in tqdm(range(num_batches), desc="Predicting"):
                batch = test_tensor[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
                logits = model.predict(batch)
                _, top_indices = torch.topk(logits, 10, dim=1)
                top_indices = top_indices.cpu().numpy()
                
                batch_users = test_user_ids_list[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
                for u_idx, u_id in enumerate(batch_users):
                    recs = [idx_to_item.get(idx, pop_items[0]) for idx in top_indices[u_idx]]
                    user_recs_map[u_id] = recs

    # Final result
    final_data = []
    for u in test_users:
        recs = user_recs_map.get(u, pop_items)
        for item in recs:
            final_data.append({'user_id': u, 'item_id': item})
            
    sub_df = pd.DataFrame(final_data)
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'output_sasrec_proxy.csv')
    sub_df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

if __name__ == '__main__':
    main()

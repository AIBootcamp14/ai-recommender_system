"""
SASRec (Self-Attentive Sequential Recommendation) Implementation
- Uses PyTorch
- Transforms data into sequences per user
- Supports Training and Evaluation
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
    MAX_LEN = 50        # 최대 시퀀스 길이
    HIDDEN_UNITS = 64   # 임베딩 차원 (SASRec은 보통 50~100 사이 사용)
    NUM_BLOCKS = 2      # Transformer Block 개수
    NUM_HEADS = 2       # Attention Head 개수
    DROPOUT_RATE = 0.2
    LR = 0.001
    BATCH_SIZE = 128
    NUM_EPOCHS = 10     # 시간 관계상 적게 설정 (필요시 늘림)
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# --- Dataset ---
class RecSysDataset(Dataset):
    def __init__(self, user_seqs, num_items, max_len=50, is_train=True):
        self.user_seqs = user_seqs
        self.num_items = num_items
        self.max_len = max_len
        self.is_train = is_train

    def __len__(self):
        return len(self.user_seqs)

    def __getitem__(self, index):
        # user_seqs[index]는 [item_id_list, target_item] 형태가 아니라
        # 전체 시퀀스 리스트여야 함.
        # 학습: [:-1]입력 -> [1:] 예측
        # 추론: 전체 입력 -> 다음 아이템 예측
        
        seq = self.user_seqs[index]
        
        # 패딩 처리
        if len(seq) > self.max_len:
            seq = seq[-self.max_len:]
        else:
            seq = [0] * (self.max_len - len(seq)) + seq
            
        seq = torch.LongTensor(seq)
        return seq

# --- Model (SASRec Simplified) ---
class SASRec(nn.Module):
    def __init__(self, num_items, cfg):
        super(SASRec, self).__init__()
        self.num_items = num_items
        self.cfg = cfg
        
        # +1 for padding (index 0)
        self.item_emb = nn.Embedding(num_items + 1, cfg.HIDDEN_UNITS, padding_idx=0)
        self.pos_emb = nn.Embedding(cfg.MAX_LEN, cfg.HIDDEN_UNITS)
        self.emb_dropout = nn.Dropout(cfg.DROPOUT_RATE)
        
        # Transformer Blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.HIDDEN_UNITS,
            nhead=cfg.NUM_HEADS,
            dim_feedforward=cfg.HIDDEN_UNITS * 4,
            dropout=cfg.DROPOUT_RATE,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.NUM_BLOCKS)
        
        # Layer Norm
        self.ln = nn.LayerNorm(cfg.HIDDEN_UNITS)
        
        # 초기화
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
        # item_seq: (B, L)
        device = item_seq.device
        seq_len = item_seq.size(1)
        
        # Masking for generating causality (Look-ahead mask)
        # However, nn.TransformerEncoderLayer automatic mask handling is tricky.
        # We use src_mask.
        # Attention mask: (L, L) - Upper triangular is -inf
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)
        
        # Embedding
        ids = torch.arange(seq_len, dtype=torch.long, device=device)
        ids = ids.unsqueeze(0).expand(item_seq.size(0), -1)
        
        # Padding Mask (key_padding_mask) for transformer
        # (B, L) True where value is 0 (padding)
        padding_mask = (item_seq == 0)
        
        emb = self.item_emb(item_seq) + self.pos_emb(ids)
        emb = self.emb_dropout(emb)
        
        # Transformer Pass
        # src_mask makes it causal (can't see future)
        # src_key_padding_mask ignores padding
        output = self.transformer_encoder(emb, mask=mask, src_key_padding_mask=padding_mask)
        
        output = self.ln(output)
        
        return output # (B, L, H)

    def predict(self, item_seq):
        # 마지막 시점의 output만 사용하여 전체 아이템 점수 계산
        output = self.forward(item_seq) # (B, L, H)
        last_output = output[:, -1, :] # (B, H) - Last Item Embedding
        
        # 내적 (All Items)
        # item_emb.weight: (NumItems+1, H)
        # Remove padding idx 0 for prediction usually, but here just mul all
        logits = torch.matmul(last_output, self.item_emb.weight.t()) # (B, NumItems+1)
        
        # Mask padding index score
        logits[:, 0] = -1e9
        
        return logits

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    count = 0
    
    for seq in dataloader:
        seq = seq.to(device) # (B, L)
        
        # 입력: [0, 1, 2, 3] -> 마지막 제외
        # 타겟: [1, 2, 3, 4] -> 첫번째 제외 (Shifted)
        
        # 실제 데이터는 패딩[0, 0, A, B, C] 형태일 수 있음.
        # 입력: [0, 0, A, B]
        # 타겟: [0, A, B, C]
        
        # 그러나 간단한 구현을 위해 여기서는 
        # Last Item Prediction Task로만 학습 (BPR 대신 CrossEntropy)
        # 시퀀스의 모든 스텝을 학습하려면 복잡해짐.
        # 여기선 'Last Step'만 학습하는 Simplified version이 아니라
        # 시퀀스 전체를 CrossEntropy로 학습함.
        
        logits = model(seq) # (B, L, H) @ (H, V) -> (B, L, V)
        logits = torch.matmul(logits, model.item_emb.weight.t())
        
        # Shift
        # input: seq[:, :-1]
        # target: seq[:, 1:]
        # 근데 위에서 model(seq)를 다 넣었으므로,
        # logits도 맞춰야 함.
        
        # 논리적 정합성: 
        # Causal Mask가 있으므로 t 시점의 출력은 t까지의 입력만 봄.
        # 따라서 t 시점의 출력이 t+1 아이템을 예측하도록 해야 함.
        
        # logits: (B, L, V) -> t번째 벡터로 t+1 예측
        # target: (B, L) -> t+1번째 아이템 (Shift Left)
        
        # 마지막 토큰은 미래가 없으므로 제외
        # 입력의 마지막 토큰으로 예측한건 무의미(정답 레이블이 seq 밖임)
        
        # preds: logits[:, :-1, :] -> (B, L-1, V)
        # targets: seq[:, 1:] -> (B, L-1)
        
        preds = logits[:, :-1, :]
        targets = seq[:, 1:]
        
        # Flatten for CE Loss
        preds = preds.reshape(-1, preds.size(-1))
        targets = targets.reshape(-1)
        
        # Padding(0)에 대한 Loss 제외 필요
        # CrossEntropyLoss의 ignore_index=0 이용
        
        loss = criterion(preds, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        count += 1
        
    return total_loss / count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--output_dir', type=str, default='../out/')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--max_len', type=int, default=50)
    args = parser.parse_args()
    
    cfg = Config()
    cfg.NUM_EPOCHS = args.epochs
    cfg.MAX_LEN = args.max_len
    
    print("="*60)
    print("SASRec Training Started")
    print(f"Device: {cfg.DEVICE}")
    print("="*60)
    
    # 1. 데이터 로드
    print("Loading data...")
    train_df = pd.read_parquet(os.path.join(args.data_dir, 'train.parquet'))
    
    # 시간순 정렬 (이미 되어있을 수 있지만 보장)
    if train_df['event_time'].dtype == 'object':
         train_df['event_time'] = pd.to_datetime(train_df['event_time'])
    train_df = train_df.sort_values(['user_id', 'event_time'])
    
    # Item ID Mapping (0 is reserved for padding)
    items = train_df['item_id'].unique()
    item_to_idx = {item: i+1 for i, item in enumerate(items)}
    idx_to_item = {i+1: item for i, item in enumerate(items)}
    num_items = len(items)
    
    print(f"Num Items: {num_items}")
    
    # User Sequence 생성
    print("Creating User Sequences...")
    # GroupBy 후 리스트 변환은 느릴 수 있음. 최적화:
    # user_id가 정렬되어 있다고 가정하고 numpy split 사용 등..
    # 여기서는 안전하게 pandas groupby 사용
    user_grp = train_df.groupby('user_id')['item_id'].apply(lambda x: [item_to_idx[i] for i in x]).to_dict()
    
    # Train/Test Split 없음 (모든 과거 데이터 학습)
    # Submission 대상 유저의 시퀀스 확보
    
    # Dataset
    all_seqs = list(user_grp.values())
    dataset = RecSysDataset(all_seqs, num_items, max_len=cfg.MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0) # mps/cuda 이슈 방지 위해 worker 0
    
    # Model
    model = SASRec(num_items, cfg).to(cfg.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # Padding 무시
    
    # Training Loop
    print("\nTraining...")
    for epoch in range(cfg.NUM_EPOCHS):
        start_time = time.time()
        avg_loss = train(model, dataloader, optimizer, criterion, cfg.DEVICE)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{cfg.NUM_EPOCHS} | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")
        
    print("\nTraining Complete!")
    
    # --- Prediction ---
    print("\nGenerating Submission...")
    sample_sub = pd.read_csv(os.path.join(args.data_dir, 'sample_submission.csv'))
    test_users = sample_sub['user_id'].unique()
    
    model.eval()
    results = []
    
    # Batch Prediction for speed
    # Test User들도 Dataset으로 만들어서 Batch 처리 가능
    # 여기서는 직관성을 위해 단순 Loop (성능상 조금 느릴 수 있음)
    
    # Test User Sequences
    test_seqs = []
    valid_users = []
    
    for u in test_users:
        if u in user_grp:
            seq = user_grp[u]
            if len(seq) > cfg.MAX_LEN:
                seq = seq[-cfg.MAX_LEN:]
            else:
                seq = [0] * (cfg.MAX_LEN - len(seq)) + seq
            test_seqs.append(seq)
            valid_users.append(u)
        else:
            # Cold Start User (이력 없음) -> Popularity (여기선 처리 안함, 빈 리스트)
            # 나중에 Popularity로 채우기 위해 빈 리스트 저장 안함 or 별도 처리
            pass
            
    # Convert to Tensor
    test_tensor = torch.LongTensor(test_seqs).to(cfg.DEVICE)
    # Batch 처리
    batch_size = 256
    num_batches = (len(test_tensor) + batch_size - 1) // batch_size
    
    user_recs_map = {}
    
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Predicting"):
            batch = test_tensor[i*batch_size : (i+1)*batch_size]
            
            # Predict
            logits = model.predict(batch) # (B, V)
            
            # Top-10
            # scores, indices = torch.topk(logits, 10)
            # 이미 본 아이템 필터링은 생략 (SASRec은 보통 다음 아이템 예측이므로 Re-consumption 허용하기도 함)
            # 필요하면 scores에 기록된 아이템 마스킹
            
            _, top_indices = torch.topk(logits, 10, dim=1)
            
            top_indices = top_indices.cpu().numpy() # (B, 10)
            
            # Map back to User ID
            current_users = valid_users[i*batch_size : (i+1)*batch_size]
            
            for u_idx, u_id in enumerate(current_users):
                recs = [idx_to_item[idx] for idx in top_indices[u_idx]]
                user_recs_map[u_id] = recs
                
    # Global Popularity for Cold Start
    # 간단히 train_df에서 계산
    pop_items = train_df['item_id'].value_counts().head(10).index.tolist()
    
    # Final Result List
    final_data = []
    for u in test_users:
        recs = user_recs_map.get(u, pop_items)
        for item in recs:
            final_data.append({'user_id': u, 'item_id': item})
            
    # Save
    sub_df = pd.DataFrame(final_data)
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'output_sasrec_v1.csv')
    sub_df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()

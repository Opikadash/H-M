# train.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import lightgbm as lgb
from typing import List, Dict, Any

# Load processed data
with open("./data/processed/articles.json", "r") as f:
    articles = json.load(f)

with open("./data/processed/users.json", "r") as f:
    users = json.load(f)

with open("./data/processed/train.json", "r") as f:
    train_transactions = json.load(f)

with open("./data/processed/test.json", "r") as f:
    test_transactions = json.load(f)

with open("./data/processed/recent_purchases.json", "r") as f:
    recent_transactions = json.load(f)

transactions = train_transactions + test_transactions

# SASRec Model
class SASRec(nn.Module):
    def __init__(self, num_items, embedding_dim, num_layers, dropout=0.2):
        super(SASRec, self).__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(200, embedding_dim)  # Max sequence length 200
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=4,
                dim_feedforward=embedding_dim * 4,
                dropout=dropout,
                activation="relu"
            ) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, num_items)
    
    def forward(self, item_seq, positions):
        item_emb = self.item_embedding(item_seq)
        pos_emb = self.position_embedding(positions)
        x = item_emb + pos_emb
        x = self.dropout(x)
        
        for layer in self.transformer_layers:
            x = layer(x)
        
        logits = self.fc(x)
        return logits

class PurchaseDataset(Dataset):
    def __init__(self, transactions, user_ids, product_ids, max_seq_len=50, mask_ratio=0.8):
        self.transactions = transactions
        self.user_ids = user_ids
        self.product_ids = product_ids
        self.max_seq_len = max_seq_len
        self.mask_ratio = mask_ratio
        
        self.user_to_items = {}
        for t in transactions:
            user_id = t["userId"]
            if user_id not in self.user_to_items:
                self.user_to_items[user_id] = []
            self.user_to_items[user_id].append(t["productId"])
        
        for user_id in self.user_to_items:
            self.user_to_items[user_id].sort(key=lambda x: transactions[[t["productId"] for t in transactions].index(x)]["timestamp"])
    
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        items = self.user_to_items.get(user_id, [])
        
        if len(items) > self.max_seq_len:
            items = items[-self.max_seq_len:]
        else:
            items = [0] * (self.max_seq_len - len(items)) + items
        
        # Mask items for prediction
        masked_items = items.copy()
        for i in range(len(masked_items)):
            if np.random.random() < self.mask_ratio:
                masked_items[i] = 0  # Masked token
        
        positions = list(range(self.max_seq_len))
        
        # Convert product IDs to indices
        item_indices = [self.product_ids.index(item) if item in self.product_ids else 0 for item in items]
        masked_indices = [self.product_ids.index(item) if item in self.product_ids else 0 for item in masked_items]
        
        return {
            "item_seq": torch.tensor(masked_indices, dtype=torch.long),
            "positions": torch.tensor(positions, dtype=torch.long),
            "labels": torch.tensor(item_indices, dtype=torch.long)
        }

def pretrain_sasrec():
    """Pretrains the SASRec model with masked item prediction."""
    print("Pretraining SASRec...")
    
    # Prepare data
    user_ids = sorted(list(set(u["id"] for u in users)))
    product_ids = sorted(list(set(p["id"] for p in articles)))
    dataset = PurchaseDataset(train_transactions, user_ids, product_ids, mask_ratio=0.8)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # Initialize model
    num_items = len(product_ids)
    model = SASRec(num_items=num_items, embedding_dim=64, num_layers=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Training loop
    model.train()
    for epoch in range(50):
        total_loss = 0
        for batch in dataloader:
            item_seq = batch["item_seq"]
            positions = batch["positions"]
            labels = batch["labels"]
            
            optimizer.zero_grad()
            logits = model(item_seq, positions)
            loss = criterion(logits.view(-1, num_items), labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/50, Loss: {total_loss / len(dataloader):.4f}")
    
    return model

def fine_tune_sasrec(model):
    """Fine-tunes the SASRec model using contrastive learning with hard negatives."""
    print("Fine-tuning SASRec with contrastive learning...")
    
    user_ids = sorted(list(set(u["id"] for u in users)))
    product_ids = sorted(list(set(p["id"] for p in articles)))
    dataset = PurchaseDataset(train_transactions, user_ids, product_ids, mask_ratio=0.0)  # No masking for fine-tuning
    batch_size = 256
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    
    # Fine-tuning loop with dynamic batch sizing
    model.train()
    for epoch in range(10):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        total_loss = 0
        for batch in dataloader:
            item_seq = batch["item_seq"]
            positions = batch["positions"]
            labels = batch["labels"]
            
            # Get embeddings for contrastive learning
            with torch.no_grad():
                embeddings = model(item_seq, positions)[:, -1, :]  # Last position embedding
            
            # Contrastive loss with hard negatives
            positive_pairs = embeddings  # Use the same sequence as positive
            negative_indices = torch.randint(0, len(product_ids), (embeddings.size(0),))
            negative_items = torch.tensor([product_ids[i] for i in negative_indices])
            negative_seq = item_seq.clone()
            negative_seq[:, -1] = negative_items
            with torch.no_grad():
                negative_embeddings = model(negative_seq, positions)[:, -1, :]
            
            # Contrastive loss
            pos_sim = F.cosine_similarity(embeddings, positive_pairs)
            neg_sim = F.cosine_similarity(embeddings, negative_embeddings)
            loss = -torch.log(torch.sigmoid(pos_sim - neg_sim)).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Dynamic batch sizing (increase from 256 to 1024)
        batch_size = min(1024, batch_size + 128)
        
        print(f"Epoch {epoch + 1}/10, Loss: {total_loss / len(dataloader):.4f}, Batch Size: {batch_size}")
    
    return model

def distill_to_lightgbm(sasrec_model):
    """Distills the SASRec model into a LightGBM model with a 3:1 compression ratio."""
    print("Distilling SASRec to LightGBM...")
    
    # Generate soft labels using SASRec
    user_ids = sorted(list(set(u["id"] for u in users)))
    product_ids = sorted(list(set(p["id"] for p in articles)))
    dataset = PurchaseDataset(train_transactions, user_ids, product_ids, mask_ratio=0.0)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    sasrec_model.eval()
    soft_labels = []
    features = []
    with torch.no_grad():
        for batch in dataloader:
            item_seq = batch["item_seq"]
            positions = batch["positions"]
            logits = sasrec_model(item_seq, positions)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            soft_labels.extend(probs.numpy())
            
            # Use item sequence embeddings as features
            embeddings = sasrec_model.item_embedding(item_seq)
            features.extend(embeddings[:, -1, :].numpy())
    
    # Prepare LightGBM dataset
    features = np.array(features)  # Shape: (num_samples, 64)
    soft_labels = np.array(soft_labels)  # Shape: (num_samples, num_items)
    
    # Train LightGBM (simplified to a ranking task)
    train_data = lgb.Dataset(features, label=np.argmax(soft_labels, axis=1))
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "num_leaves": 31,  # Reduced for 3:1 compression
        "learning_rate": 0.05,
        "num_iterations": 100
    }
    gbm = lgb.train(params, train_data)
    
    print("Distillation complete!")
    return gbm

def main():
    # Pretrain SASRec
    sasrec_model = pretrain_sasrec()
    
    # Fine-tune SASRec
    sasrec_model = fine_tune_sasrec(sasrec_model)
    
    # Distill to LightGBM
    gbm_model = distill_to_lightgbm(sasrec_model)
    
    # Save the LightGBM model
    gbm_model.save_model(os.path.join(DATA_PATHS["output_dir"], "lightgbm_model.txt"))
    print("Saved LightGBM model.")

if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import numpy as np
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for the transformer model"""
    def __init__(self, d_model, max_seq_length=200):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class SASRec(nn.Module):
    """Self-Attentive Sequential Recommendation Model"""
    def __init__(self, num_items, embedding_dim=64, num_heads=4, 
                 dropout=0.2, max_seq_length=100):
        super(SASRec, self).__init__()
        
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embedding_dim, max_seq_length)
        
        # Multi-head self-attention layer
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=embedding_dim, 
            num_heads=num_heads, 
            dropout=dropout
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )
        
        # Layer normalization and dropout
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.output_layer = nn.Linear(embedding_dim, num_items + 1)
        
    def forward(self, seq, mask=None):
        # Get item embeddings and add positional encoding
        item_emb = self.item_embedding(seq)
        item_emb = self.pos_encoding(item_emb)
        
        # Create attention mask for padding
        if mask is None:
            mask = torch.zeros_like(seq).masked_fill(seq == 0, 1).bool()
        
        # Apply self-attention (transpose for PyTorch attention layer)
        attn_input = item_emb.transpose(0, 1)
        attn_output, _ = self.attention_layer(
            attn_input, attn_input, attn_input, 
            key_padding_mask=mask
        )
        attn_output = attn_output.transpose(0, 1)
        
        # Residual connection and layer normalization
        out = self.layer_norm1(item_emb + self.dropout(attn_output))
        
        # Feed-forward network
        ffn_output = self.ffn(out)
        out = self.layer_norm2(out + self.dropout(ffn_output))
        
        # Get final prediction
        logits = self.output_layer(out)
        
        return logits

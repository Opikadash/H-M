import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv

class GNNRecommender(nn.Module):
    def __init__(self, num_users, num_items, hidden_dims, embedding_dim=64):
        super(GNNRecommender, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # GNN layers
        self.conv1 = GraphConv(embedding_dim, hidden_dims[0])
        self.conv2 = GraphConv(hidden_dims[0], hidden_dims[1])
        self.conv3 = GraphConv(hidden_dims[1], hidden_dims[2])
        
        # Prediction layer
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dims[2] * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, g, user_indices, item_indices):
        # Initialize node features
        user_feat = self.user_embedding(torch.arange(self.user_embedding.num_embeddings))
        item_feat = self.item_embedding(torch.arange(self.item_embedding.num_embeddings))
        
        # Concatenate user and item features
        x = torch.cat([user_feat, item_feat], dim=0)
        
        # Apply GNN layers
        x = F.relu(self.conv1(g, x))
        x = F.relu(self.conv2(g, x))
        x = self.conv3(g, x)
        
        # Get user and item embeddings from final GNN layer
        user_embs = x[:self.user_embedding.num_embeddings]
        item_embs = x[self.user_embedding.num_embeddings:]
        
        # Extract embeddings for specific user-item pairs
        user_emb = user_embs[user_indices]
        item_emb = item_embs[item_indices]
        
        # Concatenate and predict
        concat_emb = torch.cat([user_emb, item_emb], dim=1)
        pred = self.predictor(concat_emb)
        
        return pred
    
    def create_graph(self, interactions):
        """Create bipartite graph from user-item interactions"""
        user_nodes = interactions['user_id'].unique()
        item_nodes = interactions['item_id'].unique()
        
        src_nodes = interactions['user_id'].values
        dst_nodes = interactions['item_id'].values + len(user_nodes)  # Offset item indices
        
        # Create bidirectional graph (user→item and item→user)
        g = dgl.graph((torch.tensor(src_nodes), torch.tensor(dst_nodes)))
        g.add_edges(torch.tensor(dst_nodes), torch.tensor(src_nodes))
        
        return g




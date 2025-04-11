# preprocess.py
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import gzip
from sentence_transformers import SentenceTransformer
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import umap
from typing import List, Dict, Any

# Set up project directory (adjust for Colab or VSCode)
project_dir = ''  # Colab
# project_dir = '.'  # VSCode
data_dir = os.path.join(project_dir, 'data')
processed_dir = os.path.join(project_dir, 'data', 'processed')

# Path configuration
DATA_PATHS = {
    "transactions": os.path.join(data_dir, "transactions_train.csv"),
    "articles": os.path.join(data_dir, "articles.csv"),
    "customers": os.path.join(data_dir, "customers.csv"),
    "image_embeddings": os.path.join(data_dir, "embeddings.csv.gz"),
    "output_dir": processed_dir
}

def ensure_output_dir():
    """Ensure the output directory exists."""
    try:
        os.makedirs(DATA_PATHS["output_dir"], exist_ok=True)
        print(f"Ensured output directory exists: {DATA_PATHS['output_dir']}")
    except Exception as e:
        print(f"Error creating output directory {DATA_PATHS['output_dir']}: {e}")
        raise

def load_transactions():
    """Loads transaction data from CSV, filtering for 2019 and 2020."""
    try:
        print(f"Loading transactions from {DATA_PATHS['transactions']}")
        df = pd.read_csv(DATA_PATHS["transactions"])
        
        # Map to PurchaseRecord format
        df["transactionId"] = df["customer_id"] + "_" + df["article_id"] + "_" + df["t_dat"]
        purchases = df[["customer_id", "article_id", "t_dat", "transactionId"]].rename(columns={
            "customer_id": "userId",
            "article_id": "productId",
            "t_dat": "timestamp"
        }).to_dict("records")
        
        # Filter for 2019 and 2020
        purchases = [
            p for p in purchases
            if datetime.strptime(p["timestamp"], "%Y-%m-%d").year in [2019, 2020]
        ]
        
        print(f"Loaded {len(purchases)} transactions for 2019 and 2020")
        return purchases
    except Exception as e:
        print(f"Error loading transactions: {e}")
        return []

def load_articles():
    """Loads article (product) data from CSV."""
    try:
        print(f"Loading articles from {DATA_PATHS['articles']}")
        df = pd.read_csv(DATA_PATHS["articles"])
        
        products = []
        for _, row in df.iterrows():
            product = {
                "id": str(row["article_id"]),
                "title": row.get("prod_name", "Unknown Product"),
                "description": row.get("detail_desc", ""),
                "category": [row.get("product_type_name", ""), row.get("product_group_name", "")],
                "imageUrl": f"/images/articles/{row['article_id']}.jpg",
                "price": float(row.get("price", 0)),
                "attributes": {
                    "color": row.get("colour_group_name", ""),
                    "department": row.get("department_name", ""),
                    "garmentGroup": row.get("garment_group_name", ""),
                    "index": row.get("index_code", ""),
                    "section": row.get("section_name", ""),
                    "indexGroup": row.get("index_group_name", ""),
                    "graphicalAppearance": row.get("graphical_appearance_name", ""),
                    "perceivedColour": row.get("perceived_colour_value_name", ""),
                    "perceivedColourMaster": row.get("perceived_colour_master_name", ""),
                    "pattern": "Floral" if "floral" in str(row.get("graphical_appearance_name", "")).lower() else "Striped" if "striped" in str(row.get("graphical_appearance_name", "")).lower() else ""
                }
            }
            products.append(product)
        
        print(f"Loaded {len(products)} articles")
        return products
    except Exception as e:
        print(f"Error loading articles: {e}")
        return []

def load_image_embeddings():
    """Loads precomputed image embeddings from CSV (if available)."""
    try:
        print(f"Loading image embeddings from {DATA_PATHS['image_embeddings']}")
        with gzip.open(DATA_PATHS["image_embeddings"], "rt", encoding="utf-8") as f:
            df = pd.read_csv(f)
        
        embeddings_map = {}
        expected_dim = 768
        for _, row in df.iterrows():
            article_id = str(row["article_id"])
            embedding = [float(row[str(i)]) for i in range(expected_dim)]
            embeddings_map[article_id] = embedding
        
        print(f"Loaded image embeddings for {len(embeddings_map)} articles")
        return embeddings_map
    except Exception as e:
        print(f"Error loading image embeddings: {e}")
        return {}

def load_customers():
    """Loads customer data from CSV."""
    try:
        print(f"Loading customers from {DATA_PATHS['customers']}")
        df = pd.read_csv(DATA_PATHS["customers"])
        
        customer_map = {}
        for _, row in df.iterrows():
            customer_map[row["customer_id"]] = {
                "age": int(row["age"]) if pd.notna(row.get("age")) else None,
                "fashionNewsFrequency": row.get("fashion_news_frequency", ""),
                "activeStatus": row.get("Active", "UNKNOWN"),
                "club": row.get("club_member_status", ""),
                "fnFrequency": row.get("FN", 0)
            }
        
        print(f"Loaded {len(customer_map)} customers")
        return customer_map
    except Exception as e:
        print(f"Error loading customers: {e}")
        return {}

def create_rolling_windows(purchases, window_days=30):
    """Creates rolling 30-day purchase windows, excluding pandemic outliers (2020-03 to 2020-06)."""
    windows = {}
    
    # Sort purchases by timestamp
    sorted_purchases = sorted(purchases, key=lambda x: datetime.strptime(x["timestamp"], "%Y-%m-%d"))
    
    # Exclude pandemic outliers (2020-03 to 2020-06)
    filtered_purchases = [
        p for p in sorted_purchases
        if not (datetime.strptime(p["timestamp"], "%Y-%m-%d") >= datetime(2020, 3, 1) and
                datetime.strptime(p["timestamp"], "%Y-%m-%d") <= datetime(2020, 6, 30))
    ]
    
    # Create rolling 30-day windows
    for purchase in filtered_purchases:
        date = datetime.strptime(purchase["timestamp"], "%Y-%m-%d")
        window_start = date - timedelta(days=date.day - 1)  # Start of the month
        window_key = f"{window_start.year}-{window_start.month:02d}"
        if window_key not in windows:
            windows[window_key] = []
        windows[window_key].append(purchase)
    
    print(f"Created {len(windows)} rolling windows")
    return windows

def filter_last_6_months(purchases):
    """Filters purchases to the last 6 months."""
    if not purchases:
        return []
    
    sorted_purchases = sorted(purchases, key=lambda x: datetime.strptime(x["timestamp"], "%Y-%m-%d"))
    last_date = datetime.strptime(sorted_purchases[-1]["timestamp"], "%Y-%m-%d")
    cutoff_date = last_date - timedelta(days=6*30)
    
    recent_purchases = [
        p for p in sorted_purchases
        if datetime.strptime(p["timestamp"], "%Y-%m-%d") >= cutoff_date
    ]
    
    print(f"Filtered to {len(recent_purchases)} purchases in the last 6 months")
    return recent_purchases

def generate_text_embeddings(products):
    """Generates text embeddings using Sentence-BERT (384-dim) on product descriptions."""
    print("Generating text embeddings using Sentence-BERT...")
    model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim embeddings
    
    descriptions = [p["description"] if p["description"] else p["title"] for p in products]
    text_embeddings = model.encode(descriptions, show_progress_bar=True)
    
    embedding_map = {}
    for product, embedding in zip(products, text_embeddings):
        embedding_map[product["id"]] = embedding.tolist()
    
    print(f"Generated text embeddings for {len(embedding_map)} products")
    return embedding_map

def merge_text_embeddings(products, text_embeddings):
    """Merges text embeddings into product data."""
    print("Merging text embeddings into product data...")
    updated_products = []
    for product in products:
        embedding = text_embeddings.get(product["id"])
        if embedding:
            product["textEmbedding"] = embedding
        else:
            product["textEmbedding"] = [0.0] * 384
        updated_products.append(product)
    
    print(f"Merged text embeddings for {len(updated_products)} products")
    return updated_products

class GNN(torch.nn.Module):
    """Simple GNN using Graph Convolutional Networks (GCN)."""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

def generate_gnn_embeddings(users, products, transactions):
    """Generates user embeddings using a GNN on the purchase graph."""
    print("Generating user embeddings using GNN...")

    # Create user and product ID mappings
    user_ids = sorted(list(set(u["id"] for u in users)))
    product_ids = sorted(list(set(p["id"] for p in products)))
    user_id_map = {uid: i for i, uid in enumerate(user_ids)}
    product_id_map = {pid: i + len(user_ids) for i, pid in enumerate(product_ids)}
    
    # Create edge index for the bipartite graph (user -> product)
    edge_index = [[], []]
    for t in transactions:
        user_idx = user_id_map[t["userId"]]
        product_idx = product_id_map[t["productId"]]
        edge_index[0].append(user_idx)  # Source: user
        edge_index[1].append(product_idx)  # Target: product
        edge_index[0].append(product_idx)  # Bidirectional edge
        edge_index[1].append(user_idx)
    
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    
    # Create node features (initially using text embeddings for products, zeros for users)
    num_nodes = len(user_ids) + len(product_ids)
    x = torch.zeros((num_nodes, 384))  # 384-dim from Sentence-BERT
    for product in products:
        product_idx = product_id_map[product["id"]]
        x[product_idx] = torch.tensor(product["textEmbedding"])
    
    # Create the graph data
    data = Data(x=x, edge_index=edge_index)
    
    # Define the GNN model
    model = GNN(in_channels=384, hidden_channels=128, out_channels=384)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # Pretrain the GNN for 50 epochs
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        # Simple self-supervised loss: reconstruct node features
        loss = torch.nn.functional.mse_loss(out, data.x)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/50, Loss: {loss.item():.4f}")
    
    # Get the embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)
    
    user_embeddings = {}
    for user in users:
        user_idx = user_id_map[user["id"]]
        user_embeddings[user["id"]] = embeddings[user_idx].tolist()
    
    product_embeddings = {}
    for product in products:
        product_idx = product_id_map[product["id"]]
        product_embeddings[product["id"]] = embeddings[product_idx].tolist()
    
    print(f"Generated GNN embeddings for {len(user_embeddings)} users and {len(product_embeddings)} products")
    return user_embeddings, product_embeddings

def reduce_dimensionality(embeddings, target_dim=64, variance_retained=0.98):
    """Reduces dimensionality of embeddings using UMAP, targeting 98% variance retention."""
    print(f"Reducing dimensionality to {target_dim}D with UMAP...")
    
    # Convert embeddings to numpy array
    embedding_list = list(embeddings.values())
    embedding_array = np.array(embedding_list)
    
    # Use UMAP for dimensionality reduction
    reducer = umap.UMAP(n_components=target_dim, random_state=42)
    reduced_embeddings = reducer.fit_transform(embedding_array)
    
    # Map back to dictionary
    reduced_embedding_map = {}
    for (key, _), reduced_embedding in zip(embeddings.items(), reduced_embeddings):
        reduced_embedding_map[key] = reduced_embedding.tolist()
    
    print(f"Reduced embeddings to {target_dim}D for {len(reduced_embedding_map)} items")
    return reduced_embedding_map

def create_time_based_split(purchases, test_months=3):
    """Creates a time-based train/test split with the last 3 months for test."""
    sorted_purchases = sorted(purchases, key=lambda x: datetime.strptime(x["timestamp"], "%Y-%m-%d"))
    
    if not sorted_purchases:
        return {"train": [], "test": []}
    
    last_date = datetime.strptime(sorted_purchases[-1]["timestamp"], "%Y-%m-%d")
    cutoff_date = last_date - timedelta(days=test_months*30)
    
    train = [p for p in sorted_purchases if datetime.strptime(p["timestamp"], "%Y-%m-%d") < cutoff_date]
    test = [p for p in sorted_purchases if datetime.strptime(p["timestamp"], "%Y-%m-%d") >= cutoff_date]
    
    print(f"Split data: {len(train)} train, {len(test)} test records")
    
    # Identify cold-start users and products (5% new users/items in test)
    train_user_ids = set(p["userId"] for p in train)
    train_product_ids = set(p["productId"] for p in train)
    
    test_user_ids = set(p["userId"] for p in test)
    test_product_ids = set(p["productId"] for p in test)
    
    cold_start_users = [p for p in test if p["userId"] not in train_user_ids]
    cold_start_products = [p for p in test if p["productId"] not in train_product_ids]
    
    # Ensure 5% cold-start subset
    cold_start_user_count = max(1, int(len(test_user_ids) * 0.05))
    cold_start_product_count = max(1, int(len(test_product_ids) * 0.05))
    
    cold_start_users = cold_start_users[:cold_start_user_count]
    cold_start_products = cold_start_products[:cold_start_product_count]
    
    print(f"Cold start data: {len(cold_start_users)} user records, {len(cold_start_products)} product records")
    
    return {"train": train, "test": test, "cold_start_users": cold_start_users, "cold_start_products": cold_start_products}

def save_to_json(data, filename):
    """Saves data to JSON format."""
    try:
        print(f"Saving data to {filename}")
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Successfully saved {filename}")
    except Exception as e:
        print(f"Error saving to JSON file {filename}: {e}")
        raise

def create_processed_dataset():
    """Creates processed dataset for recommendation training."""
    try:
        print("Starting dataset processing pipeline...")
        ensure_output_dir()
        
        # Load raw data
        transactions = load_transactions()
        articles = load_articles()
        customer_map = load_customers()
        image_embeddings = load_image_embeddings()
        
        # Generate text embeddings using Sentence-BERT
        text_embeddings = generate_text_embeddings(articles)
        products_with_text_embeddings = merge_text_embeddings(articles, text_embeddings)
        
        # Create rolling windows
        windows = create_rolling_windows(transactions, window_days=30)
        
        # Filter transactions to the last 6 months for recommendation purposes
        recent_purchases = filter_last_6_months(transactions)
        
        # Create user data structures
        user_ids = set(t["userId"] for t in transactions)
        users = [
            {
                "id": user_id,
                "purchaseHistory": [t for t in transactions if t["userId"] == user_id],
                "demographics": customer_map.get(user_id, {})
            }
            for user_id in user_ids
        ]
        
        print(f"Created user data for {len(users)} users")
        
        # Generate GNN embeddings
        user_embeddings, product_embeddings = generate_gnn_embeddings(users, products_with_text_embeddings, transactions)
        
        # Reduce dimensionality using UMAP
        user_embeddings_reduced = reduce_dimensionality(user_embeddings, target_dim=64)
        product_embeddings_reduced = reduce_dimensionality(product_embeddings, target_dim=64)
        
        # Add embeddings to user and product data
        users_with_embeddings = [
            {**user, "embedding": user_embeddings_reduced[user["id"]]}
            for user in users
        ]
        
        products_with_embeddings = [
            {**product, "embedding": product_embeddings_reduced[product["id"]]}
            for product in products_with_text_embeddings
        ]
        
        # Create train/test split (last 3 months for test)
        split = create_time_based_split(transactions, test_months=3)
        
        # Save to JSON files
        save_to_json(products_with_embeddings, os.path.join(DATA_PATHS["output_dir"], "articles.json"))
        save_to_json(users_with_embeddings, os.path.join(DATA_PATHS["output_dir"], "users.json"))
        save_to_json(split["train"], os.path.join(DATA_PATHS["output_dir"], "train.json"))
        save_to_json(split["test"], os.path.join(DATA_PATHS["output_dir"], "test.json"))
        save_to_json(split["cold_start_users"], os.path.join(DATA_PATHS["output_dir"], "cold_start_users.json"))
        save_to_json(split["cold_start_products"], os.path.join(DATA_PATHS["output_dir"], "cold_start_products.json"))
        save_to_json(windows, os.path.join(DATA_PATHS["output_dir"], "rolling_windows.json"))
        save_to_json(recent_purchases, os.path.join(DATA_PATHS["output_dir"], "recent_purchases.json"))
        
        print("Dataset processing complete!")
        return products_with_embeddings, users_with_embeddings, transactions, recent_purchases, split
    except Exception as e:
        print(f"Error in dataset processing: {e}")
        raise

if __name__ == "__main__":
    create_processed_dataset()

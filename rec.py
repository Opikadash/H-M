import json
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class FashionRecommendationSystem:
    def __init__(self, data_dir: str = "./data/processed/"):
        """
        Initialize the recommendation system with pre-loaded data
        Args:
            data_dir: Path to directory containing processed JSON files
        """
        self.data_dir = data_dir
        self.articles = []
        self.customers = []
        self.transactions = []
        self.recent_transactions = []
        
        # Data structures for fast lookup
        self.article_dict = {}
        self.customer_dict = {}
        self.article_embeddings = []
        self.customer_embeddings = []
        self.article_ids = []
        
        # Nearest Neighbors models
        self.article_nn = None
        self.customer_nn = None
        
        # Purchase statistics
        self.purchase_counts = defaultdict(int)
        self.article_popularity = defaultdict(int)
        
        self.load_data()
        self.build_indices()
    
    def load_data(self):
        """Load and preprocess all required data files"""
        print("Loading data...")
        
        # Load articles data
        articles_path = os.path.join(self.data_dir, "articles.json")
        if os.path.exists(articles_path):
            with open(articles_path, 'r') as f:
                self.articles = json.load(f)
            self.article_dict = {a['id']: a for a in self.articles}
            self.article_ids = [a['id'] for a in self.articles]
            
            # Extract embeddings if available
            self.article_embeddings = np.array([
                a.get('embedding', [0.0]*64) for a in self.articles
            ], dtype=np.float32)
        
        # Load customers data
        customers_path = os.path.join(self.data_dir, "users.json")
        if os.path.exists(customers_path):
            with open(customers_path, 'r') as f:
                self.customers = json.load(f)
            self.customer_dict = {c['id']: c for c in self.customers}
            
            # Extract customer embeddings
            self.customer_embeddings = np.array([
                c.get('embedding', [0.0]*64) for c in self.customers
            ], dtype=np.float32)
        
        # Load transactions
        transactions_path = os.path.join(self.data_dir, "transactions_combined.json")
        if os.path.exists(transactions_path):
            with open(transactions_path, 'r') as f:
                self.transactions = json.load(f)
        
        # Load recent transactions
        recent_path = os.path.join(self.data_dir, "recent_purchases.json")
        if os.path.exists(recent_path):
            with open(recent_path, 'r') as f:
                self.recent_transactions = json.load(f)
            
            # Precompute purchase counts
            for t in self.recent_transactions:
                self.purchase_counts[t['productId']] += 1
            
            # Calculate article popularity scores
            for article in self.articles:
                self.article_popularity[article['id']] = self.purchase_counts.get(article['id'], 0)
    
    def build_indices(self):
        """Build nearest neighbor indices for fast similarity search"""
        print("Building search indices...")
        
        # Article similarity index
        if len(self.article_embeddings) > 0:
            self.article_nn = NearestNeighbors(
                n_neighbors=50,
                metric='cosine',
                algorithm='brute'  # Best for cosine similarity
            )
            self.article_nn.fit(self.article_embeddings)
        
        # Customer similarity index
        if len(self.customer_embeddings) > 0:
            self.customer_nn = NearestNeighbors(
                n_neighbors=20,
                metric='cosine',
                algorithm='brute'
            )
            self.customer_nn.fit(self.customer_embeddings)
    
    def recommend_for_customer(self, customer_id: str, num_items: int = 12) -> List[Dict]:
        """
        Get personalized recommendations for a customer
        Args:
            customer_id: ID of the customer
            num_items: Number of recommendations to return
        Returns:
            List of recommended article dictionaries
        """
        if customer_id not in self.customer_dict:
            return self.get_trending_items(num_items)
        
        # Get similar customers first
        customer_idx = list(self.customer_dict.keys()).index(customer_id)
        _, similar_cust_indices = self.customer_nn.kneighbors(
            [self.customer_embeddings[customer_idx]],
            n_neighbors=5
        )
        
        # Get articles purchased by similar customers
        similar_customers = [list(self.customer_dict.keys())[i] for i in similar_cust_indices[0]]
        candidate_articles = set()
        
        for t in self.transactions:
            if t['userId'] in similar_customers:
                candidate_articles.add(t['productId'])
        
        # Rank candidate articles
        if not candidate_articles:
            return self.get_trending_items(num_items)
        
        # Convert to embedding space
        candidate_indices = [
            i for i, article in enumerate(self.articles)
            if article['id'] in candidate_articles
        ]
        
        if not candidate_indices:
            return self.get_trending_items(num_items)
        
        # Get most similar articles to customer's preferences
        customer_embedding = self.customer_embeddings[customer_idx]
        candidate_embeddings = self.article_embeddings[candidate_indices]
        
        # Calculate similarities
        similarities = np.dot(candidate_embeddings, customer_embedding) / (
            np.linalg.norm(candidate_embeddings, axis=1) * np.linalg.norm(customer_embedding) + 1e-9
        )
        
        # Combine with popularity
        scores = []
        for i, idx in enumerate(candidate_indices):
            article_id = self.articles[idx]['id']
            score = 0.7 * similarities[i] + 0.3 * self.article_popularity.get(article_id, 0)
            scores.append((idx, score))
        
        # Sort and get top items
        scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [x[0] for x in scores[:num_items]]
        
        return [self.format_article(self.articles[i]) for i in top_indices]
    
    def similar_items(self, article_id: str, num_items: int = 8) -> List[Dict]:
        """
        Get similar items based on product embedding
        Args:
            article_id: ID of the reference article
            num_items: Number of similar items to return
        Returns:
            List of similar article dictionaries
        """
        if article_id not in self.article_dict or self.article_nn is None:
            return []
        
        article_idx = self.article_ids.index(article_id)
        distances, indices = self.article_nn.kneighbors(
            [self.article_embeddings[article_idx]],
            n_neighbors=num_items+1  # +1 to exclude self
        )
        
        # Exclude the query article itself
        results = []
        for i in indices[0][1:num_items+1]:
            results.append(self.format_article(self.articles[i]))
        
        return results
    
    def get_trending_items(self, num_items: int = 12) -> List[Dict]:
        """
        Get currently trending items based on recent purchases
        Args:
            num_items: Number of trending items to return
        Returns:
            List of trending article dictionaries
        """
        # Sort by purchase count
        sorted_articles = sorted(
            self.articles,
            key=lambda x: self.article_popularity.get(x['id'], 0),
            reverse=True
        )
        
        return [self.format_article(a) for a in sorted_articles[:num_items]]
    
    def visual_search(self, embedding: List[float], num_items: int = 8) -> List[Dict]:
        """
        Find similar items based on visual embedding
        Args:
            embedding: Image embedding vector
            num_items: Number of results to return
        Returns:
            List of visually similar article dictionaries
        """
        if self.article_nn is None:
            return []
        
        embedding = np.array(embedding, dtype=np.float32).reshape(1, -1)
        distances, indices = self.article_nn.kneighbors(
            embedding,
            n_neighbors=num_items
        )
        
        return [self.format_article(self.articles[i]) for i in indices[0]]
    
    def format_article(self, article: Dict) -> Dict:
        """
        Standardize article dictionary format
        Args:
            article: Raw article dictionary
        Returns:
            Formatted article dictionary
        """
        return {
            'id': article['id'],
            'name': article.get('title', ''),
            'description': article.get('description', ''),
            'price': article.get('price', 0),
            'image_url': article.get('image_url', ''),
            'popularity': self.article_popularity.get(article['id'], 0)
        }

# Example usage
if __name__ == "__main__":
    # Initialize system
    rec_sys = FashionRecommendationSystem()
    
    # Example recommendations
    print("\nPersonalized recommendations:")
    print(rec_sys.recommend_for_customer("customer123", 5))
    
    print("\nSimilar items:")
    print(rec_sys.similar_items("article456", 5))
    
    print("\nTrending items:")
    print(rec_sys.get_trending_items(5))
    
    print("\nVisual search:")
    dummy_embedding = [0.1] * 64  # Replace with real image embedding
    print(rec_sys.visual_search(dummy_embedding, 5))

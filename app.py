# app.py
from flask import Flask, render_template, request
import numpy as np
from rec import FashionRecommender, articles, customers, transactions, recent_transactions

app = Flask(__name__)

# Initialize the recommender
recommender = FashionRecommender(articles, customers, transactions, recent_transactions)

def prepare_product_props(product, is_recommended=False, is_selected=False, similarity=None):
    """Prepares props for rendering a product card."""
    return {
        "product": {
            "id": product["id"],
            "title": product["name"],
            "description": product["description"],
            "category": product["category"] if "category" in product else ["Unknown"],
            "imageUrl": product.get("image", ""),
            "price": product["price"],
            "attributes": {
                "color": product["attributes"]["color"] if "color" in product["attributes"] else "",
                "pattern": product["attributes"].get("pattern", "")
            }
        },
        "isRecommended": is_recommended,
        "isAccessory": "Accessories" in product["category"],
        "isFootwear": "Footwear" in product["category"],
        "isSelected": is_selected,
        "similarity": similarity
    }

@app.route('/')
def index():
    """Renders the homepage with personalized recommendations for a user."""
    # Default to the first user if no user_id is provided
    user_id = request.args.get("user_id", customers[0]["id"])
    num_items = int(request.args.get("num_items", 8))
    
    # Get personalized recommendations
    personalized = recommender.get_personalized_recommendations(user_id, num_items=num_items)
    
    # Prepare props for each product
    products_with_props = [
        prepare_product_props(product, is_recommended=True, similarity=product["similarity"])
        for product in personalized
    ]
    
    return render_template('index.html', products=products_with_props, section_title="Personalized Recommendations")

@app.route('/trending')
def trending():
    """Renders trending items."""
    num_items = int(request.args.get("num_items", 6))
    
    # Get trending items
    trending_items = recommender.get_trending_items(num_items=num_items)
    
    # Prepare props for each product
    products_with_props = [
        prepare_product_props(product, is_recommended=False)
        for product in trending_items
    ]
    
    return render_template('index.html', products=products_with_props, section_title="Trending Items")

@app.route('/similar/<article_id>')
def similar(article_id):
    """Renders similar items for a given article."""
    num_items = int(request.args.get("num_items", 6))
    
    # Get similar items
    similar_items = recommender.get_similar_items(article_id, num_items=num_items)
    
    # Prepare props for each product
    products_with_props = [
        prepare_product_props(product, is_recommended=False, similarity=product["similarity"])
        for product in similar_items
    ]
    
    return render_template('index.html', products=products_with_props, section_title=f"Similar Items to Article {article_id}")

@app.route('/search')
def search():
    """Renders items based on a search query."""
    query = request.args.get("query", "")
    num_items = int(request.args.get("num_items", 8))
    
    if not query:
        return render_template('index.html', products=[], section_title="Search Results (No Query)")
    
    # Get name-based recommendations
    search_results = recommender.get_name_based_recommendations(query, num_items=num_items)
    
    # Prepare props for each product
    products_with_props = [
        prepare_product_props(product, is_recommended=False)
        for product in search_results
    ]
    
    return render_template('index.html', products=products_with_props, section_title=f"Search Results for '{query}'")

@app.route('/image-based', methods=['GET', 'POST'])
def image_based():
    """Renders recommendations based on a simulated image embedding."""
    num_items = int(request.args.get("num_items", 8))
    
    # Simulate an image embedding (in a real app, you'd process an uploaded image)
    simulated_image_embedding = np.random.randn(64).tolist()  # 64D after UMAP reduction
    
    # Get image-based recommendations
    image_based_items = recommender.get_image_based_recommendations(simulated_image_embedding, num_items=num_items)
    
    # Prepare props for each product
    products_with_props = [
        prepare_product_props(product, is_recommended=False, similarity=product["similarity"])
        for product in image_based_items
    ]
    
    return render_template('index.html', products=products_with_props, section_title="Image-Based Recommendations (Simulated)")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)

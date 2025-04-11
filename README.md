# H&M Personalized Fashion Recommendation System
Implementation of different models and analyzing their accuracy as per the kaggle competition standard. 
After research and evaluating different models ,the conclusion reached was that a hybrid approch seems the way to go and my proposal for the recommendation model is hybrid pipeline of SASRec+GNN+Faiss

# 1. Performance Considerations:
   - Implemented: Added dask support for loading large datasets (Step 6).
   - Implemented: Reduced batch sizes for Sentence-BERT (64), GNN (64), SASRec (32), and TabNet (512) to handle memory constraints.
  - The script uses Parquet files for faster read/write and smaller file sizes.
  - GNN and SASRec training use efficient data loaders (NeighborLoader, batching).

# 2. Model Improvements:
  - Implemented: SASRec now uses 4 blocks and 8 heads for better sequential modeling (Step 10).
  - Implemented: Faiss now uses an IVF index with 100 clusters for faster search (Step 11).
  - Implemented: TabNet hyperparameters adjusted (n_d=128, n_a=128, n_steps=5) for better performance (Step 12).
  - GNN already includes dropout, L2 regularization, and early stopping to prevent overfitting.

# 3. Embedding Quality:
  - Implemented: Upgraded Sentence-BERT to all-mpnet-base-v2 (768D) for better text embeddings (Step 9a).
   - Implemented: Adjusted UMAP n_components to retain at least 98% variance based on PCA (Step 9c).
   - User embeddings combine GNN and SASRec for relational and sequential patterns.

# 4. Evaluation Insights:
   - Implemented: Tested different ensemble weights (TabNet vs. SASRec) and selected the best based on MAP@12 (Step 13).
   - Implemented: Increased the number of candidate items retrieved by Faiss to 200 (Step 13).
   - SASRec and TabNet have been fine-tuned (more blocks/heads for SASRec, adjusted hyperparameters for TabNet).
   - Metrics (RMSE, MSE, MAE, R-squared, MAP@12) are computed and saved for analysis.

# 5. Next Steps:
   - Implemented: Tested different ensemble weights and saved the best submission (Step 13).
   - Use Faiss indices for real-time retrieval of similar users/items.
   - Example: Retrieve top-k similar items for a given item
      item_index = faiss.read_index(f"{Config.processed_dir}/item_faiss_index.bin")
     query_embedding = item_combined_embeddings[0:1]  # Example query
     distances, indices = item_index.search(query_embedding, k=5)
   - Use TabNet feature importance for business insights (e.g., which features drive purchases).
   - Use the evaluation metrics to compare different model configurations or ensemble weights.



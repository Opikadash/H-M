import os
import pandas as pd
import json
from tqdm import tqdm

def build_image_mapping(image_dir):
    """Scan all images and create article_id to path mapping"""
    mapping = {}
    
    # Walk through all image directories
    for root, _, files in tqdm(os.walk(image_dir), desc="Scanning folders"):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')):
                # Extract possible article IDs from filename
                clean_name = file.split('.')[0].lstrip('0')  # Remove .jpg and leading zeros
                
                # Map both versions (with/without leading zero)
                mapping[clean_name] = os.path.join(root, file)
                if clean_name != file.split('.')[0]:
                    mapping[file.split('.')[0]] = os.path.join(root, file)
    
    return mapping

if __name__ == '__main__':
    image_dir = "C:/Users/KIIT/H-M-Fashion-RecSys/data/images/"
    output_path = "C:/Users/KIIT/H-M-Fashion-RecSys/data/processed/article_to_image_mapping.csv"
    
    # Create mapping
    print("Building image mapping...")
    mapping = build_image_mapping(image_dir)
    
    # Convert to DataFrame and save
    mapping_df = pd.DataFrame({
        "article_id": list(mapping.keys()),
        "image_path": list(mapping.values())
    })
    
    # Extract just the image ID from path for consistency
    mapping_df["image_id"] = mapping_df["image_path"].str.extract(r'(\d+)\.jpg$')[0]
    
    mapping_df.to_csv(output_path, index=False)
    print(f"Saved mapping to {output_path}")
    print(f"Found {len(mapping_df)} images")

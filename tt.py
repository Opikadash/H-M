#import pandas as pdimport gzip

#with gzip.open("C:/Users/KIIT/H-M-Fashion-RecSys/data/processed/swin_tiny_patch4_window7_224_emb.csv.gz", "rt") as f:df = pd.read_csv(f)
    #print(df.head())
import os
import pandas as pd
import torch
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
import gzip
from tqdm import tqdm
import csv
from PIL import Image
import timm

# Configuration should be at the TOP level
CONFIG = {
    "input_files": {
        "articles": "C:/Users/KIIT/H-M-Fashion-RecSys/data/articles.csv",
        "mappings": "C:/Users/KIIT/H-M-Fashion-RecSys/data/processed/article_to_image_mapping.csv",
        "existing_embeddings": [
            "C:/Users/KIIT/H-M-Fashion-RecSys/data/processed/swin_tiny_patch4_window7_224_emb.csv",
            "C:/Users/KIIT/H-M-Fashion-RecSys/data/processed/opencv_embeddings.csv",
            "C:/Users/KIIT/H-M-Fashion-RecSys/data/processed/embeddings.csv"
        ]
    },
    "output_files": {
        "embeddings": "C:/Users/KIIT/H-M-Fashion-RecSys/data/processed/consolidated_embeddings.csv",
        "compressed": "C:/Users/KIIT/H-M-Fashion-RecSys/data/processed/consolidated_embeddings.csv.gz"
    },
    "image_dir": "C:/Users/KIIT/H-M-Fashion-RecSys/data/images/",
    "batch_size": 64,
    "model_name": "swin_tiny_patch4_window7_224"
}

def verify_paths():
    """Check if all required files exist before processing"""
    missing = []
    for key, path in CONFIG["input_files"].items():
        if isinstance(path, list):
            for p in path:
                if not os.path.exists(p):
                    print(f"Note: Optional file not found - {p}")
        elif not os.path.exists(path):
            missing.append(path)
    
    # Only critical paths
    critical_paths = [
        CONFIG["input_files"]["articles"],
        CONFIG["input_files"]["mappings"],
        CONFIG["image_dir"]
    ]
    
    missing_critical = [p for p in critical_paths if not os.path.exists(p)]
    
    if missing_critical:
        print("ERROR: Missing required files/directories:")
        for path in missing_critical:
            print(f"- {path}")
        print("\nPlease ensure:")
        print(f"1. articles.csv exists at {CONFIG['input_files']['articles']}")
        print(f"2. Mapping file exists at {CONFIG['input_files']['mappings']}")
        print(f"3. Image directory exists at {CONFIG['image_dir']}")
        return False
    return True

class CV2Dataset(Dataset):
    def __init__(self, article_ids, path_lookup, existing_embeddings):
        self.article_ids = article_ids
        self.path_lookup = path_lookup
        self.existing_embeddings = existing_embeddings
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.article_ids)
    
    def __getitem__(self, idx):
        article_id = self.article_ids[idx]
        
        # Use existing embedding if available
        if article_id in self.existing_embeddings:
            return article_id, torch.tensor(self.existing_embeddings[article_id]), False
        
        # Otherwise process image
        img_path = self.path_lookup.get(article_id, "")
        if os.path.exists(img_path):
            try:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                return article_id, self.transform(img), True
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
        return article_id, torch.zeros(3, 224, 224), False

def load_existing_embeddings():
    """Load and merge all existing embedding files"""
    embeddings = {}
    for path in CONFIG["input_files"]["existing_embeddings"]:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                for _, row in df.iterrows():
                    article_id = str(row['article_id'])
                    if article_id not in embeddings:
                        embeddings[article_id] = row.drop('article_id').values.tolist()
            except Exception as e:
                print(f"Warning: Could not load {path} - {str(e)}")
    return embeddings

def main():
    # Verify paths first
    if not verify_paths():
        exit(1)
    
    # Load existing embeddings
    print("Loading existing embeddings...")
    existing_embeddings = load_existing_embeddings()
    print(f"Loaded {len(existing_embeddings)} existing embeddings")
    
    # Load articles and mappings
    print("Loading articles and mappings...")
    articles_df = pd.read_csv(CONFIG["input_files"]["articles"], dtype={"article_id": str})
    mapping_df = pd.read_csv(CONFIG["input_files"]["mappings"], dtype={"article_id": str, "image_path": str})
    
    # Create dataset
    print("Preparing dataset...")
    dataset = CV2Dataset(
        articles_df["article_id"].tolist(),
        dict(zip(mapping_df["article_id"], mapping_df["image_path"])),
        existing_embeddings
    )
    
    # Initialize model
    print(f"Loading {CONFIG['model_name']} model...")
    model = timm.create_model(CONFIG["model_name"], pretrained=True, num_classes=0).eval()
    
    # Process and save
    os.makedirs(os.path.dirname(CONFIG["output_files"]["embeddings"]), exist_ok=True)
    
    with open(CONFIG["output_files"]["embeddings"], "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["article_id"] + [f"dim_{i}" for i in range(768)])
        
        # Write existing embeddings first
        for article_id, embedding in existing_embeddings.items():
            writer.writerow([article_id] + embedding)
        
        # Process remaining articles
        dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=False)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Processing images"):
                article_ids, images, needs_processing = batch
                if needs_processing.any():
                    embeddings = model(images[needs_processing]).cpu().numpy()
                    ptr = 0
                    for i, article_id in enumerate(article_ids):
                        if needs_processing[i]:
                            writer.writerow([article_id] + embeddings[ptr].tolist())
                            ptr += 1

    # Compress output
    print("Compressing results...")
    with open(CONFIG["output_files"]["embeddings"], "rb") as f_in:
        with gzip.open(CONFIG["output_files"]["compressed"], "wb") as f_out:
            f_out.writelines(f_in)

    print(f"\nSuccess! Output saved to:")
    print(f"- {CONFIG['output_files']['embeddings']}")
    print(f"- {CONFIG['output_files']['compressed']}")

if __name__ == "__main__":
    main()

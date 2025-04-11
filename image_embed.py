import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
import timm
import os
import gzip
from tqdm import tqdm
import csv
from torch.utils.data import Dataset, DataLoader
import numpy as np

def main():
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # Enable cuDNN benchmark for optimal performance

    # Load the articles data with optimized pandas settings
    articles_path = "C:/Users/KIIT/H-M-Fashion-RecSys/data/articles.csv"
    articles_df = pd.read_csv(articles_path, usecols=["article_id"], dtype={"article_id": str})
    article_ids = articles_df["article_id"].tolist()

    # Load the article to image mapping with optimized settings
    mapping_path = "C:/Users/KIIT/H-M-Fashion-RecSys/data/processed/article_to_image_mapping.csv"
    mapping_df = pd.read_csv(mapping_path, dtype={"article_id": str, "image_id": str})
    article_to_image_mapping = dict(zip(mapping_df["article_id"], mapping_df["image_id"]))

    # Define the image directory
    image_dir = "C:/Users/KIIT/H-M-Fashion-RecSys/data/images/"

    # Load the Swin Transformer model with optimizations
    model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=0)
    model = model.to(device)
    model.eval()

    # Freeze all model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Define image preprocessing with optimized transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create a dataset class for efficient loading
    class FashionDataset(Dataset):
        def __init__(self, article_ids, mapping, image_dir):
            self.article_ids = article_ids
            self.mapping = mapping
            self.image_dir = image_dir
            self.cache = {}  # Simple cache for found image paths
            
        def __len__(self):
            return len(self.article_ids)
        
        def find_image_path(self, article_id, image_id):
            if (article_id, image_id) in self.cache:
                return self.cache[(article_id, image_id)]
                
            # Try different folder naming patterns
            folder_patterns = [
                image_id[:3],  # First 3 digits
                article_id[:2].zfill(3),  # First 2 digits of article ID
                image_id[:2].zfill(3)  # First 2 digits of image ID
            ]
            
            # Try different image filename patterns
            filename_patterns = [
                f"0{image_id}.jpg",
                f"{image_id}.jpg",
                f"0{article_id}.jpg",
                f"{article_id}.jpg"
            ]
            
            for folder in folder_patterns:
                folder_path = os.path.join(self.image_dir, folder)
                if os.path.exists(folder_path):
                    for filename in filename_patterns:
                        image_path = os.path.join(folder_path, filename)
                        if os.path.exists(image_path):
                            self.cache[(article_id, image_id)] = image_path
                            return image_path
            return None
        
        def __getitem__(self, idx):
            article_id = self.article_ids[idx]
            image_id = self.mapping.get(article_id, article_id)
            image_path = self.find_image_path(article_id, image_id)
            
            if image_path:
                try:
                    image = Image.open(image_path).convert("RGB")
                    image = transform(image)
                    return article_id, image, True
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
            
            # Return dummy data if image not found
            dummy_image = torch.zeros(3, 224, 224)
            return article_id, dummy_image, False

    # Batch processing function with optimized inference
    @torch.no_grad()
    def process_batch(model, batch):
        article_ids, images, valid_flags = batch
        images = images.to(device, non_blocking=True)
        
        # Process only valid images
        valid_indices = [i for i, valid in enumerate(valid_flags) if valid]
        if valid_indices:
            valid_images = images[valid_indices]
            embeddings = model(valid_images)
            embeddings = embeddings.cpu().numpy()
        else:
            embeddings = np.zeros((0, 768))
        
        # Create full batch results with zeros for invalid images
        full_embeddings = np.zeros((len(images), 768))
        for i, idx in enumerate(valid_indices):
            full_embeddings[idx] = embeddings[i]
        
        return article_ids, full_embeddings

    # Setup DataLoader with single worker (to avoid multiprocessing issues on Windows)
    dataset = FashionDataset(article_ids, article_to_image_mapping, image_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=512,  # Larger batch size for better GPU utilization
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues on Windows
        pin_memory=True
    )

    # Output configuration
    output_path = "C:/Users/KIIT/H-M-Fashion-RecSys/data/processed/swin_tiny_patch4_window7_224_emb.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write headers
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["article_id"] + [f"dim_{i}" for i in range(768)])

    # Process data in batches with progress bar
    with torch.no_grad(), open(output_path, 'a', newline='') as f:
        writer = csv.writer(f)
        
        for batch in tqdm(dataloader, desc="Processing images"):
            article_ids, embeddings = process_batch(model, batch)
            
            # Write results to CSV
            for article_id, embedding in zip(article_ids, embeddings):
                writer.writerow([article_id] + embedding.tolist())

    print(f"Saved embeddings to {output_path}")

    # Compress the output with optimized settings
    output_gz_path = output_path + ".gz"
    with open(output_path, "rb") as f_in:
        with gzip.open(output_gz_path, "wb", compresslevel=6) as f_out:  # Balanced compression level
            f_out.writelines(f_in)

    print(f"Compressed embeddings to {output_gz_path}")

if __name__ == '__main__':
    main()

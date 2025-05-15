import os
import glob
from PIL import Image
import random

def explore_directory(path):
    """Recursively explore a directory and print its structure"""
    print(f"Exploring directory: {path}")
    
    try:
        # List all files and directories
        contents = os.listdir(path)
        print(f"Contents: {contents}")
        
        # Count files by type
        file_types = {}
        for item in contents:
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                print(f"\nFound directory: {item}")
                explore_directory(item_path)
            else:
                ext = os.path.splitext(item)[1].lower()
                if ext not in file_types:
                    file_types[ext] = 0
                file_types[ext] += 1
        
        if file_types:
            print(f"File types in {path}:")
            for ext, count in file_types.items():
                print(f"  {ext}: {count} files")
                
        # Try to find and sample some image files
        image_files = glob.glob(os.path.join(path, "**", "*.jpg"), recursive=True)
        if not image_files:
            image_files = glob.glob(os.path.join(path, "**", "*.png"), recursive=True)
            
        if image_files:
            print(f"\nFound {len(image_files)} images. Sampling a few:")
            samples = random.sample(image_files, min(5, len(image_files)))
            for img_path in samples:
                try:
                    img = Image.open(img_path)
                    print(f"  {img_path}: {img.size}, mode: {img.mode}")
                except Exception as e:
                    print(f"  Error opening {img_path}: {e}")
    
    except Exception as e:
        print(f"Error exploring {path}: {e}")

if __name__ == "__main__":
    dataset_path = "C:/Users/abdal/.cache/kagglehub/datasets/silverstone1903/deep-fashion-multimodal/versions/1"
    explore_directory(dataset_path) 
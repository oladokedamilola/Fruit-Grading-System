"""
Combine FIDS30 graded images with Fruits-360 extracted images
"""

import shutil
from pathlib import Path

def combine_datasets():
    """Combine FIDS30 and Fruits-360 datasets"""
    
    # Paths
    fids30_path = Path("ml/datasets/raw_fids30")  # Your FIDS30 graded images
    fruits360_path = Path("ml/datasets/raw")
    final_path = Path("ml/datasets/raw_combined")
    
    fruits = ["apples", "mangos", "oranges"]
    grades = ["A", "B", "C"]
    
    final_path.mkdir(parents=True, exist_ok=True)
    
    # Copy Fruits-360 images (these are already in ml/datasets/raw)
    for fruit in fruits:
        for grade in grades:
            source = fruits360_path / fruit / grade
            dest = final_path / fruit / grade
            if source.exists():
                dest.mkdir(parents=True, exist_ok=True)
                print(f"Copying {fruit}/{grade} from Fruits-360...")
                for img in source.glob("*"):
                    shutil.copy2(img, dest)
    
    # Copy FIDS30 graded images
    for fruit in fruits:
        for grade in grades:
            source = fids30_path / fruit / grade
            dest = final_path / fruit / grade
            if source.exists():
                dest.mkdir(parents=True, exist_ok=True)
                print(f"Copying {fruit}/{grade} from FIDS30...")
                for img in source.glob("*"):
                    shutil.copy2(img, dest)
    
    # Count final images
    print("\n📊 Combined Dataset Summary:")
    total = 0
    for fruit in fruits:
        for grade in grades:
            dest = final_path / fruit / grade
            if dest.exists():
                count = len(list(dest.glob("*")))
                print(f"  {fruit}/{grade}: {count} images")
                total += count
    print(f"\nTotal images: {total}")

if __name__ == "__main__":
    combine_datasets()
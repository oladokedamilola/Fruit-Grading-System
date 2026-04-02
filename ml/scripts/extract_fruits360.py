"""
Extract specific fruits from Fruits-360 dataset
Copy only apples, mangoes, oranges, tomatoes to your dataset
"""

import os
import shutil
from pathlib import Path
import random

def extract_fruits360():
    """Extract required fruits from Fruits-360"""
    
    # Paths
    fruits360_path = Path("Fruits-360")  # Change to your Fruits-360 folder location
    dest_path = Path("ml/datasets/raw")
    
    # Fruits we need (Fruits-360 has multiple varieties)
    fruit_mapping = {
        "apples": [
            "Apple Red Delicious", "Apple Golden", "Apple Red",
            "Apple Braeburn", "Apple Crimson Snow", "Apple Granny Smith"
        ],
        "mangos": ["Mango"],
        "oranges": ["Orange"],
        "tomatoes": ["Tomato"]
    }
    
    # Grade A destination
    for dest_fruit, source_varieties in fruit_mapping.items():
        dest_grade_a = dest_path / dest_fruit / "A"
        dest_grade_a.mkdir(parents=True, exist_ok=True)
        
        for variety in source_varieties:
            # Check Training folder
            source_train = fruits360_path / "Training" / variety
            if source_train.exists():
                print(f"Copying {variety} training images to {dest_fruit}/A...")
                for img in source_train.glob("*.jpg"):
                    # Use variety name to avoid filename conflicts
                    new_name = f"{variety.replace(' ', '_')}_{img.name}"
                    shutil.copy2(img, dest_grade_a / new_name)
            
            # Also check Test folder
            source_test = fruits360_path / "Test" / variety
            if source_test.exists():
                print(f"Copying {variety} test images to {dest_fruit}/A...")
                for img in source_test.glob("*.jpg"):
                    new_name = f"{variety.replace(' ', '_')}_{img.name}"
                    shutil.copy2(img, dest_grade_a / new_name)
    
    print("\n✅ Extraction complete!")
    
    # Count extracted images
    print("\n📊 Extracted counts:")
    for fruit in ["apples", "mangos", "oranges", "tomatoes"]:
        grade_a_path = dest_path / fruit / "A"
        if grade_a_path.exists():
            count = len(list(grade_a_path.glob("*.jpg")))
            print(f"  {fruit}/A: {count} images")

if __name__ == "__main__":
    extract_fruits360()
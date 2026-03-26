"""
Simplified Data Preprocessing Module - No external augmentation libraries required
Handles image preprocessing and dataset splitting
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import json
import random
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm

class SimpleDataPreprocessor:
    """Simplified data preprocessing without heavy dependencies"""
    
    def __init__(self, base_path="ml/datasets"):
        self.base_path = Path(base_path)
        self.raw_path = self.base_path / "raw"
        self.processed_path = self.base_path / "processed"
        self.annotations_path = self.base_path / "annotations"
        
        # Create directories
        self.processed_path.mkdir(parents=True, exist_ok=True)
        self.annotations_path.mkdir(parents=True, exist_ok=True)
        
        # Define fruit types and grades
        self.fruit_types = ["apples", "mangos", "oranges"]
        self.grades = ["A", "B", "C"]
        
        # Create mapping
        self.fruit_to_idx = {fruit: idx for idx, fruit in enumerate(self.fruit_types)}
        self.grade_to_idx = {grade: idx for idx, grade in enumerate(self.grades)}
    
    def scan_dataset(self):
        """Scan the dataset and create metadata"""
        print("\n🔍 Scanning dataset...")
        
        data = []
        
        for fruit in self.fruit_types:
            for grade in self.grades:
                grade_path = self.raw_path / fruit / grade
                if not grade_path.exists():
                    print(f"⚠ Warning: {grade_path} does not exist")
                    continue
                
                # Get all image files
                image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
                images = []
                for ext in image_extensions:
                    images.extend(grade_path.glob(ext))
                
                for img_path in images:
                    data.append({
                        'image_path': str(img_path),
                        'fruit_type': fruit,
                        'grade': grade,
                        'fruit_idx': self.fruit_to_idx[fruit],
                        'grade_idx': self.grade_to_idx[grade]
                    })
        
        df = pd.DataFrame(data)
        print(f"✓ Found {len(df)} images")
        
        for fruit in self.fruit_types:
            for grade in self.grades:
                count = len(df[(df['fruit_type'] == fruit) & (df['grade'] == grade)])
                if count > 0:
                    print(f"  {fruit}/{grade}: {count} images")
        
        # Save metadata
        df.to_csv(self.annotations_path / "dataset_metadata.csv", index=False)
        print(f"✓ Metadata saved to {self.annotations_path / 'dataset_metadata.csv'}")
        
        return df
    
    def split_dataset(self, df, test_size=0.15, val_size=0.15, random_state=42):
        """Split dataset into train, validation, and test sets"""
        print("\n📊 Splitting dataset...")
        
        if len(df) == 0:
            print("⚠ No images found. Please add images to ml/datasets/raw/")
            return None, None, None
        
        # Create stratified split
        stratify_col = df['fruit_type'] + '_' + df['grade']
        
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state,
            stratify=stratify_col
        )
        
        # Second split: separate validation from train
        stratify_col_train_val = train_val_df['fruit_type'] + '_' + train_val_df['grade']
        val_ratio = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_ratio, random_state=random_state,
            stratify=stratify_col_train_val
        )
        
        print(f"✓ Training set: {len(train_df)} images")
        print(f"✓ Validation set: {len(val_df)} images")
        print(f"✓ Test set: {len(test_df)} images")
        
        # Save splits
        train_df.to_csv(self.annotations_path / "train_split.csv", index=False)
        val_df.to_csv(self.annotations_path / "validation_split.csv", index=False)
        test_df.to_csv(self.annotations_path / "test_split.csv", index=False)
        
        # Create summary
        summary = {
            'total_images': len(df),
            'train': len(train_df),
            'validation': len(val_df),
            'test': len(test_df),
            'fruit_distribution': df['fruit_type'].value_counts().to_dict(),
            'grade_distribution': df['grade'].value_counts().to_dict()
        }
        
        with open(self.annotations_path / "dataset_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        return train_df, val_df, test_df
    
    def simple_augmentation(self, image):
        """Simple data augmentation without external libraries"""
        # Random horizontal flip
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        # Random brightness adjustment
        if random.random() > 0.5:
            brightness = random.uniform(0.7, 1.3)
            image = np.clip(image * brightness, 0, 255).astype(np.uint8)
        
        # Random rotation
        if random.random() > 0.5:
            angle = random.randint(-15, 15)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, matrix, (w, h))
        
        return image
    
    def create_augmented_copies(self, df, target_per_class=15):
        """Create augmented copies for classes with few images"""
        print("\n🔄 Creating augmented copies for small classes...")
        
        augmented_count = 0
        
        for fruit in self.fruit_types:
            for grade in self.grades:
                class_df = df[(df['fruit_type'] == fruit) & (df['grade'] == grade)]
                current_count = len(class_df)
                
                if current_count < target_per_class and current_count > 0:
                    needed = target_per_class - current_count
                    print(f"  {fruit}/{grade}: {current_count} → adding {needed} augmentations")
                    
                    for idx, row in class_df.iterrows():
                        if needed <= 0:
                            break
                        
                        # Load image
                        img = cv2.imread(row['image_path'])
                        if img is None:
                            continue
                        
                        # Create augmented version
                        for aug_idx in range(min(2, needed)):  # Max 2 per original
                            aug_img = self.simple_augmentation(img)
                            
                            # Save augmented image
                            original_path = Path(row['image_path'])
                            aug_filename = f"{original_path.stem}_aug{aug_idx}{original_path.suffix}"
                            aug_path = original_path.parent / aug_filename
                            cv2.imwrite(str(aug_path), aug_img)
                            needed -= 1
                            augmented_count += 1
        
        print(f"✓ Created {augmented_count} augmented images")
        return augmented_count
    
    def generate_eda_report(self, df):
        """Generate simple EDA report"""
        print("\n📊 Generating EDA Report...")
        
        # Create output directory
        output_dir = Path("ml/outputs/eda")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Basic statistics
        stats = {
            'total_images': len(df),
            'fruit_counts': df['fruit_type'].value_counts().to_dict(),
            'grade_counts': df['grade'].value_counts().to_dict(),
            'fruit_grade_matrix': pd.crosstab(df['fruit_type'], df['grade']).to_dict()
        }
        
        # Save stats
        with open(output_dir / "eda_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✓ EDA report saved to {output_dir}")
        
        # Print summary
        print("\n" + "=" * 50)
        print("DATASET SUMMARY")
        print("=" * 50)
        print(f"Total images: {stats['total_images']}")
        print("\nBy fruit:")
        for fruit, count in stats['fruit_counts'].items():
            print(f"  {fruit}: {count}")
        print("\nBy grade:")
        for grade, count in stats['grade_counts'].items():
            print(f"  Grade {grade}: {count}")
        
        return stats


def main():
    """Main function"""
    print("=" * 60)
    print("🍎 Fruit Grading System - Data Preprocessing")
    print("=" * 60)
    
    # Initialize preprocessor
    preprocessor = SimpleDataPreprocessor()
    
    # Scan dataset
    df = preprocessor.scan_dataset()
    
    if len(df) == 0:
        print("\n❌ No images found!")
        print("\nPlease ensure your images are in:")
        print("  ml/datasets/raw/apples/A/")
        print("  ml/datasets/raw/apples/B/")
        print("  ml/datasets/raw/apples/C/")
        print("  ml/datasets/raw/mangos/A/")
        print("  ml/datasets/raw/mangos/B/")
        print("  ml/datasets/raw/mangos/C/")
        print("  ml/datasets/raw/oranges/A/")
        print("  ml/datasets/raw/oranges/B/")
        print("  ml/datasets/raw/oranges/C/")
        return
    
    # Create augmented copies for small classes
    preprocessor.create_augmented_copies(df, target_per_class=15)
    
    # Rescan after augmentation
    print("\n📊 Rescanning after augmentation...")
    df = preprocessor.scan_dataset()
    
    # Split dataset
    train_df, val_df, test_df = preprocessor.split_dataset(df)
    
    if train_df is not None:
        # Generate EDA report
        preprocessor.generate_eda_report(df)
        
        print("\n" + "=" * 60)
        print("✅ Data preprocessing complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Run model training: python ml/train.py")
        print("2. Launch web app: python webapp/app.py")
    else:
        print("\n⚠ Not enough images to split. Add more images and run again.")


if __name__ == "__main__":
    main()
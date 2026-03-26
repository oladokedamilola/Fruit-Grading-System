"""
Data Preprocessing Module for Fruit Grading System
Handles image preprocessing, augmentation, and dataset splitting
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
from tqdm import tqdm
import random

class ImagePreprocessor:
    """Handles image preprocessing and augmentation"""
    
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        
        # Define augmentation pipeline
        self.augmentation_pipeline = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Resize(height=target_size[0], width=target_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Define validation/test preprocessing (no augmentation)
        self.preprocess_pipeline = A.Compose([
            A.Resize(height=target_size[0], width=target_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_image(self, image_path):
        """
        Load image from path
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def preprocess_single(self, image, augment=False):
        """
        Preprocess a single image
        """
        if image is None:
            return None
        
        # Convert to float32
        image = image.astype(np.float32)
        
        # Apply preprocessing pipeline
        if augment:
            processed = self.augmentation_pipeline(image=image)
        else:
            processed = self.preprocess_pipeline(image=image)
        
        return processed['image']
    
    def preprocess_batch(self, image_paths, augment=False, show_progress=True):
        """
        Preprocess a batch of images
        """
        images = []
        valid_paths = []
        
        iterator = tqdm(image_paths) if show_progress else image_paths
        for img_path in iterator:
            img = self.load_image(img_path)
            if img is not None:
                processed = self.preprocess_single(img, augment=augment)
                if processed is not None:
                    images.append(processed)
                    valid_paths.append(img_path)
        
        return np.array(images), valid_paths

class DatasetBuilder:
    """Builds and organizes the dataset for training"""
    
    def __init__(self, base_path="ml/datasets"):
        self.base_path = Path(base_path)
        self.raw_path = self.base_path / "raw"
        self.processed_path = self.base_path / "processed"
        self.annotations_path = self.base_path / "annotations"
        
        self.preprocessor = ImagePreprocessor()
        
        # Define fruit types and mapping
        self.fruit_types = ["apple", "mango", "orange", "tomato"]
        self.grades = ["A", "B", "C"]
        
        # Create mapping dictionaries
        self.fruit_to_idx = {fruit: idx for idx, fruit in enumerate(self.fruit_types)}
        self.idx_to_fruit = {idx: fruit for fruit, idx in self.fruit_to_idx.items()}
        self.grade_to_idx = {grade: idx for idx, grade in enumerate(self.grades)}
        self.idx_to_grade = {idx: grade for grade, idx in self.grade_to_idx.items()}
    
    def scan_dataset(self):
        """
        Scan the raw dataset and create a dataframe of all images
        """
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
        print(f"  - Fruits: {df['fruit_type'].value_counts().to_dict()}")
        print(f"  - Grades: {df['grade'].value_counts().to_dict()}")
        
        # Save dataset metadata
        df.to_csv(self.annotations_path / "dataset_metadata.csv", index=False)
        
        return df
    
    def split_dataset(self, df, test_size=0.15, val_size=0.15, random_state=42):
        """
        Split dataset into train, validation, and test sets
        """
        print("\n📊 Splitting dataset...")
        
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, 
            stratify=df[['fruit_type', 'grade']]
        )
        
        # Second split: separate validation from train
        val_ratio = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_ratio, random_state=random_state,
            stratify=train_val_df[['fruit_type', 'grade']]
        )
        
        print(f"✓ Training set: {len(train_df)} images")
        print(f"✓ Validation set: {len(val_df)} images")
        print(f"✓ Test set: {len(test_df)} images")
        
        # Save splits
        splits = {
            'train': train_df.to_dict('records'),
            'validation': val_df.to_dict('records'),
            'test': test_df.to_dict('records')
        }
        
        with open(self.annotations_path / "dataset_splits.json", 'w') as f:
            json.dump(splits, f, indent=2)
        
        # Also save as CSV for easy access
        train_df.to_csv(self.annotations_path / "train_split.csv", index=False)
        val_df.to_csv(self.annotations_path / "validation_split.csv", index=False)
        test_df.to_csv(self.annotations_path / "test_split.csv", index=False)
        
        return train_df, val_df, test_df
    
    def create_tf_dataset(self, df, batch_size=32, augment=False, shuffle=True):
        """
        Create TensorFlow dataset from dataframe
        """
        def load_and_preprocess(image_path, fruit_idx, grade_idx):
            # Load image
            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, [224, 224])
            image = tf.cast(image, tf.float32) / 255.0
            
            return image, {'fruit_type': fruit_idx, 'grade': grade_idx}
        
        # Create dataset from dataframe
        image_paths = df['image_path'].values
        fruit_labels = df['fruit_idx'].values
        grade_labels = df['grade_idx'].values
        
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, fruit_labels, grade_labels))
        dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset

class DataAugmentation:
    """Advanced data augmentation techniques"""
    
    def __init__(self):
        # Define augmentation strategies for different scenarios
        self.basic_augmentation = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.advanced_augmentation = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.defect_simulation = A.Compose([
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, p=0.5),
            A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.5, p=0.3),
            A.RandomRain(slant_lower=-10, slant_upper=10, drop_width=1, drop_height=1, p=0.3)
        ])
    
    def augment_image(self, image, strategy='basic'):
        """
        Apply augmentation to an image
        """
        if strategy == 'basic':
            augmented = self.basic_augmentation(image=image)
        elif strategy == 'advanced':
            augmented = self.advanced_augmentation(image=image)
        elif strategy == 'defect':
            augmented = self.defect_simulation(image=image)
        else:
            return image
        
        return augmented['image']
    
    def create_augmented_copies(self, df, output_dir, target_multiplier=2):
        """
        Create augmented copies of images to balance dataset
        """
        print(f"\n🔄 Creating augmented copies (target multiplier: {target_multiplier}x)...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        augmented_data = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            original_path = Path(row['image_path'])
            fruit = row['fruit_type']
            grade = row['grade']
            
            # Load original image
            image = cv2.imread(str(original_path))
            if image is None:
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create multiple augmented versions
            for i in range(target_multiplier - 1):  # -1 because original exists
                # Randomly select augmentation strategy
                strategy = random.choice(['basic', 'advanced', 'defect'])
                augmented = self.augment_image(image, strategy=strategy)
                
                # Convert back to BGR for saving
                augmented_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                
                # Save augmented image
                output_filename = f"{original_path.stem}_aug_{i}{original_path.suffix}"
                output_path = output_dir / fruit / grade / output_filename
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_path), augmented_bgr)
                
                augmented_data.append({
                    'original_path': str(original_path),
                    'augmented_path': str(output_path),
                    'fruit_type': fruit,
                    'grade': grade,
                    'augmentation_strategy': strategy
                })
        
        # Save augmentation metadata
        aug_df = pd.DataFrame(augmented_data)
        aug_df.to_csv(output_dir.parent / "annotations/augmentation_metadata.csv", index=False)
        
        print(f"✓ Created {len(augmented_data)} augmented images")
        return aug_df

def create_exploratory_analysis(df, output_dir="ml/outputs/eda"):
    """
    Create exploratory data analysis visualizations
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 1. Class distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Fruit distribution
    fruit_counts = df['fruit_type'].value_counts()
    axes[0].bar(fruit_counts.index, fruit_counts.values)
    axes[0].set_title('Fruit Type Distribution')
    axes[0].set_xlabel('Fruit Type')
    axes[0].set_ylabel('Count')
    
    # Grade distribution
    grade_counts = df['grade'].value_counts()
    axes[1].bar(grade_counts.index, grade_counts.values, color=['green', 'orange', 'red'])
    axes[1].set_title('Grade Distribution')
    axes[1].set_xlabel('Grade')
    axes[1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Fruit-Grade heatmap
    pivot_table = pd.crosstab(df['fruit_type'], df['grade'])
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, fmt='d', cmap='YlOrRd')
    plt.title('Fruit Type vs Grade Distribution')
    plt.tight_layout()
    plt.savefig(output_dir / "fruit_grade_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Sample images visualization
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    axes = axes.flatten()
    
    fruits = df['fruit_type'].unique()
    grades = df['grade'].unique()
    
    for idx, fruit in enumerate(fruits):
        for grade_idx, grade in enumerate(grades):
            sample = df[(df['fruit_type'] == fruit) & (df['grade'] == grade)].iloc[0]
            img = cv2.imread(sample['image_path'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            ax_idx = idx * len(grades) + grade_idx
            axes[ax_idx].imshow(img)
            axes[ax_idx].set_title(f'{fruit.title()} - Grade {grade}')
            axes[ax_idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "sample_images.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ EDA visualizations saved to {output_dir}")
    
    # Generate summary statistics
    summary = {
        'total_images': len(df),
        'fruit_types': len(fruits),
        'grades': len(grades),
        'fruit_distribution': fruit_counts.to_dict(),
        'grade_distribution': grade_counts.to_dict(),
        'fruit_grade_matrix': pivot_table.to_dict()
    }
    
    with open(output_dir / "eda_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary

def main():
    """
    Main function to run the data preparation pipeline
    """
    print("=" * 60)
    print("🍎 Phase 1: Data Collection & Preparation")
    print("=" * 60)
    
    # Initialize classes
    acquirer = DataAcquisition()
    organizer = DatasetOrganizer()
    builder = DatasetBuilder()
    
    # Step 1: Create directory structure for custom data
    print("\n" + "=" * 60)
    print("Step 1: Setting up data directories")
    print("=" * 60)
    acquirer.create_directory_structure_for_custom_data()
    acquirer.organize_public_datasets()
    
    # Step 2: Validate dataset structure
    print("\n" + "=" * 60)
    print("Step 2: Validating dataset")
    print("=" * 60)
    stats = organizer.validate_structure()
    organizer.create_annotation_template()
    
    # Step 3: Scan and create dataset metadata
    print("\n" + "=" * 60)
    print("Step 3: Building dataset metadata")
    print("=" * 60)
    df = builder.scan_dataset()
    
    if len(df) == 0:
        print("\n❌ No images found in dataset!")
        print("\nPlease add images to the following structure:")
        print("ml/datasets/raw/")
        for fruit in builder.fruit_types:
            for grade in builder.grades:
                print(f"  └── {fruit}/{grade}/")
        print("\nThen run this script again.")
        return
    
    # Step 4: Exploratory Data Analysis
    print("\n" + "=" * 60)
    print("Step 4: Exploratory Data Analysis")
    print("=" * 60)
    summary = create_exploratory_analysis(df)
    print(f"\n📊 Dataset Summary:")
    print(f"  - Total images: {summary['total_images']}")
    print(f"  - Fruit types: {summary['fruit_types']}")
    print(f"  - Grades: {summary['grades']}")
    
    # Step 5: Split dataset
    print("\n" + "=" * 60)
    print("Step 5: Splitting dataset")
    print("=" * 60)
    train_df, val_df, test_df = builder.split_dataset(df)
    
    # Step 6: Optional - Create augmented copies for imbalanced classes
    print("\n" + "=" * 60)
    print("Step 6: Checking class balance")
    print("=" * 60)
    
    # Check if any class has less than 500 images
    min_images = 500
    low_classes = []
    for fruit in builder.fruit_types:
        for grade in builder.grades:
            count = len(df[(df['fruit_type'] == fruit) & (df['grade'] == grade)])
            if count < min_images:
                low_classes.append(f"{fruit}/{grade}: {count} images")
    
    if low_classes:
        print(f"\n⚠ Some classes have less than {min_images} images:")
        for cls in low_classes:
            print(f"  - {cls}")
        
        print("\nConsider collecting more images or using augmentation.")
        
        # Ask user if they want to create augmented copies
        response = input("\nDo you want to create augmented copies to balance? (y/n): ")
        if response.lower() == 'y':
            augmenter = DataAugmentation()
            # Create augmented copies for low classes
            low_df = df[df.apply(lambda x: 
                len(df[(df['fruit_type'] == x['fruit_type']) & 
                       (df['grade'] == x['grade'])]) < min_images, axis=1)]
            aug_df = augmenter.create_augmented_copies(
                low_df, 
                builder.processed_path / "augmented",
                target_multiplier=2
            )
            print("✓ Augmented copies created")
    else:
        print(f"✓ All classes have at least {min_images} images")
    
    print("\n" + "=" * 60)
    print("✅ Phase 1 Complete!")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. Review EDA visualizations in ml/outputs/eda/")
    print("2. Check dataset splits in ml/datasets/annotations/")
    print("3. Proceed to Phase 2: Model Development")
    print("\nFiles created:")
    print("  - Dataset metadata: ml/datasets/annotations/dataset_metadata.csv")
    print("  - Train/Val/Test splits: ml/datasets/annotations/*_split.csv")
    print("  - EDA visualizations: ml/outputs/eda/")
    print("  - Grading guidelines: ml/datasets/annotations/grading_guidelines.md")

if __name__ == "__main__":
    main()
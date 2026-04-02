"""
Augment and create synthetic grades for fruit grading
Takes clean fruit images and creates Grade A, B, C variants
"""

import cv2
import numpy as np
from pathlib import Path
import random
import shutil
from tqdm import tqdm
import os

class FruitAugmenter:
    """Create augmented and graded fruit images"""
    
    def __init__(self, source_dir, target_dir, target_per_grade=100):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.target_per_grade = target_per_grade
        
        # Create target structure
        for fruit in self.get_fruits():
            for grade in ['A', 'B', 'C']:
                (self.target_dir / fruit / grade).mkdir(parents=True, exist_ok=True)
    
    def get_fruits(self):
        """Get list of fruits from source directory"""
        fruits = []
        for item in self.source_dir.iterdir():
            if item.is_dir():
                # Only include our chosen fruits
                if item.name in ['apples', 'bananas', 'mangos', 'oranges', 'pineapples']:
                    fruits.append(item.name)
        return sorted(fruits)
    
    def load_fruit_images(self, fruit):
        """Load all images for a specific fruit"""
        fruit_dir = self.source_dir / fruit
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
            images.extend(fruit_dir.glob(ext))
        return images
    
    def create_grade_a_copies(self, fruit, original_images):
        """Create multiple copies of Grade A images with slight variations"""
        grade_a_dir = self.target_dir / fruit / "A"
        
        # Clear existing
        for f in grade_a_dir.glob("*"):
            f.unlink()
        
        images_per_original = max(1, self.target_per_grade // len(original_images))
        
        print(f"  Creating Grade A copies...")
        for img_path in tqdm(original_images, desc="    Processing"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            for i in range(images_per_original):
                # Slight variations for Grade A
                varied = self._grade_a_variation(img)
                
                # Save with original name as base
                name = f"{img_path.stem}_A_{i:03d}{img_path.suffix}"
                cv2.imwrite(str(grade_a_dir / name), varied)
        
        final_count = len(list(grade_a_dir.glob("*")))
        print(f"    Grade A: {final_count} images")
        return final_count
    
    def create_grade_b_copies(self, fruit, grade_a_images):
        """Create Grade B images by adding minor defects"""
        grade_b_dir = self.target_dir / fruit / "B"
        grade_a_dir = self.target_dir / fruit / "A"
        
        # Clear existing
        for f in grade_b_dir.glob("*"):
            f.unlink()
        
        # Get Grade A images (already in target)
        if not grade_a_images:
            grade_a_images = list(grade_a_dir.glob("*"))
        
        images_needed = self.target_per_grade
        images_available = len(grade_a_images)
        
        if images_available == 0:
            print(f"    No Grade A images for {fruit}, skipping")
            return 0
        
        copies_per_image = max(1, images_needed // images_available)
        
        print(f"  Creating Grade B images (minor defects)...")
        for img_path in tqdm(grade_a_images[:images_needed], desc="    Processing"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            for i in range(copies_per_image):
                # Add minor defects
                degraded = self._add_minor_defects(img)
                
                name = f"{img_path.stem}_B_{i:03d}{img_path.suffix}"
                cv2.imwrite(str(grade_b_dir / name), degraded)
        
        final_count = len(list(grade_b_dir.glob("*")))
        print(f"    Grade B: {final_count} images")
        return final_count
    
    def create_grade_c_copies(self, fruit, grade_a_images):
        """Create Grade C images by adding major defects"""
        grade_c_dir = self.target_dir / fruit / "C"
        grade_a_dir = self.target_dir / fruit / "A"
        
        # Clear existing
        for f in grade_c_dir.glob("*"):
            f.unlink()
        
        # Get Grade A images
        if not grade_a_images:
            grade_a_images = list(grade_a_dir.glob("*"))
        
        images_needed = self.target_per_grade
        images_available = len(grade_a_images)
        
        if images_available == 0:
            print(f"    No Grade A images for {fruit}, skipping")
            return 0
        
        copies_per_image = max(1, images_needed // images_available)
        
        print(f"  Creating Grade C images (major defects)...")
        for img_path in tqdm(grade_a_images[:images_needed], desc="    Processing"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            for i in range(copies_per_image):
                # Add major defects
                degraded = self._add_major_defects(img)
                
                name = f"{img_path.stem}_C_{i:03d}{img_path.suffix}"
                cv2.imwrite(str(grade_c_dir / name), degraded)
        
        final_count = len(list(grade_c_dir.glob("*")))
        print(f"    Grade C: {final_count} images")
        return final_count
    
    def _grade_a_variation(self, img):
        """Slight variations for Grade A (minor brightness/rotation)"""
        varied = img.copy()
        
        # Random brightness (very slight)
        brightness = random.uniform(0.92, 1.08)
        varied = np.clip(varied * brightness, 0, 255).astype(np.uint8)
        
        # Random small rotation
        if random.random() > 0.5:
            angle = random.randint(-5, 5)
            h, w = varied.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            varied = cv2.warpAffine(varied, matrix, (w, h))
        
        return varied
    
    def _add_minor_defects(self, img):
        """Add minor defects for Grade B"""
        degraded = img.copy()
        h, w = degraded.shape[:2]
        
        # Add small spots (1-3)
        num_spots = random.randint(1, 3)
        for _ in range(num_spots):
            x = random.randint(15, w-15)
            y = random.randint(15, h-15)
            radius = random.randint(3, 8)
            cv2.circle(degraded, (x, y), radius, (70, 60, 55), -1)
        
        # Slight brightness reduction
        brightness = random.uniform(0.82, 0.92)
        degraded = np.clip(degraded * brightness, 0, 255).astype(np.uint8)
        
        # Random slight rotation
        if random.random() > 0.5:
            angle = random.randint(-8, 8)
            h, w = degraded.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            degraded = cv2.warpAffine(degraded, matrix, (w, h))
        
        return degraded
    
    def _add_major_defects(self, img):
        """Add major defects for Grade C"""
        degraded = img.copy()
        h, w = degraded.shape[:2]
        
        # Add large bruises (2-4)
        num_bruises = random.randint(2, 4)
        for _ in range(num_bruises):
            x = random.randint(20, w-20)
            y = random.randint(20, h-20)
            radius = random.randint(15, 35)
            cv2.circle(degraded, (x, y), radius, (55, 48, 42), -1)
        
        # Add scratches
        for _ in range(random.randint(2, 5)):
            x1 = random.randint(10, w-10)
            y1 = random.randint(10, h-10)
            x2 = x1 + random.randint(-40, 40)
            y2 = y1 + random.randint(-40, 40)
            cv2.line(degraded, (x1, y1), (x2, y2), (50, 45, 40), random.randint(3, 7))
        
        # Darken significantly
        brightness = random.uniform(0.55, 0.75)
        degraded = np.clip(degraded * brightness, 0, 255).astype(np.uint8)
        
        # Add noise
        noise = np.random.randint(0, 40, degraded.shape, dtype=np.uint8)
        degraded = np.clip(degraded + noise, 0, 255).astype(np.uint8)
        
        # Random rotation
        angle = random.randint(-15, 15)
        h, w = degraded.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        degraded = cv2.warpAffine(degraded, matrix, (w, h))
        
        return degraded
    
    def process_all_fruits(self):
        """Process all fruits in the source directory"""
        fruits = self.get_fruits()
        
        print("=" * 60)
        print("🍎 Fruit Augmentation and Grading")
        print("=" * 60)
        print(f"Source: {self.source_dir}")
        print(f"Target: {self.target_dir}")
        print(f"Target per grade: {self.target_per_grade} images")
        print(f"Fruits to process: {', '.join(fruits)}")
        print()
        
        results = {}
        
        for fruit in fruits:
            print(f"\n{'='*50}")
            print(f"Processing: {fruit.upper()}")
            print(f"{'='*50}")
            
            # Load original images
            original_images = self.load_fruit_images(fruit)
            print(f"Original images: {len(original_images)}")
            
            if len(original_images) == 0:
                print(f"⚠ No images found for {fruit}, skipping")
                continue
            
            # Step 1: Create Grade A copies with variations
            grade_a_count = self.create_grade_a_copies(fruit, original_images)
            
            # Step 2: Create Grade B from Grade A
            grade_a_path = self.target_dir / fruit / "A"
            grade_a_images = list(grade_a_path.glob("*"))
            grade_b_count = self.create_grade_b_copies(fruit, grade_a_images)
            
            # Step 3: Create Grade C from Grade A
            grade_c_count = self.create_grade_c_copies(fruit, grade_a_images)
            
            results[fruit] = {
                'original': len(original_images),
                'grade_a': grade_a_count,
                'grade_b': grade_b_count,
                'grade_c': grade_c_count,
                'total': grade_a_count + grade_b_count + grade_c_count
            }
        
        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        grand_total = 0
        for fruit, stats in results.items():
            print(f"\n{fruit.upper()}:")
            print(f"  Original: {stats['original']}")
            print(f"  Grade A: {stats['grade_a']}")
            print(f"  Grade B: {stats['grade_b']}")
            print(f"  Grade C: {stats['grade_c']}")
            print(f"  Total: {stats['total']}")
            grand_total += stats['total']
        
        print(f"\n{'='*60}")
        print(f"GRAND TOTAL: {grand_total} images")
        print(f"{'='*60}")
        
        return results


def main():
    """Main function"""
    # Configure paths
    source_dir = "C:/Users/PC/Desktop/Fruit Grading Project/datasets"
    target_dir = "ml/datasets/raw"
    
    # Check if source exists
    if not Path(source_dir).exists():
        print(f"❌ Source directory not found: {source_dir}")
        print("Please check the path to your FIDS30 dataset")
        return
    
    print(f"✅ Source directory found: {source_dir}")
    
    # Create target directory
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    
    # Create augmenter with target 100 per grade
    augmenter = FruitAugmenter(
        source_dir=source_dir,
        target_dir=target_dir,
        target_per_grade=100
    )
    
    # Process all fruits
    results = augmenter.process_all_fruits()
    
    print("\n✅ Done!")
    print("\nNext steps:")
    print("1. Run data preprocessing: python ml/src/data_preprocessing_simple.py")
    print("2. Train the model: python ml/train_simple_cnn.py")


if __name__ == "__main__":
    main()
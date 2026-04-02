"""
Augment fruits to reach exact target per grade
Ensures balanced dataset with 100 images per grade per fruit
"""

import cv2
import numpy as np
from pathlib import Path
import random
import shutil
from tqdm import tqdm
import math

class BalancedAugmenter:
    """Create exactly target number of images per grade"""
    
    def __init__(self, source_dir, target_dir, target_per_grade=100):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.target_per_grade = target_per_grade
        
        # Clear and create target structure
        for fruit in self.get_fruits():
            for grade in ['A', 'B', 'C']:
                grade_dir = self.target_dir / fruit / grade
                if grade_dir.exists():
                    shutil.rmtree(grade_dir)
                grade_dir.mkdir(parents=True, exist_ok=True)
    
    def get_fruits(self):
        """Get list of fruits from source directory"""
        fruits = []
        for item in self.source_dir.iterdir():
            if item.is_dir() and item.name in ['apples', 'bananas', 'mangos', 'oranges', 'pineapples']:
                fruits.append(item.name)
        return sorted(fruits)
    
    def load_fruit_images(self, fruit):
        """Load all images for a specific fruit"""
        fruit_dir = self.source_dir / fruit
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
            images.extend(fruit_dir.glob(ext))
        return images
    
    def augment_to_target(self, original_images, target_count, grade_type='A'):
        """Augment images to reach exact target count"""
        if len(original_images) == 0:
            return []
        
        generated = []
        needed = target_count
        original_count = len(original_images)
        
        # How many copies per original needed
        copies_per_original = math.ceil(needed / original_count)
        
        for idx, img_path in enumerate(original_images):
            if needed <= 0:
                break
                
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Determine how many copies from this image
            copies_for_this = min(copies_per_original, needed)
            
            for i in range(copies_for_this):
                if grade_type == 'A':
                    augmented = self._grade_a_variation(img)
                elif grade_type == 'B':
                    augmented = self._add_minor_defects(img)
                else:  # Grade C
                    augmented = self._add_major_defects(img)
                
                # Unique filename
                name = f"{img_path.stem}_{grade_type}_{idx:03d}_{i:03d}{img_path.suffix}"
                generated.append((augmented, name))
                needed -= 1
                
                if needed <= 0:
                    break
        
        return generated
    
    def process_fruit(self, fruit, original_images):
        """Process a single fruit to create balanced grades"""
        print(f"\n{'='*50}")
        print(f"Processing: {fruit.upper()}")
        print(f"{'='*50}")
        print(f"Original images: {len(original_images)}")
        print(f"Target per grade: {self.target_per_grade}")
        
        # Create Grade A (clean with variations)
        print(f"\n  Creating Grade A ({self.target_per_grade} images)...")
        grade_a_images = self.augment_to_target(
            original_images, 
            self.target_per_grade, 
            grade_type='A'
        )
        
        grade_a_dir = self.target_dir / fruit / "A"
        for img, name in tqdm(grade_a_images, desc="    Saving Grade A"):
            cv2.imwrite(str(grade_a_dir / name), img)
        
        grade_a_count = len(grade_a_images)
        print(f"    Grade A: {grade_a_count} images")
        
        # Use Grade A images as base for B and C (better quality)
        # First, collect the generated Grade A images
        base_images = list(grade_a_dir.glob("*"))
        
        # Create Grade B
        print(f"\n  Creating Grade B ({self.target_per_grade} images)...")
        grade_b_images = self.augment_to_target(
            base_images, 
            self.target_per_grade, 
            grade_type='B'
        )
        
        grade_b_dir = self.target_dir / fruit / "B"
        for img, name in tqdm(grade_b_images, desc="    Saving Grade B"):
            cv2.imwrite(str(grade_b_dir / name), img)
        
        grade_b_count = len(grade_b_images)
        print(f"    Grade B: {grade_b_count} images")
        
        # Create Grade C
        print(f"\n  Creating Grade C ({self.target_per_grade} images)...")
        grade_c_images = self.augment_to_target(
            base_images, 
            self.target_per_grade, 
            grade_type='C'
        )
        
        grade_c_dir = self.target_dir / fruit / "C"
        for img, name in tqdm(grade_c_images, desc="    Saving Grade C"):
            cv2.imwrite(str(grade_c_dir / name), img)
        
        grade_c_count = len(grade_c_images)
        print(f"    Grade C: {grade_c_count} images")
        
        return {
            'original': len(original_images),
            'grade_a': grade_a_count,
            'grade_b': grade_b_count,
            'grade_c': grade_c_count,
            'total': grade_a_count + grade_b_count + grade_c_count
        }
    
    def _grade_a_variation(self, img):
        """Slight variations for Grade A"""
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
        
        # Small horizontal flip
        if random.random() > 0.5:
            varied = cv2.flip(varied, 1)
        
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
        """Process all fruits"""
        fruits = self.get_fruits()
        
        print("=" * 60)
        print("🍎 Balanced Fruit Augmentation")
        print("=" * 60)
        print(f"Source: {self.source_dir}")
        print(f"Target: {self.target_dir}")
        print(f"Target per grade: {self.target_per_grade} images")
        print(f"Fruits: {', '.join(fruits)}")
        
        results = {}
        
        for fruit in fruits:
            original_images = self.load_fruit_images(fruit)
            results[fruit] = self.process_fruit(fruit, original_images)
        
        # Summary
        print("\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)
        
        grand_total = 0
        for fruit, stats in results.items():
            print(f"\n{fruit.upper()}:")
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
    source_dir = "C:/Users/PC/Desktop/Fruit Grading Project/datasets"
    target_dir = "ml/datasets/raw"
    
    if not Path(source_dir).exists():
        print(f"❌ Source not found: {source_dir}")
        return
    
    print(f"✅ Source found: {source_dir}")
    
    # Clear existing target to start fresh
    target_path = Path(target_dir)
    if target_path.exists():
        print(f"⚠ Clearing existing target directory...")
        shutil.rmtree(target_path)
    
    # Create augmenter with target 100 per grade
    augmenter = BalancedAugmenter(
        source_dir=source_dir,
        target_dir=target_dir,
        target_per_grade=100
    )
    
    results = augmenter.process_all_fruits()
    
    print("\n✅ Done!")
    print("\nNext steps:")
    print("1. Run data preprocessing: python ml/src/data_preprocessing_simple.py")
    print("2. Train the model: python ml/train_simple_cnn.py")


if __name__ == "__main__":
    main()
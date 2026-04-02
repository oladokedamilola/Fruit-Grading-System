"""
Create Grade B and C images from perfect Grade A images
by adding synthetic defects and imperfections
"""

import cv2
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm

def add_defect(image, severity='medium'):
    """Add synthetic defects to an image"""
    img = image.copy()
    h, w = img.shape[:2]
    
    if severity == 'light':  # Grade B - minor imperfections
        # Add small spots
        num_spots = random.randint(1, 3)
        for _ in range(num_spots):
            x = random.randint(10, w-10)
            y = random.randint(10, h-10)
            radius = random.randint(3, 8)
            cv2.circle(img, (x, y), radius, (80, 70, 60), -1)
        
        # Slight color variation
        brightness = random.uniform(0.85, 0.95)
        img = np.clip(img * brightness, 0, 255).astype(np.uint8)
        
    elif severity == 'heavy':  # Grade C - significant defects
        # Add large bruises
        num_bruises = random.randint(2, 4)
        for _ in range(num_bruises):
            x = random.randint(20, w-20)
            y = random.randint(20, h-20)
            radius = random.randint(15, 30)
            cv2.circle(img, (x, y), radius, (60, 50, 45), -1)
        
        # Add scratches
        for _ in range(random.randint(2, 5)):
            x1 = random.randint(10, w-10)
            y1 = random.randint(10, h-10)
            x2 = x1 + random.randint(-30, 30)
            y2 = y1 + random.randint(-30, 30)
            cv2.line(img, (x1, y1), (x2, y2), (50, 45, 40), random.randint(2, 5))
        
        # Darken overall
        brightness = random.uniform(0.6, 0.8)
        img = np.clip(img * brightness, 0, 255).astype(np.uint8)
        
        # Add noise
        noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    return img

def create_synthetic_grades(source_dir, target_dir, num_per_grade=100):
    """Create synthetic Grade B and C from Grade A images"""
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Get all Grade A images
    grade_a_images = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))
    
    if len(grade_a_images) == 0:
        print(f"⚠ No images found in {source_dir}")
        return
    
    print(f"Found {len(grade_a_images)} Grade A images")
    
    # Create Grade B
    grade_b_path = target_path / "B"
    grade_b_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating Grade B images (light defects)...")
    for i in tqdm(range(min(num_per_grade, len(grade_a_images) * 2))):
        src_img = grade_a_images[i % len(grade_a_images)]
        img = cv2.imread(str(src_img))
        if img is not None:
            degraded = add_defect(img, severity='light')
            dest_name = f"grade_B_{i:04d}.jpg"
            cv2.imwrite(str(grade_b_path / dest_name), degraded)
    
    # Create Grade C
    grade_c_path = target_path / "C"
    grade_c_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating Grade C images (heavy defects)...")
    for i in tqdm(range(min(num_per_grade, len(grade_a_images) * 2))):
        src_img = grade_a_images[i % len(grade_a_images)]
        img = cv2.imread(str(src_img))
        if img is not None:
            degraded = add_defect(img, severity='heavy')
            dest_name = f"grade_C_{i:04d}.jpg"
            cv2.imwrite(str(grade_c_path / dest_name), degraded)
    
    print(f"✓ Created synthetic images:")
    print(f"  Grade B: {len(list(grade_b_path.glob('*.jpg')))} images")
    print(f"  Grade C: {len(list(grade_c_path.glob('*.jpg')))} images")

def main():
    """Create synthetic grades for all fruits"""
    
    base_path = Path("ml/datasets/raw")
    fruits = ["apples", "mangos", "oranges", "tomatoes"]
    
    for fruit in fruits:
        grade_a_path = base_path / fruit / "A"
        if grade_a_path.exists():
            print(f"\n{'='*50}")
            print(f"Processing {fruit.upper()}")
            print(f"{'='*50}")
            create_synthetic_grades(
                source_dir=grade_a_path,
                target_dir=base_path / fruit,
                num_per_grade=150  # Create 150 each of B and C
            )
        else:
            print(f"⚠ No Grade A images for {fruit}, skipping...")

if __name__ == "__main__":
    main()
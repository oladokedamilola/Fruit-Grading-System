"""
Custom Image Collection Helper
Provides utilities for capturing and organizing custom fruit images
"""

import cv2
import os
from pathlib import Path
import datetime
import json

class ImageCollector:
    """Helper class for collecting custom fruit images"""
    
    def __init__(self, base_path="ml/datasets/raw"):
        self.base_path = Path(base_path)
        self.fruit_types = ["apple", "mango", "orange", "tomato"]
        self.grades = ["A", "B", "C"]
        
        # Create directories
        for fruit in self.fruit_types:
            for grade in self.grades:
                (self.base_path / fruit / grade).mkdir(parents=True, exist_ok=True)
    
    def collect_from_webcam(self, fruit, grade, num_images=50):
        """
        Collect images using webcam
        """
        save_dir = self.base_path / fruit / grade
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print(f"\n📸 Collecting images for {fruit} - Grade {grade}")
        print(f"Target: {num_images} images")
        print("Press SPACE to capture, ESC to quit")
        
        count = 0
        while count < num_images:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Show instructions on frame
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Fruit: {fruit} - Grade: {grade}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Images captured: {count}/{num_images}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press SPACE to capture, ESC to quit", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imshow('Image Collection', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                # Save image
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{fruit}_{grade}_{timestamp}.jpg"
                filepath = save_dir / filename
                cv2.imwrite(str(filepath), frame)
                count += 1
                print(f"✓ Captured {count}/{num_images}: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n✅ Collection complete! Saved {count} images to {save_dir}")
    
    def collect_from_files(self, fruit, grade, source_folder, target_count=None):
        """
        Copy images from existing folder to dataset structure
        """
        source_folder = Path(source_folder)
        dest_folder = self.base_path / fruit / grade
        dest_folder.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        images = []
        for ext in image_extensions:
            images.extend(source_folder.glob(ext))
        
        if target_count:
            images = images[:target_count]
        
        print(f"\n📁 Copying images for {fruit} - Grade {grade}")
        print(f"Found {len(images)} images")
        
        copied = 0
        for img_path in images:
            dest_path = dest_folder / img_path.name
            if not dest_path.exists():
                import shutil
                shutil.copy2(img_path, dest_path)
                copied += 1
        
        print(f"✓ Copied {copied} images to {dest_folder}")
        return copied
    
    def create_collection_log(self):
        """
        Create a log of all collected images
        """
        log_data = {
            "collection_date": str(datetime.datetime.now()),
            "images": []
        }
        
        for fruit in self.fruit_types:
            for grade in self.grades:
                grade_path = self.base_path / fruit / grade
                if grade_path.exists():
                    images = list(grade_path.glob("*.jpg")) + list(grade_path.glob("*.png"))
                    for img in images:
                        log_data["images"].append({
                            "path": str(img),
                            "fruit": fruit,
                            "grade": grade,
                            "filename": img.name,
                            "size": img.stat().st_size
                        })
        
        log_path = self.base_path.parent / "annotations/collection_log.json"
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"✓ Collection log saved to {log_path}")
        return log_data

def main():
    """Interactive collection helper"""
    print("=" * 60)
    print("📸 Custom Image Collection Helper")
    print("=" * 60)
    
    collector = ImageCollector()
    
    while True:
        print("\nOptions:")
        print("1. Collect images from webcam")
        print("2. Copy images from existing folder")
        print("3. View collection statistics")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ")
        
        if choice == '1':
            print("\nSelect fruit type:")
            for i, fruit in enumerate(collector.fruit_types, 1):
                print(f"{i}. {fruit}")
            fruit_choice = int(input("Enter number: ")) - 1
            fruit = collector.fruit_types[fruit_choice]
            
            print("\nSelect grade:")
            for i, grade in enumerate(collector.grades, 1):
                print(f"{i}. Grade {grade}")
            grade_choice = int(input("Enter number: ")) - 1
            grade = collector.grades[grade_choice]
            
            num_images = int(input("Number of images to capture (default 50): ") or 50)
            collector.collect_from_webcam(fruit, grade, num_images)
        
        elif choice == '2':
            print("\nSelect fruit type:")
            for i, fruit in enumerate(collector.fruit_types, 1):
                print(f"{i}. {fruit}")
            fruit_choice = int(input("Enter number: ")) - 1
            fruit = collector.fruit_types[fruit_choice]
            
            print("\nSelect grade:")
            for i, grade in enumerate(collector.grades, 1):
                print(f"{i}. Grade {grade}")
            grade_choice = int(input("Enter number: ")) - 1
            grade = collector.grades[grade_choice]
            
            source = input("Source folder path: ")
            target_count = input("Target count (optional): ")
            target_count = int(target_count) if target_count else None
            
            collector.collect_from_files(fruit, grade, source, target_count)
        
        elif choice == '3':
            log = collector.create_collection_log()
            print(f"\n📊 Collection Statistics:")
            print(f"Total images: {len(log['images'])}")
            
            # Count by fruit and grade
            from collections import Counter
            fruit_counts = Counter([img['fruit'] for img in log['images']])
            grade_counts = Counter([img['grade'] for img in log['images']])
            
            print("\nBy fruit:")
            for fruit, count in fruit_counts.items():
                print(f"  {fruit}: {count}")
            
            print("\nBy grade:")
            for grade, count in grade_counts.items():
                print(f"  Grade {grade}: {count}")
        
        elif choice == '4':
            print("\nGoodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
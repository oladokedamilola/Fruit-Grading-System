"""
Interactive Image Grading Tool for Fruit Quality Classification
Manually sort images into Grade A, B, C categories
"""

import os
import shutil
from pathlib import Path
import cv2
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import sys

class ImageGrader:
    """Interactive tool to manually grade fruit images"""
    
    def __init__(self, source_dir, dest_base):
        self.source_dir = Path(source_dir)
        self.dest_base = Path(dest_base)
        self.images = []
        self.current_idx = 0
        self.fruit_name = None
        self.graded_count = 0
        self.window = None
        self.image_label = None
        self.info_label = None
        self.progress_bar = None
        
    def load_images(self, fruit):
        """Load all images for a specific fruit"""
        fruit_dir = self.source_dir / fruit
        if not fruit_dir.exists():
            print(f"⚠ Directory not found: {fruit_dir}")
            return False
            
        self.images = list(fruit_dir.glob("*.jpg")) + list(fruit_dir.glob("*.png")) + \
                      list(fruit_dir.glob("*.jpeg")) + list(fruit_dir.glob("*.JPG"))
        self.images.sort()  # Sort for consistent ordering
        self.current_idx = 0
        self.graded_count = 0
        self.fruit_name = fruit
        
        print(f"✓ Loaded {len(self.images)} images for {fruit}")
        return True
        
    def create_grading_gui(self):
        """Create GUI for manual grading"""
        if not self.images:
            print(f"✗ No images loaded for {self.fruit_name}")
            return
            
        # Create window
        self.window = tk.Tk()
        self.window.title(f"Grade {self.fruit_name.upper()} Images - Fruit Grading System")
        self.window.geometry("900x700")
        
        # Instructions
        instr_frame = tk.Frame(self.window, bg='#f0f0f0', pady=10)
        instr_frame.pack(fill='x')
        
        tk.Label(instr_frame, text="Fruit Grading Instructions", 
                 font=("Arial", 14, "bold"), bg='#f0f0f0').pack()
        tk.Label(instr_frame, text="Grade A: Excellent - No defects, perfect shape, uniform color", 
                 fg='#2ecc71', bg='#f0f0f0').pack()
        tk.Label(instr_frame, text="Grade B: Good - Minor scratches, small spots, slight color variation", 
                 fg='#f39c12', bg='#f0f0f0').pack()
        tk.Label(instr_frame, text="Grade C: Poor - Large bruises, significant defects, poor shape", 
                 fg='#e74c3c', bg='#f0f0f0').pack()
        
        # Image display
        self.image_label = tk.Label(self.window, bg='white', relief='sunken', bd=2)
        self.image_label.pack(pady=20, padx=20)
        
        # Info and progress
        info_frame = tk.Frame(self.window)
        info_frame.pack(pady=10)
        
        self.info_label = tk.Label(info_frame, text="", font=("Arial", 12))
        self.info_label.pack()
        
        self.progress_bar = ttk.Progressbar(self.window, length=400, mode='determinate')
        self.progress_bar.pack(pady=10)
        
        # Buttons
        btn_frame = tk.Frame(self.window)
        btn_frame.pack(pady=20)
        
        btn_a = tk.Button(btn_frame, text="GRADE A - EXCELLENT", 
                          command=lambda: self.grade_image('A'), 
                          bg='#2ecc71', fg='white', font=("Arial", 12, "bold"),
                          width=20, height=2)
        btn_a.pack(side=tk.LEFT, padx=10)
        
        btn_b = tk.Button(btn_frame, text="GRADE B - GOOD", 
                          command=lambda: self.grade_image('B'), 
                          bg='#f39c12', fg='white', font=("Arial", 12, "bold"),
                          width=20, height=2)
        btn_b.pack(side=tk.LEFT, padx=10)
        
        btn_c = tk.Button(btn_frame, text="GRADE C - POOR", 
                          command=lambda: self.grade_image('C'), 
                          bg='#e74c3c', fg='white', font=("Arial", 12, "bold"),
                          width=20, height=2)
        btn_c.pack(side=tk.LEFT, padx=10)
        
        # Navigation buttons
        nav_frame = tk.Frame(self.window)
        nav_frame.pack(pady=10)
        
        tk.Button(nav_frame, text="← Previous", command=self.previous_image, width=12).pack(side=tk.LEFT, padx=5)
        tk.Button(nav_frame, text="Skip / Undo", command=self.skip_image, width=12).pack(side=tk.LEFT, padx=5)
        tk.Button(nav_frame, text="Skip to End", command=self.skip_to_end, width=12).pack(side=tk.LEFT, padx=5)
        
        # Keyboard shortcuts
        self.window.bind('<a>', lambda e: self.grade_image('A'))
        self.window.bind('<b>', lambda e: self.grade_image('B'))
        self.window.bind('<c>', lambda e: self.grade_image('C'))
        self.window.bind('<Left>', lambda e: self.previous_image())
        self.window.bind('<Right>', lambda e: self.next_image())
        
        # Show first image
        self.show_current_image()
        
        self.window.mainloop()
    
    def show_current_image(self):
        """Display current image"""
        if self.current_idx >= len(self.images):
            self.complete_grading()
            return
            
        img_path = self.images[self.current_idx]
        
        # Load and display image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠ Could not load: {img_path}")
            self.next_image()
            return
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Calculate resize dimensions while maintaining aspect ratio
        max_size = 500
        h, w = img.shape[:2]
        if h > max_size or w > max_size:
            if h > w:
                new_h = max_size
                new_w = int(w * max_size / h)
            else:
                new_w = max_size
                new_h = int(h * max_size / w)
            img = cv2.resize(img, (new_w, new_h))
        
        img = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(img)
        
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk
        
        # Update info
        filename = img_path.name
        remaining = len(self.images) - self.current_idx
        self.info_label.config(
            text=f"Image {self.current_idx + 1}/{len(self.images)} | "
                 f"Graded: {self.graded_count} | "
                 f"Remaining: {remaining}\n{filename}"
        )
        
        # Update progress bar
        self.progress_bar['value'] = (self.current_idx / len(self.images)) * 100
        
    def grade_image(self, grade):
        """Grade current image and move to next"""
        if self.current_idx >= len(self.images):
            return
            
        src = self.images[self.current_idx]
        dest_dir = self.dest_base / self.fruit_name / grade
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        
        # Copy if not already graded
        if not dest.exists():
            shutil.copy2(src, dest)
            self.graded_count += 1
            print(f"✓ [{self.graded_count}] Graded as {grade}: {src.name}")
        
        # Move to next image
        self.next_image()
    
    def skip_image(self):
        """Skip current image (keep for later)"""
        if self.current_idx < len(self.images):
            print(f"⏭ Skipped: {self.images[self.current_idx].name}")
            self.next_image()
    
    def next_image(self):
        """Move to next image"""
        self.current_idx += 1
        if self.current_idx < len(self.images):
            self.show_current_image()
        else:
            self.complete_grading()
    
    def previous_image(self):
        """Go to previous image"""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.show_current_image()
    
    def skip_to_end(self):
        """Skip remaining images"""
        result = messagebox.askyesno("Skip Remaining", 
                                      f"Skip remaining {len(self.images) - self.current_idx} images?")
        if result:
            self.current_idx = len(self.images)
            self.complete_grading()
    
    def complete_grading(self):
        """Complete grading for current fruit"""
        print(f"\n{'='*50}")
        print(f"✅ Completed grading for {self.fruit_name}")
        print(f"   Total: {len(self.images)} images")
        print(f"   Graded: {self.graded_count}")
        print(f"   Skipped: {len(self.images) - self.graded_count}")
        print(f"{'='*50}\n")
        
        # Show summary of graded files
        for grade in ['A', 'B', 'C']:
            grade_dir = self.dest_base / self.fruit_name / grade
            if grade_dir.exists():
                count = len(list(grade_dir.glob("*")))
                print(f"   Grade {grade}: {count} images")
        
        messagebox.showinfo("Fruit Grading Complete", 
                           f"Completed grading {self.fruit_name}!\n"
                           f"Graded: {self.graded_count} images\n"
                           f"Skipped: {len(self.images) - self.graded_count}")
        
        self.window.destroy()


def main():
    """Main function to grade all fruits"""
    print("=" * 60)
    print("🍎 Fruit Grading System - Manual Image Grader")
    print("=" * 60)
    print("\nInstructions:")
    print("  - Press 'A' key for Grade A (Excellent)")
    print("  - Press 'B' key for Grade B (Good)")
    print("  - Press 'C' key for Grade C (Poor)")
    print("  - Press '←' for previous image")
    print("  - Press '→' for next image")
    print("  - Or click the buttons")
    print("\n" + "=" * 60)
    
    # Define fruits to grade
    fruits_to_grade = ["apples", "mangos", "oranges"]
    
    # Create grader
    grader = ImageGrader(
        source_dir="FIDS30",
        dest_base="ml/datasets/raw"
    )
    
    # Grade each fruit
    for fruit in fruits_to_grade:
        if grader.load_images(fruit):
            input(f"\nPress Enter to start grading {fruit.upper()}...")
            grader.create_grading_gui()
        else:
            print(f"⚠ Could not find {fruit} in FIDS30 folder")
    
    print("\n" + "=" * 60)
    print("✅ All fruits graded!")
    print("=" * 60)
    print("\nNext step: Run data preprocessing")
    print("python ml/src/data_preprocessing.py")


if __name__ == "__main__":
    main()
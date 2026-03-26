"""
Data Acquisition Module for Fruit Grading System
Handles downloading public datasets and organizing custom images
"""

import os
import requests
import zipfile
import shutil
from pathlib import Path
import json
import kaggle
from tqdm import tqdm
import pandas as pd

class DataAcquisition:
    """Handles downloading and organizing datasets"""
    
    def __init__(self, base_path="ml/datasets"):
        self.base_path = Path(base_path)
        self.raw_path = self.base_path / "raw"
        self.processed_path = self.base_path / "processed"
        self.annotations_path = self.base_path / "annotations"
        
        # Create directories if they don't exist
        for path in [self.raw_path, self.processed_path, self.annotations_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def download_from_kaggle(self, dataset_name, destination_folder):
        """
        Download dataset from Kaggle
        Requires Kaggle API configured
        """
        try:
            print(f"Downloading {dataset_name} from Kaggle...")
            # This requires kaggle API: pip install kaggle
            # And kaggle.json in ~/.kaggle/
            kaggle.api.dataset_download_files(
                dataset_name, 
                path=destination_folder, 
                unzip=True
            )
            print(f"✓ Downloaded {dataset_name}")
            return True
        except Exception as e:
            print(f"✗ Error downloading from Kaggle: {e}")
            return False
    
    def download_from_url(self, url, destination):
        """
        Download dataset from direct URL
        """
        try:
            print(f"Downloading from {url}...")
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(destination, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            # Extract if it's a zip file
            if destination.endswith('.zip'):
                with zipfile.ZipFile(destination, 'r') as zip_ref:
                    zip_ref.extractall(destination.parent)
                os.remove(destination)
            
            print(f"✓ Downloaded and extracted to {destination.parent}")
            return True
        except Exception as e:
            print(f"✗ Error downloading: {e}")
            return False
    
    def organize_public_datasets(self):
        """
        Organize downloaded public datasets into standardized structure
        """
        print("\n📁 Organizing public datasets...")
        
        # Define public dataset sources (you'll need to download these manually)
        public_sources = {
            "Fruit-360": {
                "source": "https://www.kaggle.com/datasets/moltean/fruits",
                "description": "Fruits 360 dataset with 131 fruits and vegetables"
            },
            "MangoDB": {
                "source": "https://www.kaggle.com/datasets/arijittingshinde/mangodb",
                "description": "Mango variety and quality dataset"
            },
            "Apple2Orange": {
                "source": "https://www.kaggle.com/datasets/utkarshsaxenadn/Apple2Orange",
                "description": "Apple and orange classification dataset"
            }
        }
        
        # Create a metadata file for public datasets
        metadata = {
            "datasets": public_sources,
            "download_instructions": "Download these datasets manually from Kaggle and place them in ml/datasets/raw/"
        }
        
        with open(self.annotations_path / "public_datasets_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("✓ Created public datasets metadata")
        print("⚠ Please download the datasets manually and place them in ml/datasets/raw/")
    
    def create_directory_structure_for_custom_data(self):
        """
        Create directory structure for custom collected images
        """
        print("\n📁 Creating custom data directory structure...")
        
        fruit_types = ["apple", "mango", "orange", "tomato"]
        grades = ["A", "B", "C"]
        
        for fruit in fruit_types:
            for grade in grades:
                grade_dir = self.raw_path / fruit / grade
                grade_dir.mkdir(parents=True, exist_ok=True)
                print(f"✓ Created: {grade_dir}")
        
        # Create README with collection instructions
        readme_content = """
# Custom Image Collection Instructions

## Setup
1. Place images in the appropriate fruit/grade folders
2. Use clear naming convention: fruit_grade_number.jpg (e.g., apple_A_001.jpg)
3. Ensure images are in JPG or PNG format

## Collection Guidelines
- Capture images in different lighting conditions (natural, artificial)
- Use various backgrounds (white, dark, natural)
- Capture from multiple angles (top, side, bottom)
- Include both whole fruits and cross-sections
- Ensure fruits are clean and well-lit

## Quality Requirements
- Minimum 500 images per fruit per grade
- Resolution: at least 640x480 pixels
- Focus: Sharp, clear images
- Format: JPG or PNG

## Grade Definitions
- Grade A: Excellent quality - No defects, uniform color, perfect shape
- Grade B: Good quality - Minor imperfections, slight color variations
- Grade C: Poor quality - Significant defects, discoloration, irregular shape
"""
        
        with open(self.raw_path / "COLLECTION_INSTRUCTIONS.md", 'w') as f:
            f.write(readme_content)
        
        print("✓ Created collection instructions")

class DatasetOrganizer:
    """Organizes and validates the dataset structure"""
    
    def __init__(self, base_path="ml/datasets"):
        self.base_path = Path(base_path)
        self.raw_path = self.base_path / "raw"
    
    def validate_structure(self):
        """
        Validate that the dataset has the correct structure
        """
        print("\n🔍 Validating dataset structure...")
        
        fruit_types = ["apple", "mango", "orange", "tomato"]
        grades = ["A", "B", "C"]
        
        stats = {}
        
        for fruit in fruit_types:
            stats[fruit] = {}
            for grade in grades:
                grade_path = self.raw_path / fruit / grade
                if grade_path.exists():
                    images = list(grade_path.glob("*.jpg")) + list(grade_path.glob("*.png"))
                    stats[fruit][grade] = len(images)
                    print(f"✓ {fruit}/{grade}: {len(images)} images")
                else:
                    stats[fruit][grade] = 0
                    print(f"⚠ {fruit}/{grade}: Directory not found")
        
        # Create statistics report
        report = {
            "validation_date": str(pd.Timestamp.now()),
            "statistics": stats,
            "total_images": sum(sum(grade.values()) for grade in stats.values())
        }
        
        with open(self.base_path / "annotations/dataset_stats.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Total images: {report['total_images']}")
        return stats
    
    def create_annotation_template(self):
        """
        Create CSV template for manual annotations
        """
        import pandas as pd
        
        # Create annotation template
        template = pd.DataFrame(columns=[
            'image_path', 'fruit_type', 'grade', 'color_uniformity',
            'surface_defects', 'size', 'ripeness', 'overall_quality',
            'annotator_name', 'annotation_date', 'notes'
        ])
        
        template_path = self.base_path / "annotations/annotation_template.csv"
        template.to_csv(template_path, index=False)
        print(f"✓ Created annotation template at {template_path}")
        
        # Create grading guidelines
        guidelines = """
# Fruit Grading Guidelines

## Grade A - Excellent
- **Color**: Uniform, typical for variety, no discoloration
- **Surface**: No visible defects, smooth texture
- **Size**: Average or above average for variety
- **Ripeness**: Optimal ripeness, no over/under-ripe
- **Shape**: Perfect shape, no deformities

## Grade B - Good
- **Color**: Slightly uneven, minor discoloration
- **Surface**: Minor scratches, small spots (<5mm)
- **Size**: Slightly below average
- **Ripeness**: Acceptable ripeness
- **Shape**: Minor shape irregularities

## Grade C - Poor/Reject
- **Color**: Significant discoloration, off-color
- **Surface**: Visible defects, bruises, fungal spots
- **Size**: Below acceptable range
- **Ripeness**: Over-ripe or under-ripe
- **Shape**: Significant deformities
"""
        
        with open(self.base_path / "annotations/grading_guidelines.md", 'w') as f:
            f.write(guidelines)
        
        print("✓ Created grading guidelines")
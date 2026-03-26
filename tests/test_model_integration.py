"""
Model Integration Tests
"""

import unittest
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from webapp.model_loader import ModelLoader
from webapp.image_processor import ImageProcessor

class ModelIntegrationTest(unittest.TestCase):
    """Test model loading and prediction"""
    
    def setUp(self):
        """Set up test components"""
        self.model_loader = ModelLoader()
        self.image_processor = ImageProcessor()
    
    def test_model_loader_initialization(self):
        """Test model loader initialization"""
        self.assertIsNotNone(self.model_loader)
    
    def test_image_processor_initialization(self):
        """Test image processor initialization"""
        self.assertIsNotNone(self.image_processor)
        self.assertEqual(len(self.image_processor.class_names), 12)
    
    def test_image_preprocessing(self):
        """Test image preprocessing function"""
        # Create a dummy image path (this test will be skipped if no image)
        test_image_path = Path('ml/datasets/raw/apple/A')
        if test_image_path.exists():
            images = list(test_image_path.glob('*.jpg')) + list(test_image_path.glob('*.png'))
            if images:
                processed = self.image_processor.preprocess(str(images[0]))
                self.assertIsNotNone(processed)
                self.assertEqual(processed.shape, (1, 224, 224, 3))

if __name__ == '__main__':
    unittest.main()
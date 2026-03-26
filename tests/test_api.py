"""
API Integration Tests for Fruit Grading System
"""

import unittest
import os
import tempfile
import json
from pathlib import Path

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from webapp.app import app

class APITestCase(unittest.TestCase):
    """Test cases for Flask API endpoints"""
    
    def setUp(self):
        """Set up test client"""
        app.config['TESTING'] = True
        app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
        self.client = app.test_client()
        
        # Create test image if needed
        self.test_image_path = self._create_test_image()
    
    def tearDown(self):
        """Clean up after tests"""
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
    
    def _create_test_image(self):
        """Create a simple test image"""
        from PIL import Image
        import numpy as np
        
        # Create a simple RGB image
        img = Image.fromarray(np.ones((100, 100, 3), dtype=np.uint8) * 128)
        temp_path = tempfile.mktemp(suffix='.jpg')
        img.save(temp_path)
        return temp_path
    
    def test_home_page(self):
        """Test home page loads"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Fruit Grading System', response.data)
    
    def test_about_page(self):
        """Test about page loads"""
        response = self.client.get('/about')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'About', response.data)
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
    
    def test_predict_no_file(self):
        """Test predict endpoint with no file"""
        response = self.client.post('/predict')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_predict_invalid_file_type(self):
        """Test predict endpoint with invalid file type"""
        with open(__file__, 'rb') as f:
            response = self.client.post('/predict', data={
                'file': (f, 'test.py')
            })
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_predict_valid_image(self):
        """Test predict endpoint with valid image"""
        with open(self.test_image_path, 'rb') as f:
            response = self.client.post('/predict', data={
                'file': (f, 'test.jpg')
            })
        
        # Even if model not loaded, should return 500 not 400
        self.assertIn(response.status_code, [200, 500])
    
    def test_batch_predict(self):
        """Test batch predict endpoint"""
        with open(self.test_image_path, 'rb') as f1, open(self.test_image_path, 'rb') as f2:
            response = self.client.post('/batch_predict', data={
                'files': [(f1, 'test1.jpg'), (f2, 'test2.jpg')]
            })
        self.assertIn(response.status_code, [200, 500])
    
    def test_404_handler(self):
        """Test 404 error handler"""
        response = self.client.get('/nonexistent')
        self.assertEqual(response.status_code, 404)

if __name__ == '__main__':
    unittest.main()
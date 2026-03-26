"""
Image Processor Module
Handles image preprocessing and prediction conversion
"""

import cv2
import numpy as np
from PIL import Image
import os
import time
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Handles image preprocessing for fruit grading"""
    
    def __init__(self):
        self.target_size = (224, 224)
        self.last_processing_time = 0
        
        # Class mappings
        self.fruit_types = ['apple', 'mango', 'orange', 'tomato']
        self.grades = ['A', 'B', 'C']
        
        # Create class names mapping (12 total classes)
        self.class_names = []
        for fruit in self.fruit_types:
            for grade in self.grades:
                self.class_names.append(f"{fruit}_grade_{grade}")
        
        logger.info(f"ImageProcessor initialized with {len(self.class_names)} classes")
    
    def preprocess(self, image_path):
        """
        Preprocess image for model input
        
        Args:
            image_path: Path to image file
        
        Returns:
            Preprocessed image array (1, 224, 224, 3)
        """
        start_time = time.time()
        
        try:
            # Load image with OpenCV
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize
            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
            
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            self.last_processing_time = time.time() - start_time
            logger.debug(f"Image processed in {self.last_processing_time:.3f}s")
            
            return img
            
        except Exception as e:
            logger.error(f"Preprocessing error: {str(e)}")
            return None
    
    def preprocess_pil(self, pil_image):
        """
        Preprocess PIL image for model input
        
        Args:
            pil_image: PIL Image object
        
        Returns:
            Preprocessed image array (1, 224, 224, 3)
        """
        start_time = time.time()
        
        try:
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Resize
            pil_image = pil_image.resize(self.target_size, Image.LANCZOS)
            
            # Convert to numpy array
            img = np.array(pil_image)
            
            # Normalize
            img = img.astype(np.float32) / 255.0
            
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            self.last_processing_time = time.time() - start_time
            
            return img
            
        except Exception as e:
            logger.error(f"PIL preprocessing error: {str(e)}")
            return None
    
    def get_prediction_result(self, predictions):
        """
        Convert model predictions to human-readable result
        
        Args:
            predictions: Model output (1, 12)
        
        Returns:
            Dictionary with grade and confidence scores
        """
        # Get prediction for first batch
        pred = predictions[0]
        
        # Get top prediction
        predicted_class = np.argmax(pred)
        confidence = float(pred[predicted_class])
        
        # Get class name
        class_name = self.class_names[predicted_class]
        
        # Parse class name
        fruit_type, _, grade = class_name.split('_')
        
        # Get confidence scores for all grades
        confidence_scores = {}
        for fruit in self.fruit_types:
            for grade in self.grades:
                class_idx = self.class_names.index(f"{fruit}_grade_{grade}")
                confidence_scores[f"{fruit}_{grade}"] = float(pred[class_idx])
        
        # Get grade-specific confidence (for this fruit)
        grade_confidences = {}
        for grade in self.grades:
            class_idx = self.class_names.index(f"{fruit_type}_grade_{grade}")
            grade_confidences[grade] = float(pred[class_idx])
        
        return {
            'success': True,
            'fruit_type': fruit_type,
            'grade': grade,
            'confidence': confidence,
            'confidence_scores': confidence_scores,
            'grade_confidences': grade_confidences,
            'processing_time': self.last_processing_time
        }
    
    def get_top_k_predictions(self, predictions, k=3):
        """
        Get top k predictions
        
        Args:
            predictions: Model output (1, 12)
            k: Number of top predictions to return
        
        Returns:
            List of top k predictions with class names and confidence
        """
        pred = predictions[0]
        top_k_idx = np.argsort(pred)[-k:][::-1]
        
        results = []
        for idx in top_k_idx:
            results.append({
                'class': self.class_names[idx],
                'confidence': float(pred[idx])
            })
        
        return results
    
    def get_confidence_visualization(self, predictions):
        """
        Generate data for confidence visualization
        
        Args:
            predictions: Model output (1, 12)
        
        Returns:
            Dictionary with visualization data
        """
        pred = predictions[0]
        
        viz_data = {}
        for fruit in self.fruit_types:
            fruit_scores = []
            for grade in self.grades:
                class_idx = self.class_names.index(f"{fruit}_grade_{grade}")
                fruit_scores.append(float(pred[class_idx]))
            viz_data[fruit] = fruit_scores
        
        return viz_data
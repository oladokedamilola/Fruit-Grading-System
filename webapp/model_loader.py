"""
Model Loader Module
Handles loading and managing the trained model
"""

import tensorflow as tf
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ModelLoader:
    """Singleton class for loading and managing the fruit grading model"""
    
    _instance = None
    _model = None
    _metadata = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance
    
    def load_model(self, model_path=None):
        """
        Load the trained model from disk
        
        Args:
            model_path: Path to model file (H5 or SavedModel)
        
        Returns:
            Loaded TensorFlow model
        """
        if model_path is None:
            # Check multiple possible locations
            possible_paths = [
                os.getenv('MODEL_PATH', 'ml/models/fruit_grading_model.h5'),
                'ml/models/fruit_grading_efficientnetb0.h5',
                'ml/models/fruit_grading_final.h5',
                'ml/models/fruit_grading_model/',
                'webapp/models/fruit_grading_model.h5'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
        
        if self._model is None:
            try:
                logger.info(f"Loading model from {model_path}")
                
                # Check if it's a SavedModel directory
                if os.path.isdir(model_path):
                    self._model = tf.keras.models.load_model(model_path)
                else:
                    # Try loading as H5
                    self._model = tf.keras.models.load_model(model_path)
                
                logger.info(f"✓ Model loaded successfully")
                
                # Load model metadata if available
                metadata_path = Path('ml/models/model_metadata.json')
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        self._metadata = json.load(f)
                    logger.info(f"✓ Model metadata loaded")
                
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                self._model = None
        
        return self._model
    
    def get_metadata(self):
        """Get model metadata"""
        return self._metadata
    
    def is_loaded(self):
        """Check if model is loaded"""
        return self._model is not None
    
    def get_model_info(self):
        """Get model information"""
        if self._model is None:
            return {'loaded': False}
        
        return {
            'loaded': True,
            'input_shape': self._model.input_shape,
            'output_shape': self._model.output_shape,
            'total_params': self._model.count_params(),
            'metadata': self._metadata
        }
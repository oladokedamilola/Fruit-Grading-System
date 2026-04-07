"""
Fruit Identifier Module
Identifies fruit type from image using Pillow (no OpenCV dependency)
Runs entirely within the web app (no API call needed)
"""

import numpy as np
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)

class FruitIdentifier:
    """Lightweight fruit classifier to identify fruit type before grading"""
    
    def __init__(self, model_path=None):
        """
        Initialize the fruit identifier
        
        Args:
            model_path: Path to the fruit classification model (unused, kept for compatibility)
        """
        self.supported_fruits = ['apples', 'mangos', 'oranges']
        self.confidence_threshold = 0.6
    
    def _preprocess_image(self, image_bytes):
        """
        Preprocess image bytes for analysis
        
        Args:
            image_bytes: Raw image bytes
        
        Returns:
            PIL Image object or None
        """
        try:
            img = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            return img
            
        except Exception as e:
            logger.error(f"Preprocessing error in fruit identifier: {e}")
            return None
    
    def _get_average_color(self, img):
        """
        Get average RGB color of the image
        
        Args:
            img: PIL Image object
        
        Returns:
            Tuple (r, g, b) average values
        """
        # Resize to smaller size for faster processing
        img_small = img.resize((100, 100))
        pixels = np.array(img_small)
        
        # Calculate average RGB
        avg_r = np.mean(pixels[:, :, 0])
        avg_g = np.mean(pixels[:, :, 1])
        avg_b = np.mean(pixels[:, :, 2])
        
        return avg_r, avg_g, avg_b
    
    def _rule_based_identify(self, img):
        """
        Rule-based identification using color analysis
        
        Args:
            img: PIL Image object
        
        Returns:
            Tuple (fruit_type, confidence) or (None, 0)
        """
        try:
            # Get average color
            avg_r, avg_g, avg_b = self._get_average_color(img)
            
            # Calculate hue approximation from RGB
            # Simple approximation: red channel dominance indicates apple/orange
            # green channel dominance indicates green apple
            # yellow-orange indicates mango
            
            # Normalize
            total = avg_r + avg_g + avg_b + 0.001
            
            r_ratio = avg_r / total
            g_ratio = avg_g / total
            
            # Classification logic
            # Apples: high red or high green
            if r_ratio > 0.4:
                # Reddish fruits: could be apple or orange
                # Check if more orange-like (higher green component)
                if g_ratio > 0.35:
                    return 'oranges', 0.65
                else:
                    return 'apples', 0.70
            elif g_ratio > 0.4:
                # Greenish fruits: likely apples (green apples)
                return 'apples', 0.65
            elif r_ratio > 0.3 and g_ratio > 0.3:
                # Yellowish-orange: mango
                return 'mangos', 0.70
            else:
                # Default to apples with low confidence
                return 'apples', 0.50
                
        except Exception as e:
            logger.error(f"Rule-based identification error: {e}")
            return None, 0
    
    def identify(self, image_bytes):
        """
        Identify fruit type from image bytes
        
        Args:
            image_bytes: Raw image bytes
        
        Returns:
            Tuple (fruit_type, confidence, is_supported)
            - fruit_type: The identified fruit type (apples, mangos, oranges) or None
            - confidence: Confidence score (0-1)
            - is_supported: Boolean indicating if the fruit is supported
        """
        img = self._preprocess_image(image_bytes)
        if img is None:
            return None, 0, False
        
        fruit_type, confidence = self._rule_based_identify(img)
        is_supported = fruit_type in self.supported_fruits if fruit_type else False
        
        logger.info(f"Fruit identification: {fruit_type} (confidence: {confidence:.2%})")
        
        return fruit_type, confidence, is_supported
    
    def is_supported_fruit(self, image_bytes):
        """
        Quick check if the uploaded image contains a supported fruit
        
        Args:
            image_bytes: Raw image bytes
        
        Returns:
            Tuple (is_supported, fruit_type, confidence, message)
        """
        fruit_type, confidence, is_supported = self.identify(image_bytes)
        
        if not is_supported:
            supported_list = ", ".join(self.supported_fruits)
            message = f"Sorry, we currently only support {supported_list}. Please upload an image of an apple, mango, or orange."
        elif fruit_type:
            message = f"✓ Identified as {fruit_type}. Proceeding with quality grading..."
        else:
            message = "Could not identify the fruit. Please ensure the image is clear and contains a fruit."
        
        return is_supported, fruit_type, confidence, message


# Singleton instance
_identifier = None

def get_fruit_identifier():
    """Get or create the fruit identifier singleton"""
    global _identifier
    if _identifier is None:
        _identifier = FruitIdentifier()
    return _identifier
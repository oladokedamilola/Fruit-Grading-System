"""
ML API Client - Communicates with the Fruit Grading API server
"""

import requests
import logging
from flask import current_app, has_app_context

logger = logging.getLogger(__name__)

class MLAPIClient:
    """Client for the Fruit Grading ML API"""
    
    def __init__(self, api_url=None, timeout=90):
        self.api_url = api_url or 'http://localhost:5001'
        self.timeout = timeout
    
    def health_check(self):
        """Check if ML API is healthy"""
        try:
            response = requests.get(
                f"{self.api_url}/health",
                timeout=5
            )
            return response.status_code == 200 and response.json().get('status') == 'healthy'
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def predict(self, image_file):
        """
        Send image to ML API for prediction.
        The API now handles:
        1. Fruit identification (using pre-trained MobileNetV2)
        2. Quality grading (using fine-tuned MobileNetV2)
        
        Returns:
            Tuple (success, result_dict, error_message)
            - On success: result contains fruit_type, grade, confidence, grade_confidences, etc.
            - On unsupported fruit: success=False, error contains message, unsupported_fruit=True
        """
        try:
            # Reset file pointer
            image_file.seek(0)
            
            # Read file bytes for debugging
            file_bytes = image_file.read()
            print(f"[ML Client] File size: {len(file_bytes)} bytes")
            image_file.seek(0)
            
            # Prepare files
            files = {
                'file': (image_file.filename, image_file, image_file.content_type)
            }
            
            print(f"[ML Client] Calling API: {self.api_url}/predict")
            print(f"[ML Client] File: {image_file.filename}")
            
            # Make request to ML API
            response = requests.post(
                f"{self.api_url}/predict",
                files=files,
                timeout=self.timeout
            )
            
            print(f"[ML Client] Response status: {response.status_code}")
            print(f"[ML Client] Response preview: {response.text[:200]}")
            
            # Handle successful response (200)
            if response.status_code == 200:
                result = response.json()
                print(f"[ML Client] Success! Keys: {list(result.keys())}")
                return True, result, None
            
            # Handle unsupported fruit (400 with unsupported_fruit flag)
            elif response.status_code == 400:
                try:
                    error_data = response.json()
                    # Check if this is an unsupported fruit error
                    if error_data.get('unsupported_fruit'):
                        print(f"[ML Client] Unsupported fruit detected: {error_data.get('detected_fruit', 'unknown')}")
                        return False, None, error_data.get('error', 'Unsupported fruit')
                    else:
                        return False, None, error_data.get('error', f'API error: {response.status_code}')
                except:
                    return False, None, f'API error: {response.status_code}'
            
            # Handle other error status codes
            else:
                try:
                    error_data = response.json()
                    error_msg = error_data.get('error', f'API error: {response.status_code}')
                except:
                    error_msg = f'API error: {response.status_code} - {response.text[:100]}'
                print(f"[ML Client] Error: {error_msg}")
                return False, None, error_msg
                
        except requests.exceptions.Timeout:
            print("[ML Client] Timeout - API may be waking up")
            return False, None, "Request timed out. The API may be waking up. Please try again."
        except requests.exceptions.ConnectionError as e:
            print(f"[ML Client] Connection error: {e}")
            return False, None, f"Cannot connect to ML API: {str(e)}"
        except Exception as e:
            print(f"[ML Client] Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            logger.error(f"Prediction API error: {e}")
            return False, None, f"Prediction error: {str(e)}"
    
    def identify_only(self, image_file):
        """
        Optional: Only identify fruit type (no grading)
        Uses the /identify-only endpoint for debugging
        
        Args:
            image_file: File object from Flask request
        
        Returns:
            Tuple (fruit_type, confidence, is_supported, fruit_confidences)
        """
        try:
            image_file.seek(0)
            files = {
                'file': (image_file.filename, image_file, image_file.content_type)
            }
            
            response = requests.post(
                f"{self.api_url}/identify-only",
                files=files,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return (
                    result.get('fruit_type'),
                    result.get('confidence', 0),
                    result.get('is_supported', False),
                    result.get('fruit_confidences', {})
                )
            else:
                return None, 0, False, {}
                
        except Exception as e:
            logger.error(f"Identify-only error: {e}")
            return None, 0, False, {}
    
    def get_status(self):
        """Get detailed status of ML API"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None


# Global variable for ML client
_ml_client = None

def get_ml_client():
    """Get or create ML client instance"""
    global _ml_client
    
    if has_app_context():
        api_url = current_app.config.get('ML_API_URL', 'http://localhost:5001')
        timeout = current_app.config.get('ML_API_TIMEOUT', 90)
        
        if _ml_client is None:
            _ml_client = MLAPIClient(api_url=api_url, timeout=timeout)
        elif _ml_client.api_url != api_url:
            _ml_client = MLAPIClient(api_url=api_url, timeout=timeout)
    else:
        if _ml_client is None:
            _ml_client = MLAPIClient()
    
    return _ml_client


def init_ml_client(app):
    """Initialize ML client with app config"""
    global _ml_client
    with app.app_context():
        api_url = app.config.get('ML_API_URL', 'http://localhost:5001')
        timeout = app.config.get('ML_API_TIMEOUT', 90)
        _ml_client = MLAPIClient(api_url=api_url, timeout=timeout)
        return _ml_client
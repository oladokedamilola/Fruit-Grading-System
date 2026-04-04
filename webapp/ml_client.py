"""
ML API Client - Communicates with the Fruit Grading API server
"""

import requests
import base64
import logging
from flask import current_app, has_app_context

logger = logging.getLogger(__name__)

class MLAPIClient:
    """Client for the Fruit Grading ML API"""
    
    def __init__(self, api_url=None, timeout=30):
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
        Send image to ML API for prediction
        
        Args:
            image_file: File object from Flask request
        
        Returns:
            Tuple (success, result_dict, error_message)
        """
        try:
            # Reset file pointer to beginning
            image_file.seek(0)
            
            # Prepare multipart form data
            files = {
                'file': (image_file.filename, image_file, image_file.content_type)
            }
            
            print(f"📤 Calling ML API at: {self.api_url}/predict")
            print(f"📁 File: {image_file.filename}")
            
            # Make request to ML API
            response = requests.post(
                f"{self.api_url}/predict",
                files=files,
                timeout=self.timeout
            )
            
            print(f"📥 Response status: {response.status_code}")
            print(f"📄 Response text preview: {response.text[:200]}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Parsed result keys: {list(result.keys())}")
                return True, result, None
            else:
                try:
                    error_data = response.json()
                    error_msg = error_data.get('error', f'API error: {response.status_code}')
                except:
                    error_msg = f'API error: {response.status_code} - {response.text[:100]}'
                print(f"❌ API error: {error_msg}")
                return False, None, error_msg
                
        except requests.exceptions.Timeout:
            print("⏰ Request timeout")
            return False, None, "ML API request timed out. Please try again."
        except requests.exceptions.ConnectionError as e:
            print(f"🔌 Connection error: {e}")
            return False, None, "Cannot connect to ML API. Please ensure the API server is running."
        except Exception as e:
            print(f"💥 Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            logger.error(f"Prediction API error: {e}")
            return False, None, f"Prediction error: {str(e)}"
    
    def get_status(self):
        """Get detailed status of ML API"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None


# Global variable for ML client (initialized within app context)
_ml_client = None

def get_ml_client():
    """Get or create ML client instance - call within app context"""
    global _ml_client
    
    # Try to get config from current_app if in context
    if has_app_context():
        api_url = current_app.config.get('ML_API_URL', 'http://localhost:5001')
        timeout = current_app.config.get('ML_API_TIMEOUT', 30)
        
        if _ml_client is None:
            _ml_client = MLAPIClient(api_url=api_url, timeout=timeout)
        elif _ml_client.api_url != api_url:
            # Update if config changed
            _ml_client = MLAPIClient(api_url=api_url, timeout=timeout)
    else:
        # Fallback for when not in app context (should not happen)
        if _ml_client is None:
            _ml_client = MLAPIClient()
    
    return _ml_client


def init_ml_client(app):
    """Initialize ML client with app config (call during app startup)"""
    global _ml_client
    with app.app_context():
        api_url = app.config.get('ML_API_URL', 'http://localhost:5001')
        timeout = app.config.get('ML_API_TIMEOUT', 30)
        _ml_client = MLAPIClient(api_url=api_url, timeout=timeout)
        return _ml_client

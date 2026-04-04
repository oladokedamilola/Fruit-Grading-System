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
        Send image to ML API for prediction
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
            
            if response.status_code == 200:
                result = response.json()
                print(f"[ML Client] Success! Keys: {list(result.keys())}")
                return True, result, None
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
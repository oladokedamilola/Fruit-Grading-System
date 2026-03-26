"""
Configuration Module
Centralized configuration for the web application
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration"""
    
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key-change-in-production')
    DEBUG = os.getenv('FLASK_DEBUG', '1') == '1'
    
    # Upload settings
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'webapp/static/uploads')
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16MB
    
    # Model settings
    MODEL_PATH = os.getenv('MODEL_PATH', 'ml/models/fruit_grading_efficientnetb0.h5')
    IMG_HEIGHT = int(os.getenv('IMG_HEIGHT', 224))
    IMG_WIDTH = int(os.getenv('IMG_WIDTH', 224))
    
    # Fruit classes
    FRUIT_TYPES = ['apple', 'mango', 'orange', 'tomato']
    GRADES = ['A', 'B', 'C']
    
    # Application settings
    APP_NAME = 'Fruit Grading System'
    APP_VERSION = '1.0.0'
    
    @classmethod
    def init_app(cls, app):
        """Initialize application with config"""
        # Create upload folder if it doesn't exist
        os.makedirs(cls.UPLOAD_FOLDER, exist_ok=True)
        
        # Set secret key if not set
        if not app.config.get('SECRET_KEY'):
            app.config['SECRET_KEY'] = cls.SECRET_KEY

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.getenv('SECRET_KEY')  # Must be set in environment

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = False
    TESTING = True
    UPLOAD_FOLDER = 'webapp/static/test_uploads'
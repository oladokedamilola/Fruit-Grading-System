"""
Configuration Module for Fruit Grading System
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Base configuration"""
    
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key-change-in-production')
    DEBUG = os.getenv('FLASK_DEBUG', 'False') == 'True'
    
    # Database settings
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # API settings
    ML_API_URL = os.getenv('ML_API_URL', 'http://localhost:5001')
    ML_API_TIMEOUT = int(os.getenv('ML_API_TIMEOUT', 30))
    
    # Upload settings
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))
    FRUIT_IDENTIFIER_MODEL_PATH = os.getenv('FRUIT_IDENTIFIER_MODEL_PATH', None)
    # Fruit classes (for display purposes)
    FRUIT_TYPES = ['apples', 'mangos', 'oranges']
    GRADES = ['A', 'B', 'C']
    
    # Application settings
    APP_NAME = 'FruitSight'
    APP_VERSION = '2.0.0'
    
    # Anonymous user limits
    ANONYMOUS_PREDICTION_LIMIT = int(os.getenv('ANONYMOUS_LIMIT', 3))
    
    @classmethod
    def init_app(cls, app):
        """Initialize application with config"""
        pass


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///instance/fruit_grading.db')
    ML_API_URL = os.getenv('ML_API_URL', 'http://localhost:5001')


class ProductionConfig(Config):
    """Production configuration for Render"""
    DEBUG = False
    
    # Render provides DATABASE_URL (PostgreSQL)
    DATABASE_URL = os.environ.get('DATABASE_URL')
    if DATABASE_URL and DATABASE_URL.startswith('postgres://'):
        DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
    SQLALCHEMY_DATABASE_URI = DATABASE_URL or 'sqlite:///instance/fruit_grading.db'
    
    # Hugging Face Space URL - Get from environment variable
    # Format: https://USERNAME-fruitsight-api.hf.space
    ML_API_URL = os.environ.get('HUGGINGFACE_API_URL', os.environ.get('ML_API_URL', 'https://fruitsight-api.hf.space'))
    
    # Ensure the URL has the correct format
    if ML_API_URL and not ML_API_URL.startswith('http'):
        ML_API_URL = f'https://{ML_API_URL}'
    
    # Remove trailing slash if present
    if ML_API_URL and ML_API_URL.endswith('/'):
        ML_API_URL = ML_API_URL[:-1]


# Environment selection
config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

ENV = os.getenv('FLASK_ENV', 'development')
CurrentConfig = config_map.get(ENV, DevelopmentConfig)
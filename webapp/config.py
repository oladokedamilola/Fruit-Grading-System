"""
Configuration Module for Fruit Grading System
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Base configuration"""
    
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key-change-in-production')
    DEBUG = os.getenv('FLASK_DEBUG', 'False') == 'True'
    
    # Database settings
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Upload settings
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', '/tmp/uploads')
    
    # Model settings
    MODEL_PATH = os.getenv('MODEL_PATH', 'ml/models/fruit_grading_simple_cnn.keras')
    IMG_HEIGHT = int(os.getenv('IMG_HEIGHT', 224))
    IMG_WIDTH = int(os.getenv('IMG_WIDTH', 224))
    
    # Fruit classes
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
        os.makedirs(cls.UPLOAD_FOLDER, exist_ok=True)
        app.config['UPLOAD_FOLDER'] = cls.UPLOAD_FOLDER


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///instance/fruit_grading.db')
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'webapp/static/uploads')


class ProductionConfig(Config):
    """Production configuration for Render"""
    DEBUG = False
    
    # Render provides DATABASE_URL (PostgreSQL)
    @property
    def SQLALCHEMY_DATABASE_URI(self):
        DATABASE_URL = os.environ.get('DATABASE_URL')
        if DATABASE_URL and DATABASE_URL.startswith('postgres://'):
            DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
        return DATABASE_URL or 'sqlite:///instance/fruit_grading.db'
    
    # Use /tmp for ephemeral storage on Render
    UPLOAD_FOLDER = '/tmp/uploads'


# Environment selection
config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

ENV = os.getenv('FLASK_ENV', 'development')
CurrentConfig = config_map.get(ENV, DevelopmentConfig)
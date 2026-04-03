"""
Database models for user authentication and history tracking
"""

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
import json

db = SQLAlchemy()

class User(UserMixin, db.Model):
    """User model for authentication"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship with predictions
    predictions = db.relationship('Prediction', backref='user', lazy=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat()
        }

class Prediction(db.Model):
    """Prediction history model"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    session_id = db.Column(db.String(100), nullable=True)
    fruit_type = db.Column(db.String(50), nullable=False)
    grade = db.Column(db.String(1), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    confidence_scores = db.Column(db.Text, nullable=True)  # JSON string
    grade_confidences = db.Column(db.Text, nullable=True)  # JSON string for A/B/C breakdown
    image_data = db.Column(db.Text, nullable=True)  # Base64 encoded image
    image_filename = db.Column(db.String(200), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    ip_address = db.Column(db.String(50), nullable=True)
    session_id = db.Column(db.String(100), nullable=True)  # For anonymous tracking
    
    def to_dict(self):
        return {
            'id': self.id,
            'fruit_type': self.fruit_type,
            'grade': self.grade,
            'confidence': self.confidence,
            'confidence_scores': json.loads(self.confidence_scores) if self.confidence_scores else {},
            'grade_confidences': json.loads(self.grade_confidences) if self.grade_confidences else {},
            'image_data': self.image_data,
            'created_at': self.created_at.isoformat()
        }

class AnonymousUsage(db.Model):
    """Track anonymous user usage by session ID"""
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(100), unique=True, nullable=False)
    prediction_count = db.Column(db.Integer, default=0)
    last_used = db.Column(db.DateTime, default=datetime.utcnow)
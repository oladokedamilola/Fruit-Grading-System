"""
Main Flask Application with Authentication and Usage Limits
"""

import os
import sys
import uuid
import json
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, flash, redirect, url_for
from flask_cors import CORS
from flask_login import LoginManager, login_required, current_user, login_user, logout_user
from flask_migrate import Migrate
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from pathlib import Path

# Get absolute path to instance folder
BASE_DIR = Path(__file__).resolve().parent.parent
INSTANCE_PATH = BASE_DIR / 'instance'
INSTANCE_PATH.mkdir(exist_ok=True)

from webapp.config import Config, CurrentConfig
from webapp.models import db, User, Prediction, AnonymousUsage
from webapp.auth import auth_bp
from webapp.ml_client import init_ml_client
import logging
import base64

load_dotenv()
logger = logging.getLogger(__name__)

# Initialize Flask app with configuration
app = Flask(__name__)
app.config.from_object(CurrentConfig)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{INSTANCE_PATH}/fruit_grading.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')

# Session configuration for flash messages
app.config['SESSION_TYPE'] = 'filesystem'

# Initialize extensions
db.init_app(app)
CORS(app)
migrate = Migrate(app, db)

# Setup Login Manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login_page'
login_manager.login_message = 'Please login to continue'
login_manager.login_message_category = 'warning'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Register blueprints
app.register_blueprint(auth_bp, url_prefix='/auth')

# ============================================
# Initialize ML API client (inside app context)
# ============================================

with app.app_context():
    db.create_all()
    print("✓ Database tables verified")
    
    # Initialize ML client
    ml_client = init_ml_client(app)
    print(f"✓ ML API client initialized (URL: {app.config.get('ML_API_URL', 'http://localhost:5001')})")
    
    # Check ML API health
    if ml_client.health_check():
        print("✓ ML API is healthy")
    else:
        print("⚠ ML API is not responding. Predictions will not work.")

# Allowed file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

from datetime import date

def get_anonymous_usage_count(session_id):
    """Get number of predictions made by anonymous user today"""
    usage = AnonymousUsage.query.filter_by(session_id=session_id).first()
    
    # If no record exists, return 0
    if not usage:
        return 0
    
    # If the record is from a different day, reset count
    if usage.date != date.today():
        usage.prediction_count = 0
        usage.date = date.today()
        db.session.commit()
        return 0
    
    return usage.prediction_count

def increment_anonymous_usage(session_id):
    """Increment anonymous user's prediction count for today"""
    usage = AnonymousUsage.query.filter_by(session_id=session_id).first()
    
    if usage:
        # Reset if different day
        if usage.date != date.today():
            usage.prediction_count = 1
            usage.date = date.today()
            usage.last_used = datetime.utcnow()
        else:
            usage.prediction_count += 1
            usage.last_used = datetime.utcnow()
    else:
        usage = AnonymousUsage(
            session_id=session_id, 
            prediction_count=1,
            date=date.today()
        )
        db.session.add(usage)
    
    db.session.commit()

def save_prediction(user_id, session_id, result, image_data=None):
    """Save prediction to database with image"""
    
    prediction = Prediction(
        user_id=user_id,
        session_id=session_id if not user_id else None,
        fruit_type=result['fruit_type'],
        grade=result['grade'],
        confidence=result['confidence'],
        confidence_scores=json.dumps(result.get('grade_confidences', {})),
        grade_confidences=json.dumps(result.get('grade_confidences', {})),
        image_data=image_data,
        ip_address=request.remote_addr
    )
    db.session.add(prediction)
    db.session.commit()
    return prediction.id

# ============================================
# Page Routes
# ============================================

@app.route('/')
def index():
    """Home/Landing page"""
    return render_template('index.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/upload')
def upload_page():
    """Upload/Grade page"""
    return render_template('upload.html')

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard with history"""
    return render_template('dashboard.html')

# ============================================
# Authentication Routes (HTML pages)
# ============================================

@app.route('/login')
def login_page():
    """Login page"""
    if current_user.is_authenticated:
        flash('You are already logged in!', 'info')
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/register')
def register_page():
    """Register page"""
    if current_user.is_authenticated:
        flash('You are already logged in!', 'info')
        return redirect(url_for('dashboard'))
    return render_template('register.html')

# ============================================
# API Routes
# ============================================

@app.route('/api/login', methods=['POST'])
def api_login():
    """API endpoint for login"""
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    # Find user by email instead of username
    user = User.query.filter_by(email=email).first()
    
    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({'success': False, 'error': 'Invalid email or password'}), 401
    
    login_user(user)
    flash(f'Welcome back, {user.username}!', 'success')
    
    return jsonify({
        'success': True,
        'user': user.to_dict(),
        'redirect': url_for('dashboard')
    })

@app.route('/api/register', methods=['POST'])
def api_register():
    """API endpoint for registration (email only)"""
    data = request.get_json()
    
    email = data.get('email')
    password = data.get('password')
    
    # Validate
    if not email or not password:
        return jsonify({'success': False, 'error': 'Email and password are required'}), 400
    
    if User.query.filter_by(email=email).first():
        return jsonify({'success': False, 'error': 'Email already registered'}), 400
    
    if len(password) < 8:
        return jsonify({'success': False, 'error': 'Password must be at least 8 characters'}), 400
    
    # Generate username from email
    username = email.split('@')[0]
    # Ensure username is unique
    base_username = username
    counter = 1
    while User.query.filter_by(username=username).first():
        username = f"{base_username}{counter}"
        counter += 1
    
    # Create user
    user = User(
        username=username,
        email=email,
        password_hash=generate_password_hash(password)
    )
    
    db.session.add(user)
    db.session.commit()
    
    login_user(user)
    flash(f'Welcome to FruitSight, {username}! Your account has been created.', 'success')
    
    return jsonify({
        'success': True,
        'user': user.to_dict(),
        'redirect': url_for('dashboard')
    })

@app.route('/api/logout', methods=['POST'])
def api_logout():
    """API endpoint for logout"""
    logout_user()
    flash('You have been logged out successfully.', 'info')
    return jsonify({'success': True, 'redirect': url_for('index')})

@app.route('/api/check-session')
def api_check_session():
    """Check if user is logged in and get usage stats"""
    if current_user.is_authenticated:
        return jsonify({
            'logged_in': True,
            'user': current_user.to_dict(),
            'is_anonymous': False
        })
    else:
        # Track anonymous session
        session_id = session.get('session_id')
        if not session_id:
            session_id = str(uuid.uuid4())
            session['session_id'] = session_id
        
        # Get usage count
        usage = AnonymousUsage.query.filter_by(session_id=session_id).first()
        remaining = max(0, 3 - (usage.prediction_count if usage else 0))
        
        return jsonify({
            'logged_in': False,
            'is_anonymous': True,
            'remaining_predictions': remaining,
            'session_id': session_id
        })

# ============================================
# Prediction Route - Uses ML API
# ============================================

@app.route('/predict', methods=['POST'])
def predict():
    """Predict fruit grade using ML API"""
    
    # Check usage limits
    if current_user.is_authenticated:
        user_id = current_user.id
        session_id = None
    else:
        session_id = session.get('session_id')
        if not session_id:
            session_id = str(uuid.uuid4())
            session['session_id'] = session_id
        
        usage_count = get_anonymous_usage_count(session_id)
        if usage_count >= 3:
            return jsonify({
                'success': False,
                'error': 'Anonymous usage limit reached. Please login or register for unlimited access.',
                'limit_reached': True,
                'login_required': True
            }), 403
    
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'File type not allowed. Allowed: JPG, JPEG, PNG'}), 400
    
    try:
        # Call ML API
        success, result, error = ml_client.predict(file)
        
        if not success:
            return jsonify({'success': False, 'error': error}), 500
        
        result['success'] = True
        
        # Save prediction to database
        prediction_id = save_prediction(
            user_id=current_user.id if current_user.is_authenticated else None,
            session_id=session_id if not current_user.is_authenticated else None,
            result=result,
            image_data=result.get('image_base64')
        )
        result['prediction_id'] = prediction_id
        
        # Increment anonymous usage if needed
        if not current_user.is_authenticated:
            increment_anonymous_usage(session_id)
            remaining = 2 - get_anonymous_usage_count(session_id)
            result['remaining_predictions'] = remaining if remaining >= 0 else 0
        
        # Flash message for logged-in users
        if current_user.is_authenticated:
            flash(f'Your {result["fruit_type"]} was graded as Grade {result["grade"]}!', 'success')
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'success': False, 'error': f'Prediction error: {str(e)}'}), 500

@app.route('/results')
def results_page():
    """Results page"""
    return render_template('results.html')

# ============================================
# Enhanced History Routes
# ============================================

@app.route('/api/history')
@login_required
def api_get_history():
    """Get user's prediction history with pagination and filters"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('limit', 10, type=int)
    grade_filter = request.args.get('grade', 'all')
    fruit_filter = request.args.get('fruit', 'all')
    sort_order = request.args.get('sort', 'desc')
    
    # Build query
    query = Prediction.query.filter_by(user_id=current_user.id)
    
    # Apply filters
    if grade_filter != 'all':
        query = query.filter_by(grade=grade_filter)
    if fruit_filter != 'all':
        query = query.filter_by(fruit_type=fruit_filter)
    
    # Apply sorting
    if sort_order == 'desc':
        query = query.order_by(Prediction.created_at.desc())
    else:
        query = query.order_by(Prediction.created_at.asc())
    
    # Get total count
    total = query.count()
    
    # Paginate
    predictions = query.offset((page - 1) * per_page).limit(per_page).all()
    
    return jsonify({
        'success': True,
        'predictions': [p.to_dict() for p in predictions],
        'total': total,
        'pagination': {
            'current_page': page,
            'per_page': per_page,
            'total_pages': (total + per_page - 1) // per_page,
            'total_items': total
        }
    })

@app.route('/api/prediction/<int:prediction_id>')
@login_required
def api_get_prediction(prediction_id):
    """Get a single prediction by ID"""
    prediction = Prediction.query.filter_by(id=prediction_id, user_id=current_user.id).first_or_404()
    
    return jsonify({
        'success': True,
        'prediction': prediction.to_dict()
    })

@app.route('/history')
@login_required
def history_page():
    """History page"""
    return render_template('history.html')

@app.route('/result/<int:prediction_id>')
@login_required
def result_page(prediction_id):
    """Individual result page"""
    return render_template('result.html')

@app.route('/api/profile')
@login_required
def api_get_profile():
    """Get user profile data (API)"""
    predictions = Prediction.query.filter_by(user_id=current_user.id).all()
    
    # Calculate average confidence
    avg_confidence = 0
    if predictions:
        avg_confidence = sum(p.confidence for p in predictions) / len(predictions)
    
    return jsonify({
        'success': True,
        'user': current_user.to_dict(),
        'total_predictions': len(predictions),
        'avg_confidence': avg_confidence
    })

# ============================================
# Transfer Anonymous Prediction
# ============================================

@app.route('/api/transfer-prediction', methods=['POST'])
@login_required
def transfer_prediction():
    """Transfer anonymous prediction to logged-in user"""
    data = request.get_json()
    prediction_id = data.get('prediction_id')
    
    if not prediction_id:
        return jsonify({'success': False, 'error': 'No prediction ID provided'}), 400
    
    # Get the anonymous prediction
    prediction = Prediction.query.filter_by(id=prediction_id, user_id=None).first()
    
    if not prediction:
        return jsonify({'success': False, 'error': 'Prediction not found or already transferred'}), 404
    
    # Transfer to current user
    prediction.user_id = current_user.id
    prediction.session_id = None
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': 'Prediction saved to your account!',
        'prediction_id': prediction.id
    })

# ============================================
# Health Check
# ============================================

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'ml_api_healthy': ml_client.health_check(),
        'authenticated': current_user.is_authenticated if hasattr(current_user, 'is_authenticated') else False,
        'timestamp': datetime.now().isoformat()
    })



@app.route('/debug-api')
def debug_api():
    """Debug endpoint to test ML API connection"""
    import requests
    try:
        # Test health
        health_resp = requests.get(f"{app.config.get('ML_API_URL')}/health", timeout=10)
        
        return jsonify({
            'api_url': app.config.get('ML_API_URL'),
            'health_status': health_resp.status_code,
            'health_response': health_resp.json() if health_resp.ok else None,
            'ml_client_healthy': ml_client.health_check()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
# ============================================
# Error Handlers
# ============================================

@app.errorhandler(404)
def not_found_error(error):
    """404 Not Found error handler"""
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'error': 'Endpoint not found'}), 404
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """500 Internal Server Error handler"""
    db.session.rollback()
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'error': 'Internal server error'}), 500
    return render_template('500.html'), 500

@app.errorhandler(403)
def forbidden_error(error):
    """403 Forbidden error handler"""
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    return render_template('403.html'), 403

@app.errorhandler(413)
def too_large_error(error):
    """413 Request Entity Too Large error handler"""
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'error': 'File too large. Max size: 16MB'}), 413
    return render_template('413.html'), 413

@app.errorhandler(429)
def too_many_requests_error(error):
    """429 Too Many Requests error handler"""
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'error': 'Rate limit exceeded. Please try again later.'}), 429
    return render_template('429.html'), 429

# Generic exception handler for debugging (remove in production)
@app.errorhandler(Exception)
def handle_exception(error):
    """Generic exception handler"""
    # Log the error
    app.logger.error(f'Unhandled exception: {error}')
    
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'error': 'An unexpected error occurred'}), 500
    
    # For non-API requests, show 500 page
    return render_template('500.html'), 500

# ============================================
# Main Entry Point
# ============================================

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

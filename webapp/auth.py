"""
Authentication routes for login, register, and logout
"""

from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from webapp.models import db, User, Prediction, AnonymousUsage
from datetime import datetime
import uuid
import re

auth_bp = Blueprint('auth', __name__)

# Password validation function
def validate_password(password):
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one number"
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character"
    return True, "Password is strong"

# ============================================
# Page Routes
# ============================================

@auth_bp.route('/login-page')
def login_page():
    """Login page"""
    if current_user.is_authenticated:
        flash('You are already logged in!', 'info')
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@auth_bp.route('/register-page')
def register_page():
    """Register page"""
    if current_user.is_authenticated:
        flash('You are already logged in!', 'info')
        return redirect(url_for('dashboard'))
    return render_template('register.html')

# ============================================
# API Routes
# ============================================

@auth_bp.route('/login', methods=['POST'])
def login():
    """API endpoint for login using email"""
    data = request.get_json()
    
    email = data.get('email')
    password = data.get('password')
    
    # Find user by email
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

@auth_bp.route('/register', methods=['POST'])
def register():
    """API endpoint for registration (email + password only)"""
    try:
        data = request.get_json()
        
        # Debug logging
        print(f"Received registration data: {data}")
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        email = data.get('email')
        password = data.get('password')
        
        # Check if fields are present
        if not email or not password:
            missing_fields = []
            if not email:
                missing_fields.append('email')
            if not password:
                missing_fields.append('password')
            return jsonify({
                'success': False, 
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Validate email
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
            return jsonify({'success': False, 'error': 'Invalid email format'}), 400
        
        # Check if email already exists
        if User.query.filter_by(email=email).first():
            return jsonify({'success': False, 'error': 'Email already registered'}), 400
        
        # Validate password
        is_valid, password_message = validate_password(password)
        if not is_valid:
            return jsonify({'success': False, 'error': password_message}), 400
        
        # Generate username from email (before @ symbol)
        username = email.split('@')[0]
        # Remove invalid characters for username
        username = re.sub(r'[^a-zA-Z0-9_]', '', username)
        
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
        
        return jsonify({
            'success': True,
            'user': user.to_dict(),
            'redirect': url_for('dashboard')
        })
        
    except Exception as e:
        print(f"Registration error: {str(e)}")
        db.session.rollback()
        return jsonify({'success': False, 'error': 'An error occurred during registration'}), 500
    
    
@auth_bp.route('/logout')
@login_required
def logout():
    """Logout route"""
    logout_user()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('index'))

@auth_bp.route('/profile')
@login_required
def profile():
    """Get user profile with prediction history"""
    predictions = Prediction.query.filter_by(user_id=current_user.id)\
                                  .order_by(Prediction.created_at.desc())\
                                  .limit(50).all()
    
    return jsonify({
        'user': current_user.to_dict(),
        'predictions': [p.to_dict() for p in predictions],
        'total_predictions': len(predictions)
    })

@auth_bp.route('/check-session')
def check_session():
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
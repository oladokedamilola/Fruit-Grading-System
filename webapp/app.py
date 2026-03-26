"""
Main Flask Application for Fruit Grading System
Handles routing, file uploads, and prediction endpoints
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from dotenv import load_dotenv

from webapp.model_loader import ModelLoader
from webapp.image_processor import ImageProcessor
from webapp.config import Config

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Enable CORS
CORS(app)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize components
model_loader = ModelLoader()
image_processor = ImageProcessor()

# Load model at startup
logger.info("Loading model...")
model = model_loader.load_model()
if model:
    logger.info("✓ Model loaded successfully")
else:
    logger.error("✗ Failed to load model")

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Home page with upload form"""
    return render_template('index.html')

@app.route('/about')
def about():
    """About page with project information"""
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict fruit grade from uploaded image
    Returns JSON with grade and confidence scores
    """
    # Check if file is present
    if 'file' not in request.files:
        logger.warning("No file in request")
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    # Check if file is selected
    if file.filename == '':
        logger.warning("Empty filename")
        return jsonify({'error': 'No file selected'}), 400
    
    # Check file type
    if not allowed_file(file.filename):
        logger.warning(f"Invalid file type: {file.filename}")
        return jsonify({'error': f'File type not allowed. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    try:
        # Secure filename
        filename = secure_filename(file.filename)
        
        # Save temporarily
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        logger.info(f"File saved: {temp_path}")
        
        # Preprocess image
        processed_image = image_processor.preprocess(temp_path)
        if processed_image is None:
            logger.error(f"Failed to preprocess image: {temp_path}")
            os.remove(temp_path)
            return jsonify({'error': 'Failed to process image'}), 500
        
        # Run inference
        if model is None:
            logger.error("Model not loaded")
            os.remove(temp_path)
            return jsonify({'error': 'Model not loaded'}), 500
        
        predictions = model.predict(processed_image)
        
        # Convert predictions to grades
        result = image_processor.get_prediction_result(predictions)
        
        # Add processing time
        result['processing_time'] = image_processor.last_processing_time
        
        # Log result
        logger.info(f"Prediction: {result['fruit_type']} - Grade {result['grade']} "
                   f"(Confidence: {result['confidence']:.2%})")
        
        # Clean up temp file
        os.remove(temp_path)
        logger.info(f"Temp file removed: {temp_path}")
        
        return jsonify(result)
        
    except RequestEntityTooLarge:
        logger.error("File too large")
        return jsonify({'error': f'File too large. Max size: {app.config["MAX_CONTENT_LENGTH"] // (1024*1024)}MB'}), 400
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction for multiple images
    Returns JSON with results for all images
    """
    # Check if files are present
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('files')
    if len(files) == 0:
        return jsonify({'error': 'No files selected'}), 400
    
    if len(files) > 10:
        return jsonify({'error': 'Maximum 10 files per batch'}), 400
    
    results = []
    
    for file in files:
        if not allowed_file(file.filename):
            results.append({
                'filename': file.filename,
                'error': 'Invalid file type'
            })
            continue
        
        try:
            filename = secure_filename(file.filename)
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(temp_path)
            
            processed_image = image_processor.preprocess(temp_path)
            if processed_image is not None and model is not None:
                predictions = model.predict(processed_image)
                result = image_processor.get_prediction_result(predictions)
                result['filename'] = filename
                results.append(result)
            else:
                results.append({
                    'filename': filename,
                    'error': 'Processing failed'
                })
            
            os.remove(temp_path)
            
        except Exception as e:
            logger.error(f"Batch prediction error for {file.filename}: {str(e)}")
            results.append({
                'filename': file.filename,
                'error': 'Processing error'
            })
    
    return jsonify({
        'success': True,
        'total': len(results),
        'results': results
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.errorhandler(404)
def not_found(error):
    """404 error handler"""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """500 error handler"""
    logger.error(f"Internal error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(
        debug=os.getenv('FLASK_DEBUG', '1') == '1',
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000))
    )
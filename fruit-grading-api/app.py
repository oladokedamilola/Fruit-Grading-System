import os
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from datetime import datetime
from pathlib import Path
import urllib.request

app = Flask(__name__)
CORS(app)

# ============================================
# Configuration
# ============================================

FRUIT_TYPES = ['apples', 'mangos', 'oranges']
GRADES = ['A', 'B', 'C']
SUPPORTED_FRUITS = ['apples', 'mangos', 'oranges']

# ImageNet class indices for fruits (from MobileNetV2)
# These are the class indices that correspond to our fruits
IMAGENET_FRUIT_INDICES = {
    'apples': [948],  # Granny Smith, Red Delicious, etc. (multiple indices)
    'mangos': [955],   # mango
    'oranges': [951, 950]  # orange, navel orange
}

# Expanded mapping with more specific classes
IMAGENET_FRUIT_MAPPING = {
    'apples': ['Granny Smith', 'Red Delicious', 'Crab apple', 'Pomelo'],
    'mangos': ['mango'],
    'oranges': ['orange', 'navel orange']
}

# Class names for grader model (9 classes)
grader_class_names = [f"{fruit}_{grade}" for fruit in FRUIT_TYPES for grade in GRADES]

# ============================================
# Model Paths
# ============================================

def find_grader_model():
    """Find the fine-tuned grading model"""
    possible_paths = ['model.keras', 'fruit_grading_mobilenet.keras', 'fruit_grading_simple_cnn.keras']
    for path in possible_paths:
        if Path(path).exists():
            return path
    return None

GRADER_MODEL_PATH = find_grader_model()

print("=" * 50)
print("🍎 Fruit Grading API Server (Two-Model Architecture)")
print("=" * 50)

# ============================================
# Load Pre-trained MobileNetV2 (Identifier)
# ============================================
print("\n📁 Loading pre-trained MobileNetV2 (ImageNet) for fruit identification...")
try:
    # Load pre-trained MobileNetV2 without the top classification layer
    base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=True,  # Keep the top layer for classification
        input_shape=(224, 224, 3)
    )
    identifier_model = base_model
    print("✓ Pre-trained MobileNetV2 loaded successfully")
except Exception as e:
    print(f"✗ Failed to load pre-trained model: {e}")
    identifier_model = None

# ============================================
# Load Fine-tuned Grader Model
# ============================================
print(f"\n📁 Grader model path: {GRADER_MODEL_PATH}")
if GRADER_MODEL_PATH:
    grader_model = tf.keras.models.load_model(GRADER_MODEL_PATH)
    print("✓ Grader model loaded successfully")
else:
    print("✗ Grader model not found")
    grader_model = None

# ============================================
# Helper Functions
# ============================================

def preprocess_image(image_bytes):
    """Preprocess image for model input (224x224)"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def identify_with_pretrained_model(image_bytes):
    """
    Use pre-trained MobileNetV2 to identify fruit type
    Returns: (fruit_type, confidence, all_predictions_summary)
    """
    processed = preprocess_image(image_bytes)
    if processed is None:
        return None, 0, {}
    
    # Get predictions from pre-trained model
    predictions = identifier_model.predict(processed, verbose=0)[0]
    
    # Decode predictions to get class names
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(
        np.expand_dims(predictions, axis=0), top=10
    )[0]
    
    print(f"🔍 Pre-trained model top predictions:")
    for i, (imagenet_id, label, score) in enumerate(decoded[:5]):
        print(f"   {i+1}: {label} ({score:.2%})")
    
    # Map ImageNet labels to our fruit types
    fruit_scores = {
        'apples': 0,
        'mangos': 0,
        'oranges': 0
    }
    
    # Keywords to match
    fruit_keywords = {
        'apples': ['apple', 'granny smith', 'red delicious', 'crab apple', 'pomelo'],
        'mangos': ['mango'],
        'oranges': ['orange', 'navel orange', 'mandarin', 'tangerine']
    }
    
    for imagenet_id, label, score in decoded:
        label_lower = label.lower()
        for fruit, keywords in fruit_keywords.items():
            for keyword in keywords:
                if keyword in label_lower:
                    fruit_scores[fruit] += score
                    break
    
    # Normalize scores
    total = sum(fruit_scores.values())
    if total > 0:
        for fruit in fruit_scores:
            fruit_scores[fruit] = fruit_scores[fruit] / total
    
    # Get fruit with highest score
    identified_fruit = max(fruit_scores, key=fruit_scores.get)
    confidence = fruit_scores[identified_fruit]
    
    # If confidence is too low, consider it unsupported
    if confidence < 0.3:
        return None, confidence, fruit_scores
    
    return identified_fruit, confidence, fruit_scores

def get_grade_for_fruit(image_bytes, fruit_type):
    """
    Get grade predictions for a specific fruit using the fine-tuned grader model
    """
    processed = preprocess_image(image_bytes)
    if processed is None:
        return None, 0, {}
    
    predictions = grader_model.predict(processed, verbose=0)[0]
    
    grade_confidences = {}
    for grade in GRADES:
        idx = grader_class_names.index(f"{fruit_type}_{grade}")
        grade_confidences[grade] = float(predictions[idx])
    
    predicted_grade = max(grade_confidences, key=grade_confidences.get)
    confidence = grade_confidences[predicted_grade]
    
    return predicted_grade, confidence, grade_confidences

# ============================================
# Routes
# ============================================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'identifier_model_loaded': identifier_model is not None,
        'grader_model_loaded': grader_model is not None,
        'supported_fruits': SUPPORTED_FRUITS,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Two-stage prediction:
    1. Identify fruit type using pre-trained MobileNetV2
    2. If supported, get quality grade using fine-tuned grader model
    """
    if grader_model is None:
        return jsonify({'error': 'Grader model not loaded'}), 500
    
    if identifier_model is None:
        return jsonify({'error': 'Identifier model not loaded'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        image_bytes = file.read()
        
        # ============================================
        # STEP 1: Identify Fruit Type using Pre-trained Model
        # ============================================
        fruit_type, fruit_confidence, fruit_confidences = identify_with_pretrained_model(image_bytes)
        
        if fruit_type is None:
            return jsonify({
                'success': False,
                'error': "Could not identify a supported fruit. Please upload an image of an apple, mango, or orange.",
                'unsupported_fruit': True,
                'detected_fruit': None,
                'detected_confidence': fruit_confidence,
                'supported_fruits': SUPPORTED_FRUITS,
                'fruit_confidences': fruit_confidences
            }), 400
        
        print(f"🔍 Pre-trained model identified: {fruit_type} (confidence: {fruit_confidence:.2%})")
        
        # ============================================
        # STEP 2: Check if Supported
        # ============================================
        if fruit_type not in SUPPORTED_FRUITS:
            return jsonify({
                'success': False,
                'error': f"Unsupported fruit detected: {fruit_type}. Please upload apples, mangoes, or oranges.",
                'unsupported_fruit': True,
                'detected_fruit': fruit_type,
                'detected_confidence': fruit_confidence,
                'supported_fruits': SUPPORTED_FRUITS,
                'fruit_confidences': fruit_confidences
            }), 400
        
        # ============================================
        # STEP 3: Get Quality Grade using Fine-tuned Model
        # ============================================
        predicted_grade, grade_confidence, grade_confidences = get_grade_for_fruit(image_bytes, fruit_type)
        
        print(f"✅ Supported fruit: {fruit_type}")
        print(f"📊 Grade: {predicted_grade} (confidence: {grade_confidence:.2%})")
        
        # ============================================
        # STEP 4: Return Response
        # ============================================
        return jsonify({
            'success': True,
            'fruit_type': fruit_type,
            'fruit_identification_confidence': fruit_confidence,
            'fruit_confidences': fruit_confidences,
            'grade': predicted_grade,
            'confidence': grade_confidence,
            'grade_confidences': grade_confidences,
            'image_base64': base64.b64encode(image_bytes).decode('utf-8'),
            'processing_time': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Error in predict: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/identify-only', methods=['POST'])
def identify_only():
    """
    Optional endpoint: Only identify fruit type (no grading)
    Useful for debugging
    """
    if identifier_model is None:
        return jsonify({'error': 'Identifier model not loaded'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        image_bytes = file.read()
        fruit_type, confidence, fruit_confidences = identify_with_pretrained_model(image_bytes)
        
        return jsonify({
            'success': True,
            'fruit_type': fruit_type,
            'confidence': confidence,
            'fruit_confidences': fruit_confidences,
            'is_supported': fruit_type in SUPPORTED_FRUITS if fruit_type else False,
            'supported_fruits': SUPPORTED_FRUITS
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 7860))
    print(f"\n🚀 Starting FruitSight API Server")
    print(f"   Port: {port}")
    print(f"   Supported fruits: {SUPPORTED_FRUITS}")
    print(f"   Identifier: Pre-trained MobileNetV2 (ImageNet)")
    print(f"   Grader: Fine-tuned MobileNetV2")
    print("\n" + "=" * 50)
    app.run(host='0.0.0.0', port=port)
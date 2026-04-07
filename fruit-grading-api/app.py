import os
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from datetime import datetime
from pathlib import Path

app = Flask(__name__)
CORS(app)

# ============================================
# Configuration
# ============================================

FRUIT_TYPES = ['apples', 'mangos', 'oranges']
GRADES = ['A', 'B', 'C']
SUPPORTED_FRUITS = ['apples', 'mangos', 'oranges']

# Confidence threshold for fruit identification
IDENTIFICATION_CONFIDENCE_THRESHOLD = 0.3

# Class names for grader model (9 classes)
grader_class_names = [f"{fruit}_{grade}" for fruit in FRUIT_TYPES for grade in GRADES]

# ============================================
# Model Paths
# ============================================

# Path to pre-trained model (uploaded to Space)
PRETRAINED_MODEL_PATH = 'pretrained_mobilenetv2.keras'

# Path to fine-tuned grader model
def find_grader_model():
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
print("\n📁 Loading pre-trained MobileNetV2 from local file...")
identifier_model = None
if Path(PRETRAINED_MODEL_PATH).exists():
    try:
        identifier_model = tf.keras.models.load_model(PRETRAINED_MODEL_PATH)
        print("✓ Pre-trained MobileNetV2 loaded successfully from local file")
    except Exception as e:
        print(f"✗ Failed to load pre-trained model: {e}")
else:
    print(f"✗ Pre-trained model not found at {PRETRAINED_MODEL_PATH}")

# ============================================
# Load Fine-tuned Grader Model
# ============================================
print(f"\n📁 Grader model path: {GRADER_MODEL_PATH}")
if GRADER_MODEL_PATH and Path(GRADER_MODEL_PATH).exists():
    try:
        grader_model = tf.keras.models.load_model(GRADER_MODEL_PATH)
        print("✓ Grader model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load grader model: {e}")
        grader_model = None
else:
    print("✗ Grader model not found")
    grader_model = None

# ============================================
# Helper Functions
# ============================================

def preprocess_image(image_bytes):
    """Preprocess image for model input (224x224)"""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None

def identify_with_pretrained_model(image_bytes):
    """
    Use pre-trained MobileNetV2 to identify fruit type
    Returns: (fruit_type, confidence, all_predictions_summary, top_predictions)
    """
    if identifier_model is None:
        return None, 0, {}, []
    
    processed = preprocess_image(image_bytes)
    if processed is None:
        return None, 0, {}, []
    
    try:
        # Get predictions from pre-trained model
        predictions = identifier_model.predict(processed, verbose=0)[0]
        
        # Decode predictions to get class names
        decoded = tf.keras.applications.mobilenet_v2.decode_predictions(
            np.expand_dims(predictions, axis=0), top=10
        )[0]
        
        print(f"🔍 Pre-trained model top predictions:")
        for i, (imagenet_id, label, score) in enumerate(decoded[:5]):
            print(f"   {i+1}: {label} ({score:.2%})")
        
        # Check if top prediction is a supported fruit
        top_label = decoded[0][1].lower()
        top_score = decoded[0][2]
        
        # Map to our fruit types
        fruit_mapping = {
            'apple': 'apples',
            'granny_smith': 'apples',
            'red_delicious': 'apples',
            'crab_apple': 'apples',
            'mango': 'mangos',
            'orange': 'oranges',
            'navel_orange': 'oranges',
            'mandarin': 'oranges',
            'tangerine': 'oranges'
        }
        
        identified_fruit = None
        confidence = 0
        
        for key, fruit in fruit_mapping.items():
            if key in top_label:
                identified_fruit = fruit
                confidence = top_score
                break
        
        # If no match found in mapping, try to see if any fruit keyword appears in top predictions
        if identified_fruit is None:
            fruit_keywords = {
                'apples': ['apple', 'granny', 'red delicious', 'crab apple'],
                'mangos': ['mango'],
                'oranges': ['orange', 'navel', 'mandarin', 'tangerine']
            }
            
            fruit_scores = {'apples': 0, 'mangos': 0, 'oranges': 0}
            
            for _, label, score in decoded[:10]:
                label_lower = label.lower()
                for fruit, keywords in fruit_keywords.items():
                    for keyword in keywords:
                        if keyword in label_lower:
                            fruit_scores[fruit] += score
                            break
            
            # Normalize
            total = sum(fruit_scores.values())
            if total > 0:
                for fruit in fruit_scores:
                    fruit_scores[fruit] /= total
            
            identified_fruit = max(fruit_scores, key=fruit_scores.get)
            confidence = fruit_scores[identified_fruit]
        
        # If confidence is below threshold, reject
        if confidence < IDENTIFICATION_CONFIDENCE_THRESHOLD:
            print(f"⚠ Low confidence ({confidence:.2%}) - treating as unsupported")
            return None, confidence, {}, decoded[:5]
        
        # Check if the identified fruit is supported
        if identified_fruit not in SUPPORTED_FRUITS:
            return None, confidence, {}, decoded[:5]
        
        return identified_fruit, confidence, {}, decoded[:5]
        
    except Exception as e:
        print(f"Identification error: {e}")
        import traceback
        traceback.print_exc()
        return None, 0, {}, []

def get_grade_for_fruit(image_bytes, fruit_type):
    """
    Get grade predictions for a specific fruit using the fine-tuned grader model
    """
    if grader_model is None:
        return None, 0, {}
    
    processed = preprocess_image(image_bytes)
    if processed is None:
        return None, 0, {}
    
    try:
        predictions = grader_model.predict(processed, verbose=0)[0]
        
        grade_confidences = {}
        for grade in GRADES:
            idx = grader_class_names.index(f"{fruit_type}_{grade}")
            grade_confidences[grade] = float(predictions[idx])
        
        predicted_grade = max(grade_confidences, key=grade_confidences.get)
        confidence = grade_confidences[predicted_grade]
        
        return predicted_grade, confidence, grade_confidences
        
    except Exception as e:
        print(f"Grade prediction error: {e}")
        return None, 0, {}

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
    """Two-stage prediction: Identify then Grade"""
    
    if grader_model is None:
        return jsonify({'error': 'Grader model not loaded'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        image_bytes = file.read()
        
        # STEP 1: Identify fruit type
        fruit_type, fruit_confidence, _, top_predictions = identify_with_pretrained_model(image_bytes)
        
        if fruit_type is None:
            detected_desc = "unknown object"
            if top_predictions:
                detected_desc = top_predictions[0][1]
            
            return jsonify({
                'success': False,
                'error': f"Could not identify a supported fruit. Detected: {detected_desc}. Please upload an apple, mango, or orange.",
                'unsupported_fruit': True,
                'detected_fruit': None,
                'top_predictions': [{'label': label, 'confidence': float(score)} for _, label, score in (top_predictions or [])[:3]],
                'supported_fruits': SUPPORTED_FRUITS
            }), 400
        
        if fruit_type not in SUPPORTED_FRUITS:
            return jsonify({
                'success': False,
                'error': f"Unsupported fruit detected: {fruit_type}. Please upload apples, mangoes, or oranges.",
                'unsupported_fruit': True,
                'detected_fruit': fruit_type,
                'supported_fruits': SUPPORTED_FRUITS
            }), 400
        
        print(f"✅ Fruit identified: {fruit_type} (confidence: {fruit_confidence:.2%})")
        
        # STEP 2: Get quality grade
        predicted_grade, grade_confidence, grade_confidences = get_grade_for_fruit(image_bytes, fruit_type)
        
        if predicted_grade is None:
            return jsonify({'error': 'Failed to grade fruit'}), 500
        
        return jsonify({
            'success': True,
            'fruit_type': fruit_type,
            'fruit_identification_confidence': fruit_confidence,
            'grade': predicted_grade,
            'confidence': grade_confidence,
            'grade_confidences': grade_confidences,
            'image_base64': base64.b64encode(image_bytes).decode('utf-8'),
            'processing_time': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 7860))
    print(f"\n🚀 Starting FruitSight API Server")
    print(f"   Port: {port}")
    print(f"   Identifier model: {'Loaded' if identifier_model else 'Not loaded'}")
    print(f"   Grader model: {'Loaded' if grader_model else 'Not loaded'}")
    print("\n" + "=" * 50)
    app.run(host='0.0.0.0', port=port)
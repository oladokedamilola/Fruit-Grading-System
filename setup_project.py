## setup_project.py

"""
Project Setup Script for Fruit Grading System
Automatically creates the directory structure and placeholder files
"""

import os
import sys
import subprocess
from pathlib import Path

def create_directory(path):
    """Create a directory if it doesn't exist"""
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {path}")
    except Exception as e:
        print(f"✗ Error creating directory {path}: {e}")
        return False
    return True

def create_file(path, content=""):
    """Create a file with optional content"""
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Created file: {path}")
        return True
    except Exception as e:
        print(f"✗ Error creating file {path}: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    required_version = (3, 12)
    current_version = sys.version_info[:2]
    
    if current_version >= required_version:
        print(f"✓ Python version {sys.version_info.major}.{sys.version_info.minor} - OK")
        return True
    else:
        print(f"✗ Python {required_version[0]}.{required_version[1]}+ required, found {current_version[0]}.{current_version[1]}")
        return False

def check_dependencies():
    """Check if all required packages can be imported"""
    required_packages = [
        'tensorflow', 'keras', 'cv2', 'PIL', 'numpy', 'pandas',
        'matplotlib', 'seaborn', 'sklearn', 'flask', 'flask_cors',
        'werkzeug', 'dotenv', 'jupyter'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} - installed")
        except ImportError:
            # Handle special cases
            if package == 'cv2':
                try:
                    __import__('cv2')
                    print(f"✓ opencv-python - installed")
                except:
                    missing_packages.append('opencv-python')
            elif package == 'PIL':
                try:
                    __import__('PIL')
                    print(f"✓ Pillow - installed")
                except:
                    missing_packages.append('Pillow')
            elif package == 'dotenv':
                try:
                    __import__('dotenv')
                    print(f"✓ python-dotenv - installed")
                except:
                    missing_packages.append('python-dotenv')
            else:
                missing_packages.append(package)
                print(f"✗ {package} - NOT installed")
    
    if missing_packages:
        print(f"\n⚠ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    return True

def create_directory_structure():
    """Create the complete project directory structure"""
    
    print("\n📁 Creating directory structure...")
    
    directories = [
        # ML directories
        "ml/notebooks",
        "ml/datasets/raw",
        "ml/datasets/processed",
        "ml/datasets/annotations",
        "ml/models",
        "ml/src",
        "ml/outputs/logs",
        "ml/outputs/plots",
        "ml/outputs/metrics",
        
        # Webapp directories
        "webapp/static/css",
        "webapp/static/js",
        "webapp/static/images",
        "webapp/static/uploads",
        "webapp/templates",
        
        # Tests
        "tests",
        
        # Docs
        "docs",
    ]
    
    for directory in directories:
        create_directory(directory)
    
    return True

def create_placeholder_files():
    """Create placeholder files for development"""
    
    print("\n📄 Creating placeholder files...")
    
    # ML source files
    ml_files = {
        "ml/src/__init__.py": "# ML package initialization\n",
        "ml/src/data_preprocessing.py": "\"\"\"\nData preprocessing utilities for fruit grading system\n\"\"\"\n\nimport cv2\nimport numpy as np\nfrom pathlib import Path\n\ndef load_image(image_path, target_size=(224, 224)):\n    \"\"\"Load and preprocess image\"\"\"\n    pass\n\ndef augment_image(image):\n    \"\"\"Apply data augmentation\"\"\"\n    pass\n\ndef create_data_generator():\n    \"\"\"Create data generator for training\"\"\"\n    pass\n",
        "ml/src/model_training.py": "\"\"\"\nModel training utilities for fruit grading system\n\"\"\"\n\nimport tensorflow as tf\nfrom tensorflow import keras\n\ndef create_model(num_classes=3, input_shape=(224, 224, 3)):\n    \"\"\"Create CNN model for fruit grading\"\"\"\n    pass\n\ndef train_model(model, train_data, val_data, epochs=50):\n    \"\"\"Train the fruit grading model\"\"\"\n    pass\n\ndef save_model(model, path):\n    \"\"\"Save trained model\"\"\"\n    pass\n",
        "ml/src/model_evaluation.py": "\"\"\"\nModel evaluation utilities\n\"\"\"\n\nimport numpy as np\nfrom sklearn.metrics import classification_report, confusion_matrix\n\ndef evaluate_model(model, test_data):\n    \"\"\"Evaluate model performance\"\"\"\n    pass\n\ndef plot_confusion_matrix(y_true, y_pred):\n    \"\"\"Create confusion matrix visualization\"\"\"\n    pass\n",
        "ml/src/utils.py": "\"\"\"\nUtility functions for ML pipeline\n\"\"\"\n\nimport os\nimport json\nfrom pathlib import Path\n\ndef save_json(data, filepath):\n    \"\"\"Save data to JSON file\"\"\"\n    pass\n\ndef load_json(filepath):\n    \"\"\"Load data from JSON file\"\"\"\n    pass\n"
    }
    
    # Webapp files
    webapp_files = {
        "webapp/__init__.py": "# Webapp package initialization\n",
        "webapp/app.py": "\"\"\"\nMain Flask application for fruit grading system\n\"\"\"\n\nfrom flask import Flask, render_template, request, jsonify\nfrom flask_cors import CORS\nimport os\nfrom werkzeug.utils import secure_filename\nfrom dotenv import load_dotenv\n\n# Load environment variables\nload_dotenv()\n\napp = Flask(__name__)\nCORS(app)\n\n# Configuration\napp.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'webapp/static/uploads')\napp.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))\napp.config['ALLOWED_EXTENSIONS'] = os.getenv('ALLOWED_EXTENSIONS', '.jpg,.jpeg,.png').split(',')\n\ndef allowed_file(filename):\n    \"\"\"Check if file extension is allowed\"\"\"\n    return '.' in filename and \\\n           '.' + filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']\n\n@app.route('/')\ndef index():\n    \"\"\"Home page\"\"\"\n    return render_template('index.html')\n\n@app.route('/predict', methods=['POST'])\ndef predict():\n    \"\"\"Predict fruit grade from uploaded image\"\"\"\n    # Check if file is present\n    if 'file' not in request.files:\n        return jsonify({'error': 'No file uploaded'}), 400\n    \n    file = request.files['file']\n    \n    # Check if file is selected\n    if file.filename == '':\n        return jsonify({'error': 'No file selected'}), 400\n    \n    # Check if file type is allowed\n    if not allowed_file(file.filename):\n        return jsonify({'error': 'File type not allowed'}), 400\n    \n    # Process and predict\n    # TODO: Implement prediction logic\n    \n    return jsonify({\n        'success': True,\n        'grade': 'A',\n        'confidence': 0.95\n    })\n\n@app.route('/about')\ndef about():\n    \"\"\"About page\"\"\"\n    return render_template('about.html')\n\nif __name__ == '__main__':\n    app.run(debug=True, host='0.0.0.0', port=5000)\n",
        "webapp/model_loader.py": "\"\"\"\nModel loading utilities for web application\n\"\"\"\n\nimport tensorflow as tf\nimport os\nfrom dotenv import load_dotenv\n\nload_dotenv()\n\nclass ModelLoader:\n    \"\"\"Singleton class for loading and managing the fruit grading model\"\"\"\n    \n    _instance = None\n    _model = None\n    \n    def __new__(cls):\n        if cls._instance is None:\n            cls._instance = super(ModelLoader, cls).__new__(cls)\n        return cls._instance\n    \n    def load_model(self):\n        \"\"\"Load the trained model from disk\"\"\"\n        model_path = os.getenv('MODEL_PATH', 'ml/models/fruit_grading_model.h5')\n        \n        if self._model is None:\n            try:\n                self._model = tf.keras.models.load_model(model_path)\n                print(f\"✓ Model loaded from {model_path}\")\n            except Exception as e:\n                print(f\"✗ Error loading model: {e}\")\n                self._model = None\n        \n        return self._model\n    \n    def predict(self, image):\n        \"\"\"Run prediction on preprocessed image\"\"\"\n        if self._model is None:\n            self.load_model()\n        \n        if self._model is not None:\n            return self._model.predict(image)\n        return None\n",
        "webapp/image_processor.py": "\"\"\"\nImage processing utilities for web application\n\"\"\"\n\nimport cv2\nimport numpy as np\nfrom PIL import Image\nimport os\nfrom dotenv import load_dotenv\n\nload_dotenv()\n\nclass ImageProcessor:\n    \"\"\"Handles image preprocessing for fruit grading\"\"\"\n    \n    def __init__(self):\n        self.target_size = (\n            int(os.getenv('IMG_WIDTH', 224)),\n            int(os.getenv('IMG_HEIGHT', 224))\n        )\n    \n    def preprocess(self, image_path):\n        \"\"\"Preprocess image for model input\"\"\"\n        # Load image\n        img = cv2.imread(image_path)\n        if img is None:\n            return None\n        \n        # Convert BGR to RGB\n        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n        \n        # Resize\n        img = cv2.resize(img, self.target_size)\n        \n        # Normalize\n        img = img.astype(np.float32) / 255.0\n        \n        # Add batch dimension\n        img = np.expand_dims(img, axis=0)\n        \n        return img\n    \n    def preprocess_pil(self, pil_image):\n        \"\"\"Preprocess PIL image for model input\"\"\"\n        # Resize\n        pil_image = pil_image.resize(self.target_size)\n        \n        # Convert to array\n        img = np.array(pil_image)\n        \n        # Normalize\n        img = img.astype(np.float32) / 255.0\n        \n        # Add batch dimension\n        img = np.expand_dims(img, axis=0)\n        \n        return img\n",
        "webapp/config.py": "\"\"\"\nConfiguration settings for web application\n\"\"\"\n\nimport os\nfrom dotenv import load_dotenv\n\nload_dotenv()\n\nclass Config:\n    \"\"\"Application configuration\"\"\"\n    \n    # Flask settings\n    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key-change-in-production')\n    DEBUG = os.getenv('FLASK_DEBUG', '1') == '1'\n    \n    # Upload settings\n    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'webapp/static/uploads')\n    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))\n    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}\n    \n    # Model settings\n    MODEL_PATH = os.getenv('MODEL_PATH', 'ml/models/fruit_grading_model.h5')\n    IMG_HEIGHT = int(os.getenv('IMG_HEIGHT', 224))\n    IMG_WIDTH = int(os.getenv('IMG_WIDTH', 224))\n    \n    # Fruit classes\n    FRUIT_TYPES = ['apple', 'mango', 'orange', 'tomato']\n    GRADES = ['A', 'B', 'C']\n    \n    @classmethod\n    def init_app(cls, app):\n        \"\"\"Initialize application with config\"\"\"\n        # Create upload folder if it doesn't exist\n        os.makedirs(cls.UPLOAD_FOLDER, exist_ok=True)\n"
    }
    
    # HTML templates
    template_files = {
        "webapp/templates/index.html": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fruit Grading System</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <div class="container">
        <h1>Fruit Grading System</h1>
        <p>Upload an image of a fruit to get its quality grade</p>
        
        <form id="upload-form" enctype="multipart/form-data">
            <input type="file" id="file-input" name="file" accept="image/*">
            <button type="submit">Grade Fruit</button>
        </form>
        
        <div id="result" style="display: none;">
            <h2>Result</h2>
            <p>Grade: <span id="grade"></span></p>
            <p>Confidence: <span id="confidence"></span>%</p>
        </div>
        
        <div id="loading" style="display: none;">Processing...</div>
        
        <img id="preview" style="max-width: 300px; margin-top: 20px;">
    </div>
    
    <script src="{{ url_for('static', filename='js/main.js') }}"></script>
</body>
</html>
""",
        "webapp/templates/result.html": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Grading Result - Fruit Grading System</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <div class="container">
        <h1>Grading Result</h1>
        
        <div class="result-card">
            <img src="{{ image_url }}" alt="Uploaded fruit" class="result-image">
            
            <div class="result-details">
                <h2>Grade: {{ grade }}</h2>
                <p>Confidence: {{ "%.2f"|format(confidence * 100) }}%</p>
                
                <div class="confidence-bars">
                    {% for label, score in confidence_scores.items() %}
                    <div class="confidence-item">
                        <span>Grade {{ label }}:</span>
                        <div class="bar">
                            <div class="bar-fill" style="width: {{ score * 100 }}%"></div>
                        </div>
                        <span>{{ "%.1f"|format(score * 100) }}%</span>
                    </div>
                    {% endfor %}
                </div>
                
                <a href="/" class="btn">Grade Another Fruit</a>
            </div>
        </div>
    </div>
</body>
</html>
""",
        "webapp/templates/about.html": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>About - Fruit Grading System</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <div class="container">
        <h1>About Fruit Grading System</h1>
        
        <div class="about-content">
            <h2>Project Overview</h2>
            <p>This system uses deep learning to automatically grade fruits based on quality attributes.</p>
            
            <h2>How It Works</h2>
            <ol>
                <li>Upload an image of a fruit</li>
                <li>The system processes the image using a CNN model</li>
                <li>It predicts the quality grade (A, B, or C)</li>
                <li>Results are displayed with confidence scores</li>
            </ol>
            
            <h2>Supported Fruits</h2>
            <ul>
                <li>Mangoes</li>
                <li>Apples</li>
                <li>Oranges</li>
                <li>Tomatoes</li>
            </ul>
            
            <h2>Technology Stack</h2>
            <ul>
                <li>Deep Learning: TensorFlow/Keras</li>
                <li>Web Framework: Flask</li>
                <li>Image Processing: OpenCV, Pillow</li>
                <li>Frontend: HTML5, CSS3, JavaScript</li>
            </ul>
            
            <a href="/" class="btn">Back to Home</a>
        </div>
    </div>
</body>
</html>
"""
    }
    
    # CSS and JS files
    static_files = {
        "webapp/static/css/style.css": """/* Main Styles for Fruit Grading System */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 20px;
}

.container {
    background: white;
    border-radius: 20px;
    padding: 40px;
    max-width: 600px;
    width: 100%;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    text-align: center;
}

h1 {
    color: #333;
    margin-bottom: 10px;
}

h2 {
    color: #555;
    margin: 20px 0;
}

p {
    color: #666;
    margin-bottom: 20px;
}

form {
    margin: 30px 0;
}

input[type="file"] {
    display: block;
    margin: 20px auto;
    padding: 10px;
    border: 2px dashed #ccc;
    border-radius: 10px;
    width: 100%;
    cursor: pointer;
}

button, .btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 12px 30px;
    border-radius: 25px;
    font-size: 16px;
    cursor: pointer;
    transition: transform 0.2s;
    text-decoration: none;
    display: inline-block;
}

button:hover, .btn:hover {
    transform: scale(1.05);
}

#preview {
    max-width: 300px;
    margin-top: 20px;
    border-radius: 10px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}

.result-card {
    text-align: center;
}

.result-image {
    max-width: 300px;
    border-radius: 10px;
    margin: 20px 0;
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}

.confidence-bars {
    margin: 20px 0;
    text-align: left;
}

.confidence-item {
    margin: 10px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}

.bar {
    flex: 1;
    height: 30px;
    background: #f0f0f0;
    border-radius: 15px;
    overflow: hidden;
}

.bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #667eea, #764ba2);
    transition: width 0.5s;
}

#loading {
    margin: 20px 0;
    font-size: 18px;
    color: #667eea;
}

.about-content {
    text-align: left;
}

.about-content h2 {
    margin-top: 20px;
}

.about-content ol, .about-content ul {
    margin-left: 30px;
    margin-bottom: 20px;
}

.about-content li {
    margin: 10px 0;
    color: #666;
}

@media (max-width: 768px) {
    .container {
        padding: 20px;
    }
    
    .confidence-item {
        flex-wrap: wrap;
    }
    
    .bar {
        width: 100%;
        order: 2;
    }
}
""",
        "webapp/static/js/main.js": """// Main JavaScript for Fruit Grading System

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');
    const preview = document.getElementById('preview');
    const result = document.getElementById('result');
    const loading = document.getElementById('loading');
    const gradeSpan = document.getElementById('grade');
    const confidenceSpan = document.getElementById('confidence');
    
    // Preview image when selected
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                preview.src = e.target.result;
                preview.style.display = 'block';
            }
            reader.readAsDataURL(file);
        }
    });
    
    // Handle form submission
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const file = fileInput.files[0];
        if (!file) {
            alert('Please select an image first');
            return;
        }
        
        // Show loading, hide previous result
        loading.style.display = 'block';
        result.style.display = 'none';
        
        // Prepare form data
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            // Send to server
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            // Hide loading
            loading.style.display = 'none';
            
            if (data.success) {
                // Show result
                gradeSpan.textContent = data.grade;
                confidenceSpan.textContent = (data.confidence * 100).toFixed(2);
                result.style.display = 'block';
            } else {
                alert('Error: ' + (data.error || 'Unknown error'));
            }
        } catch (error) {
            loading.style.display = 'none';
            alert('Error uploading file: ' + error.message);
        }
    });
});
"""
    }
    
    # Test files
    test_files = {
        "tests/__init__.py": "# Tests package\n",
        "tests/test_model.py": "\"\"\"\nUnit tests for model components\n\"\"\"\n\nimport unittest\n\nclass TestModel(unittest.TestCase):\n    def test_model_loading(self):\n        \"\"\"Test model loading functionality\"\"\"\n        pass\n    \n    def test_prediction(self):\n        \"\"\"Test model prediction\"\"\"\n        pass\n\nif __name__ == '__main__':\n    unittest.main()\n",
        "tests/test_api.py": "\"\"\"\nIntegration tests for API endpoints\n\"\"\"\n\nimport unittest\n\nclass TestAPI(unittest.TestCase):\n    def test_home_page(self):\n        \"\"\"Test home page loads\"\"\"\n        pass\n    \n    def test_predict_endpoint(self):\n        \"\"\"Test prediction endpoint\"\"\"\n        pass\n\nif __name__ == '__main__':\n    unittest.main()\n"
    }
    
    # Documentation files
    doc_files = {
        "docs/api_documentation.md": "# API Documentation\n\n## Endpoints\n\n### GET /\nHome page with upload form\n\n### POST /predict\nUpload image and get grade prediction\n\n**Request:**\n- Multipart form data with 'file' field containing image\n\n**Response:**\n```json\n{\n    \"success\": true,\n    \"grade\": \"A\",\n    \"confidence\": 0.95,\n    \"confidence_scores\": {\n        \"A\": 0.95,\n        \"B\": 0.04,\n        \"C\": 0.01\n    }\n}\n```\n\n### GET /about\nAbout page with project information\n",
        "docs/user_manual.md": "# User Manual\n\n## Getting Started\n\n1. Open the web application in your browser\n2. Click \"Choose File\" to select an image of a fruit\n3. Click \"Grade Fruit\" to process the image\n4. View the grade and confidence score\n\n## Supported Fruits\n\n- Mangoes\n- Apples\n- Oranges\n- Tomatoes\n\n## Grade Definitions\n\n- **Grade A**: Excellent quality, no defects\n- **Grade B**: Good quality, minor imperfections\n- **Grade C**: Poor quality, significant defects\n"
    }
    
    # Combine all files
    all_files = {**ml_files, **webapp_files, **template_files, **static_files, **test_files, **doc_files}
    
    for filepath, content in all_files.items():
        create_file(filepath, content)
    
    return True

def main():
    """Main setup function"""
    print("=" * 60)
    print("🍎 Fruit Grading System - Project Setup")
    print("=" * 60)
    
    # Check Python version
    print("\n🔍 Checking Python version...")
    if not check_python_version():
        print("\n⚠ Python version requirement not met. Please install Python 3.12+")
        sys.exit(1)
    
    # Create directory structure
    if not create_directory_structure():
        print("\n✗ Failed to create directory structure")
        sys.exit(1)
    
    # Create placeholder files
    if not create_placeholder_files():
        print("\n✗ Failed to create placeholder files")
        sys.exit(1)
    
    # Check dependencies
    print("\n🔍 Checking dependencies...")
    check_dependencies()
    
    print("\n" + "=" * 60)
    print("✅ Project setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Activate virtual environment: python -m venv venv")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Start the web app: python webapp/app.py")
    print("4. Open browser: http://localhost:5000")
    print("\nFor ML development:")
    print("1. Launch Jupyter: jupyter notebook")
    print("2. Navigate to ml/notebooks/")
    print("\nHappy coding! 🚀")

if __name__ == "__main__":
    main()
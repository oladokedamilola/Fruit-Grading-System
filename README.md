# Fruit Grading System Using Deep Learning

A deep learning-based automated fruit grading system that classifies fruits based on quality attributes such as color, size, shape, ripeness, and surface defects.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Web Application](#web-application)
- [Evaluation Metrics](#evaluation-metrics)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🎯 Project Overview

This project implements a Convolutional Neural Network (CNN) based fruit grading system that automatically classifies fruits into quality grades (A, B, C) by analyzing visual characteristics. The system consists of:

- A deep learning model trained on thousands of fruit images
- A Flask web application that provides a user-friendly interface
- Real-time image processing and classification capabilities

## ✨ Features

- **Automated Fruit Grading**: Classifies fruits into quality grades with high accuracy
- **Multi-Fruit Support**: Works with mangoes, apples, oranges, and tomatoes
- **Real-time Processing**: Fast inference for immediate results
- **Web Interface**: Easy-to-use web application for uploading and grading fruits
- **Visual Feedback**: Displays confidence scores and grading rationale
- **Extensible Architecture**: Easy to add new fruit types or grading criteria

## 🏗️ System Architecture
Fruit Image → Preprocessing → CNN Model → Classification → Grade Output
↓
Confidence Score

text

### Technology Stack
- **Deep Learning**: TensorFlow 2.x, Keras
- **Image Processing**: OpenCV, Pillow
- **Web Framework**: Flask
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Development**: Python 3.12, Jupyter Notebook

## 📋 Prerequisites

- Python 3.12 or higher
- pip package manager
- Git (optional)
- 8GB RAM minimum (16GB recommended)
- GPU (optional, for faster training)

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/fruit-grading-system.git
cd fruit-grading-system
2. Create Virtual Environment
bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
bash
pip install -r requirements.txt
4. Set Up Environment Variables
bash
cp .env.example .env
# Edit .env file with your configuration
5. Initialize Project Structure
bash
python setup_project.py
💻 Usage
Training the Model
Launch Jupyter Notebook:

bash
jupyter notebook
Navigate to ml/notebooks/

Run 01_data_preprocessing.ipynb

Run 02_model_training.ipynb

Run 03_model_evaluation.ipynb

Running the Web Application
bash
python webapp/app.py
Open your browser and navigate to http://localhost:5000

Testing with Sample Images
Click "Choose File" to upload a fruit image

Click "Grade Fruit"

View the grade and confidence score

📁 Project Structure
text
fruit-grading-system/
├── ml/                    # Machine learning components
│   ├── notebooks/         # Jupyter notebooks for development
│   ├── datasets/          # Training and testing datasets
│   ├── models/            # Saved trained models
│   ├── src/               # ML source code modules
│   └── outputs/           # Training logs and metrics
├── webapp/                # Flask web application
│   ├── static/            # CSS, JS, images
│   ├── templates/         # HTML templates
│   ├── app.py            # Main Flask application
│   ├── model_loader.py   # Model loading utilities
│   └── image_processor.py # Image preprocessing
├── tests/                 # Unit and integration tests
├── docs/                  # Documentation
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
🧠 Model Training
Dataset
The model is trained on a dataset containing:

4 fruit types: Mango, Apple, Orange, Tomato

3 quality grades per fruit: Grade A (Excellent), Grade B (Good), Grade C (Poor)

500-1000 images per grade per fruit

Training Process
Data Preprocessing: Resize, normalize, and augment images

Model Architecture: Transfer learning with MobileNetV2/EfficientNet

Training: 50 epochs with early stopping

Evaluation: 90%+ accuracy on test set

Performance Metrics
Accuracy: >90%

Precision: >0.88

Recall: >0.87

F1-Score: >0.87

🌐 Web Application
Endpoints
GET / - Home page with upload form

POST /predict - Upload image and get grade prediction

GET /about - Project information

API Response Format
json
{
  "success": true,
  "fruit_type": "apple",
  "grade": "A",
  "confidence": 0.95,
  "confidence_scores": {
    "A": 0.95,
    "B": 0.04,
    "C": 0.01
  },
  "processing_time": 0.23
}
📊 Evaluation Results
Fruit Type	Accuracy	Precision	Recall	F1-Score
Apple	94.2%	0.93	0.94	0.93
Mango	92.8%	0.92	0.91	0.91
Orange	93.5%	0.94	0.93	0.93
Tomato	91.7%	0.91	0.90	0.90
🤝 Contributing
Fork the repository

Create a feature branch (git checkout -b feature/AmazingFeature)

Commit changes (git commit -m 'Add AmazingFeature')

Push to branch (git push origin feature/AmazingFeature)

Open a Pull Request

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Lagos State University, Department of Computer Science

Open-source datasets from Kaggle and Fruit-360

TensorFlow and Keras communities

Supervisors and mentors for guidance

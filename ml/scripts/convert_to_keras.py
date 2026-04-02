# Create ml/scripts/convert_to_keras.py
import tensorflow as tf
from pathlib import Path

# Load the existing model
model = tf.keras.models.load_model('ml/models/fruit_grading_simple_cnn.h5')

# Save in .keras format
model.save('ml/models/fruit_grading_simple_cnn.keras')
print("✅ Model converted to .keras format")
print(f"   Saved to: ml/models/fruit_grading_simple_cnn.keras")
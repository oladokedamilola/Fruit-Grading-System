"""
Download pre-trained MobileNetV2 model during build
"""

import tensorflow as tf
import os

print("=" * 50)
print("Downloading pre-trained MobileNetV2 model...")
print("=" * 50)

# Download and save the model
model = tf.keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=True,
    input_shape=(224, 224, 3)
)

# Save the model weights to a local file (optional, for verification)
print("✓ Model downloaded and loaded successfully")

# Create a marker file to indicate download is complete
with open('/tmp/model_downloaded.txt', 'w') as f:
    f.write('done')

print("=" * 50)
print("Model download complete!")
print("=" * 50)
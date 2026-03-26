#!/usr/bin/env python3
"""
Simplified Training Script for Fruit Grading System
No TensorBoard, minimal dependencies
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def setup_gpu():
    """Configure GPU for training"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ GPU(s) available: {len(gpus)}")
    else:
        print("⚠ No GPU detected. Training will use CPU (slower)")

def load_data():
    """Load dataset splits"""
    annotations_path = Path("ml/datasets/annotations")
    
    train_df = pd.read_csv(annotations_path / "train_split.csv")
    val_df = pd.read_csv(annotations_path / "validation_split.csv")
    test_df = pd.read_csv(annotations_path / "test_split.csv")
    
    print(f"\n📊 Dataset Loaded:")
    print(f"  Training: {len(train_df)} images")
    print(f"  Validation: {len(val_df)} images")
    print(f"  Test: {len(test_df)} images")
    
    return train_df, val_df, test_df

def create_dataset(dataframe, batch_size=32, target_size=(224, 224), shuffle=True):
    """Create TensorFlow dataset from dataframe"""
    
    def load_and_preprocess(image_path, fruit_idx, grade_idx):
        # Load and decode image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, target_size)
        image = tf.cast(image, tf.float32) / 255.0
        
        # Create combined label (fruit * 3 + grade)
        combined_label = fruit_idx * 3 + grade_idx
        label = tf.one_hot(combined_label, depth=12)
        
        return image, label
    
    # Extract data
    image_paths = dataframe['image_path'].values
    fruit_labels = dataframe['fruit_idx'].values
    grade_labels = dataframe['grade_idx'].values
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, fruit_labels, grade_labels))
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def create_model():
    """Create EfficientNetB0 model for fruit grading"""
    # Load pre-trained EfficientNetB0
    base_model = tf.keras.applications.EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False
    
    # Add custom layers
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(12, activation='softmax')
    ])
    
    return model

def main():
    """Main training function"""
    print("=" * 60)
    print("🍎 Fruit Grading System - Simplified Training")
    print("=" * 60)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup
    setup_gpu()
    
    # Load data
    train_df, val_df, test_df = load_data()
    
    # Configuration
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 0.001
    
    print(f"\n⚙️ Training Configuration:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    
    # Create datasets
    print("\n📁 Creating datasets...")
    train_dataset = create_dataset(train_df, BATCH_SIZE, shuffle=True)
    val_dataset = create_dataset(val_df, BATCH_SIZE, shuffle=False)
    test_dataset = create_dataset(test_df, BATCH_SIZE, shuffle=False)
    
    # Create model
    print("\n🏗️ Building model...")
    model = create_model()
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Simple callbacks (no TensorBoard)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='ml/models/fruit_grading_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train model
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60)
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating on Test Set...")
    print("=" * 60)
    
    test_loss, test_acc = model.evaluate(test_dataset, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ml/outputs/training_curves.png', dpi=150)
    plt.show()
    
    # Save training history
    history_dict = {
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss)
    }
    
    with open('ml/outputs/training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    # Save model metadata
    metadata = {
        'model_type': 'EfficientNetB0',
        'input_shape': [224, 224, 3],
        'num_classes': 12,
        'fruit_types': ['apples', 'mangos', 'oranges'],
        'grades': ['A', 'B', 'C'],
        'training_date': datetime.now().isoformat(),
        'test_accuracy': float(test_acc),
        'total_images': len(train_df) + len(val_df) + len(test_df),
        'training_images': len(train_df),
        'validation_images': len(val_df),
        'test_images': len(test_df)
    }
    
    with open('ml/models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"\n📁 Model saved to: ml/models/fruit_grading_model.h5")
    print(f"📊 Results saved to: ml/outputs/")
    print(f"\n📈 Final Test Accuracy: {test_acc*100:.2f}%")
    
    return model, history

if __name__ == "__main__":
    model, history = main()
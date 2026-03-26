#!/usr/bin/env python3
"""
Simplified CNN Training Script for Small Dataset
"""

import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

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

def create_dataset(dataframe, batch_size=16, target_size=(128, 128), shuffle=True, augment=False):
    """Create TensorFlow dataset with optional augmentation"""
    
    def load_and_preprocess(image_path, fruit_idx, grade_idx):
        # Load and decode image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, target_size)
        image = tf.cast(image, tf.float32) / 255.0
        
        # Data augmentation for training
        if augment:
            # Random flip
            image = tf.image.random_flip_left_right(image)
            # Random brightness
            image = tf.image.random_brightness(image, max_delta=0.2)
            # Random contrast
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        
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

def create_simple_cnn():
    """Create a simple CNN model suitable for small datasets"""
    
    model = keras.Sequential([
        # First Conv Block
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.BatchNormalization(),
        
        # Second Conv Block
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.BatchNormalization(),
        
        # Third Conv Block
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.BatchNormalization(),
        
        # Fourth Conv Block
        keras.layers.Conv2D(256, (3, 3), activation='relu'),
        keras.layers.GlobalAveragePooling2D(),
        
        # Dense Layers
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(12, activation='softmax')
    ])
    
    return model

def main():
    """Main training function"""
    print("=" * 60)
    print("🍎 Fruit Grading System - Simple CNN Training")
    print("=" * 60)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    train_df, val_df, test_df = load_data()
    
    # Configuration
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    print(f"\n⚙️ Training Configuration:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Image Size: 128x128")
    
    # Create datasets
    print("\n📁 Creating datasets...")
    train_dataset = create_dataset(train_df, BATCH_SIZE, (128, 128), shuffle=True, augment=True)
    val_dataset = create_dataset(val_df, BATCH_SIZE, (128, 128), shuffle=False, augment=False)
    test_dataset = create_dataset(test_df, BATCH_SIZE, (128, 128), shuffle=False, augment=False)
    
    # Create model
    print("\n🏗️ Building Simple CNN model...")
    model = create_simple_cnn()
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath='ml/models/fruit_grading_simple_cnn.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
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
    plt.savefig('ml/outputs/training_curves_simple.png', dpi=150)
    plt.show()
    
    # Save results
    results = {
        'model_type': 'SimpleCNN',
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'training_epochs': len(history.history['accuracy']),
        'best_val_accuracy': float(max(history.history['val_accuracy']))
    }
    
    with open('ml/outputs/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"\n📁 Model saved to: ml/models/fruit_grading_simple_cnn.h5")
    print(f"📈 Final Test Accuracy: {test_acc*100:.2f}%")
    print(f"🏆 Best Validation Accuracy: {max(history.history['val_accuracy'])*100:.2f}%")
    
    return model, history

if __name__ == "__main__":
    model, history = main()
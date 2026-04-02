#!/usr/bin/env python3
"""
MobileNetV2 Transfer Learning for Fruit Grading
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

def create_dataset(dataframe, batch_size=32, target_size=(224, 224), shuffle=True, augment=False):
    """Create TensorFlow dataset"""
    
    def load_and_preprocess(image_path, fruit_idx, grade_idx):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, target_size)
        image = tf.cast(image, tf.float32) / 255.0
        
        if augment:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        
        combined_label = fruit_idx * 3 + grade_idx
        label = tf.one_hot(combined_label, depth=12)
        
        return image, label
    
    image_paths = dataframe['image_path'].values
    fruit_labels = dataframe['fruit_idx'].values
    grade_labels = dataframe['grade_idx'].values
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, fruit_labels, grade_labels))
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def create_mobilenet_model():
    """Create MobileNetV2 model with transfer learning"""
    
    # Load pre-trained MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False  # Freeze base layers
    
    # Add custom classification head
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(12, activation='softmax')
    ])
    
    return model, base_model

def main():
    """Main training function"""
    print("=" * 60)
    print("🍎 Fruit Grading System - MobileNetV2 Training")
    print("=" * 60)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    train_df, val_df, test_df = load_data()
    
    # Configuration
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 0.001
    
    print(f"\n⚙️ Training Configuration:")
    print(f"  Model: MobileNetV2 (Transfer Learning)")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    
    # Create datasets
    print("\n📁 Creating datasets...")
    train_dataset = create_dataset(train_df, BATCH_SIZE, (224, 224), shuffle=True, augment=True)
    val_dataset = create_dataset(val_df, BATCH_SIZE, (224, 224), shuffle=False, augment=False)
    test_dataset = create_dataset(test_df, BATCH_SIZE, (224, 224), shuffle=False, augment=False)
    
    # Create model
    print("\n🏗️ Building MobileNetV2 model...")
    model, base_model = create_mobilenet_model()
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    print(f"\n📐 Model Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=12,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath='ml/models/fruit_grading_mobilenet.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=6,
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
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_title('MobileNetV2 - Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_title('MobileNetV2 - Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ml/outputs/training_curves_mobilenet.png', dpi=150)
    plt.show()
    
    # Save results
    results = {
        'model': 'MobileNetV2',
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'training_epochs': len(history.history['accuracy'])
    }
    
    with open('ml/outputs/mobilenet_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ MobileNetV2 Training Complete!")
    print("=" * 60)
    print(f"\n📁 Model saved to: ml/models/fruit_grading_mobilenet.h5")
    print(f"📈 Test Accuracy: {test_acc*100:.2f}%")
    print(f"🏆 Best Validation Accuracy: {max(history.history['val_accuracy'])*100:.2f}%")
    
    return model, history

if __name__ == "__main__":
    model, history = main()
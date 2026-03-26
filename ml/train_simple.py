#!/usr/bin/env python3
"""
Main Training Script for Fruit Grading System
Orchestrates the complete training pipeline
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import tensorflow as tf
import pandas as pd
import numpy as np
import json
from datetime import datetime

from ml.src.model_architecture import ModelArchitecture, ModelCompiler, get_model_summary
from ml.src.model_training import TFDatasetGenerator, ModelTrainer
from ml.src.model_evaluation import ModelEvaluator, InferenceOptimizer

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

def setup_gpu():
    """Configure GPU for training"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU(s) available: {len(gpus)}")
            print(f"  - {gpus[0].name}")
        except RuntimeError as e:
            print(f"⚠ GPU configuration error: {e}")
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

def create_datasets(train_df, val_df, test_df, batch_size=32):
    """Create TensorFlow datasets"""
    # Training dataset with augmentation
    train_gen = TFDatasetGenerator(
        train_df, batch_size=batch_size, target_size=(224, 224),
        augment=True, shuffle=True
    )
    train_dataset = train_gen.create_dataset()
    
    # Validation dataset (no augmentation)
    val_gen = TFDatasetGenerator(
        val_df, batch_size=batch_size, target_size=(224, 224),
        augment=False, shuffle=False
    )
    val_dataset = val_gen.create_dataset()
    
    # Test dataset (no augmentation)
    test_gen = TFDatasetGenerator(
        test_df, batch_size=batch_size, target_size=(224, 224),
        augment=False, shuffle=False
    )
    test_dataset = test_gen.create_dataset()
    
    return train_dataset, val_dataset, test_dataset

def main():
    """Main training pipeline"""
    print("=" * 60)
    print("🍎 Fruit Grading System - Model Training Pipeline")
    print("=" * 60)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup GPU
    setup_gpu()
    
    # Load data
    train_df, val_df, test_df = load_data()
    
    # Configuration
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    MODEL_TYPE = "efficientnetb0"  # Options: mobilenetv2, efficientnetb0, resnet50
    
    print(f"\n⚙️ Training Configuration:")
    print(f"  Model: {MODEL_TYPE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    
    # Create datasets
    print("\n📁 Creating datasets...")
    train_dataset, val_dataset, test_dataset = create_datasets(
        train_df, val_df, test_df, batch_size=BATCH_SIZE
    )
    
    # Create model
    print(f"\n🏗️ Building {MODEL_TYPE} model...")
    model_builder = ModelArchitecture()
    
    if MODEL_TYPE == "mobilenetv2":
        model, base_model = model_builder.create_mobilenetv2(dropout_rate=0.3)
    elif MODEL_TYPE == "efficientnetb0":
        model, base_model = model_builder.create_efficientnetb0(dropout_rate=0.3)
    elif MODEL_TYPE == "resnet50":
        model, base_model = model_builder.create_resnet50(dropout_rate=0.3)
    else:
        model, base_model = model_builder.create_custom_cnn()
    
    # Compile model
    model = ModelCompiler.compile_model(model, learning_rate=LEARNING_RATE, optimizer='adam')
    
    # Print model summary
    model_info = get_model_summary(model)
    print(f"\n📐 Model Parameters: {model_info['total_parameters']:,}")
    
    # Setup callbacks
    model_path = Path(f"ml/models/fruit_grading_{MODEL_TYPE}.h5")
    callbacks = ModelCompiler.get_callbacks(model_path, patience=10)
    
    # Train model
    trainer = ModelTrainer(model, output_dir="ml/outputs")
    history = trainer.train(
        train_dataset, val_dataset, 
        epochs=EPOCHS, 
        callbacks=callbacks
    )
    
    # Plot training curves
    trainer.plot_training_history()
    
    # Evaluate model
    print("\n" + "=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    
    evaluator = ModelEvaluator(model, output_dir="ml/outputs")
    results = evaluator.evaluate(test_dataset, test_df)
    
    # Plot per-class metrics
    evaluator.plot_per_class_metrics(results['classification_report'])
    
    # Measure inference time
    print("\n" + "=" * 60)
    print("Inference Optimization")
    print("=" * 60)
    
    # Get a sample batch for timing
    sample_batch = next(iter(test_dataset.take(1)))
    sample_images = sample_batch[0]
    
    optimizer = InferenceOptimizer(model)
    timing = optimizer.measure_inference_time(sample_images)
    
    # Save model in multiple formats
    print("\n💾 Saving models...")
    
    # Save as H5
    trainer.save_model(model_path, format='h5')
    
    # Save as TensorFlow SavedModel
    tf_path = model_path.with_suffix('')
    trainer.save_model(tf_path, format='pb')
    
    # Save as TFLite
    tflite_model = optimizer.export_to_tflite(quantize=True)
    tflite_path = model_path.with_suffix('.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"✓ Model saved as TFLite: {tflite_path}")
    
    # Save model metadata
    metadata = {
        'model_type': MODEL_TYPE,
        'input_shape': [224, 224, 3],
        'num_classes': 12,
        'fruit_types': model_builder.fruit_types,
        'grades': model_builder.grades,
        'training_config': {
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'optimizer': 'adam'
        },
        'performance': {
            'accuracy': float(results['accuracy']),
            'macro_f1': float(results['classification_report']['macro avg']['f1-score']),
            'inference_time_ms': timing['avg_ms']
        },
        'training_date': datetime.now().isoformat(),
        'model_files': {
            'h5': str(model_path),
            'tflite': str(tflite_path),
            'saved_model': str(tf_path)
        }
    }
    
    with open('ml/models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"\n📁 Model saved to: ml/models/")
    print(f"  - fruit_grading_{MODEL_TYPE}.h5 (Keras format)")
    print(f"  - fruit_grading_{MODEL_TYPE}.tflite (TensorFlow Lite)")
    print(f"  - fruit_grading_{MODEL_TYPE}/ (SavedModel)")
    print(f"\n📊 Results saved to: ml/outputs/")
    print(f"  - training_curves.png")
    print(f"  - confusion_matrix.png")
    print(f"  - classification_report.json")
    print(f"  - evaluation_results.json")
    print(f"\n📈 Final Accuracy: {results['accuracy']*100:.2f}%")
    print(f"⚡ Inference Time: {timing['avg_ms']:.2f} ms per image")
    
    return model, results

if __name__ == "__main__":
    model, results = main()
#!/usr/bin/env python3
"""
Hyperparameter Tuning Script for Fruit Grading System
Optimizes model hyperparameters using grid search and Keras Tuner
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import tensorflow as tf
from ml.src.model_architecture import ModelArchitecture, ModelCompiler
from ml.src.model_training import HyperparameterTuner, TFDatasetGenerator
from ml.src.model_evaluation import ModelEvaluator

def load_data():
    """Load dataset splits"""
    annotations_path = Path("ml/datasets/annotations")
    train_df = pd.read_csv(annotations_path / "train_split.csv")
    val_df = pd.read_csv(annotations_path / "validation_split.csv")
    return train_df, val_df

def main():
    """Run hyperparameter tuning"""
    print("=" * 60)
    print("🔧 Hyperparameter Tuning")
    print("=" * 60)
    
    # Load data
    train_df, val_df = load_data()
    
    # Create datasets
    train_gen = TFDatasetGenerator(
        train_df, batch_size=32, target_size=(224, 224),
        augment=True, shuffle=True
    )
    train_dataset = train_gen.create_dataset()
    
    val_gen = TFDatasetGenerator(
        val_df, batch_size=32, target_size=(224, 224),
        augment=False, shuffle=False
    )
    val_dataset = val_gen.create_dataset()
    
    # Define model builder for tuning
    def model_builder(dropout_rate=0.3):
        arch = ModelArchitecture()
        model, _ = arch.create_efficientnetb0(dropout_rate=dropout_rate)
        return model
    
    # Manual grid search
    print("\n🔍 Running Manual Grid Search...")
    
    param_grid = {
        'learning_rate': [0.0001, 0.001, 0.01],
        'dropout_rate': [0.2, 0.3, 0.4],
        'batch_size': [16, 32, 64],
        'optimizer': ['adam']
    }
    
    tuner = HyperparameterTuner(model_builder)
    best_params = tuner.manual_grid_search(train_dataset, val_dataset, param_grid)
    
    print("\n" + "=" * 60)
    print("Best Hyperparameters Found:")
    print("=" * 60)
    for param, value in best_params.items():
        print(f"  {param}: {value}")

if __name__ == "__main__":
    main()
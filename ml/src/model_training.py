"""
Model Training Module for Fruit Grading System
Handles training pipeline with data generators
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

class DataGenerator:
    """Custom data generator with on-the-fly augmentation"""
    
    def __init__(self, dataframe, batch_size=32, target_size=(224, 224), 
                 augment=False, shuffle=True):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.target_size = target_size
        self.augment = augment
        self.shuffle = shuffle
        self.indices = np.arange(len(dataframe))
        
        if shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_df = self.dataframe.iloc[batch_indices]
        
        images = []
        fruit_labels = []
        grade_labels = []
        
        for _, row in batch_df.iterrows():
            # Load and preprocess image
            img = tf.keras.preprocessing.image.load_img(
                row['image_path'],
                target_size=self.target_size
            )
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = img / 255.0  # Normalize
            
            # Apply augmentation if enabled
            if self.augment:
                img = self._augment_image(img)
            
            images.append(img)
            fruit_labels.append(row['fruit_idx'])
            grade_labels.append(row['grade_idx'])
        
        images = np.array(images)
        
        # Create combined label (fruit * 3 + grade)
        combined_labels = [fruit * 3 + grade for fruit, grade in zip(fruit_labels, grade_labels)]
        combined_labels = tf.keras.utils.to_categorical(combined_labels, num_classes=12)
        
        return images, combined_labels
    
    def _augment_image(self, image):
        """Apply augmentation to image"""
        # Random flip
        if np.random.random() > 0.5:
            image = np.fliplr(image)
        
        # Random rotation
        if np.random.random() > 0.5:
            angle = np.random.uniform(-20, 20)
            image = tf.keras.preprocessing.image.apply_affine_transform(
                image, theta=angle, fill_mode='nearest'
            )
        
        # Random brightness
        if np.random.random() > 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            image = image * brightness
        
        # Random contrast
        if np.random.random() > 0.5:
            contrast = np.random.uniform(0.8, 1.2)
            mean = np.mean(image)
            image = (image - mean) * contrast + mean
        
        # Clip values
        image = np.clip(image, 0, 1)
        
        return image

class TFDatasetGenerator:
    """TensorFlow dataset generator for efficient training"""
    
    def __init__(self, dataframe, batch_size=32, target_size=(224, 224), 
                 augment=False, shuffle=True):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.target_size = target_size
        self.augment = augment
        self.shuffle = shuffle
        
    def create_dataset(self):
        """Create TensorFlow dataset"""
        # Create dataset from dataframe
        dataset = tf.data.Dataset.from_tensor_slices((
            self.dataframe['image_path'].values,
            self.dataframe['fruit_idx'].values,
            self.dataframe['grade_idx'].values
        ))
        
        # Map function to load and preprocess images
        def load_and_preprocess(image_path, fruit_idx, grade_idx):
            # Load image
            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, self.target_size)
            image = tf.cast(image, tf.float32) / 255.0
            
            # Apply augmentation
            if self.augment:
                image = self._augment_image(image)
            
            # Create combined label
            combined_label = fruit_idx * 3 + grade_idx
            label = tf.one_hot(combined_label, depth=12)
            
            return image, label
        
        dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _augment_image(self, image):
        """Apply augmentation to image"""
        # Random flip
        image = tf.image.random_flip_left_right(image)
        
        # Random brightness
        image = tf.image.random_brightness(image, max_delta=0.2)
        
        # Random contrast
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        
        # Random saturation
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        
        # Random hue
        image = tf.image.random_hue(image, max_delta=0.1)
        
        return image

class ModelTrainer:
    """Handles model training with monitoring and logging"""
    
    def __init__(self, model, output_dir="ml/outputs"):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history = None
        self.start_time = None
        
    def train(self, train_dataset, val_dataset, epochs=50, callbacks=None):
        """
        Train the model
        """
        print("\n" + "=" * 60)
        print("Starting Model Training")
        print("=" * 60)
        
        self.start_time = time.time()
        
        # Train the model
        self.history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks or [],
            verbose=1
        )
        
        # Calculate training time
        training_time = time.time() - self.start_time
        print(f"\n✓ Training completed in {training_time:.2f} seconds")
        
        # Save training history
        self._save_history()
        
        return self.history
    
    def _save_history(self):
        """Save training history to JSON"""
        history_dict = {
            'loss': self.history.history['loss'],
            'accuracy': self.history.history['accuracy'],
            'val_loss': self.history.history['val_loss'],
            'val_accuracy': self.history.history['val_accuracy'],
            'top_k_accuracy': self.history.history.get('top_k_categorical_accuracy', []),
            'val_top_k_accuracy': self.history.history.get('val_top_k_categorical_accuracy', [])
        }
        
        # Save as JSON
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        print(f"✓ Training history saved to {self.output_dir / 'training_history.json'}")
    
    def plot_training_history(self):
        """Plot training curves"""
        if not self.history:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        axes[0].plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot accuracy
        axes[1].plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[1].plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Training curves saved to {self.output_dir / 'training_curves.png'}")
    
    def save_model(self, model_path, format='h5'):
        """
        Save trained model in specified format
        """
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'h5':
            self.model.save(str(model_path))
            print(f"✓ Model saved as H5: {model_path}")
        
        elif format == 'pb':
            self.model.save(str(model_path), save_format='tf')
            print(f"✓ Model saved as TensorFlow SavedModel: {model_path}")
        
        elif format == 'tflite':
            # Convert to TensorFlow Lite
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            tflite_model = converter.convert()
            
            tflite_path = model_path.with_suffix('.tflite')
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            print(f"✓ Model saved as TFLite: {tflite_path}")
        
        else:
            print(f"Unknown format: {format}")

class HyperparameterTuner:
    """Hyperparameter tuning with Keras Tuner"""
    
    def __init__(self, model_builder, output_dir="ml/outputs/tuning"):
        self.model_builder = model_builder
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def tune_with_keras_tuner(self, train_dataset, val_dataset, max_trials=10):
        """
        Use Keras Tuner for hyperparameter optimization
        """
        try:
            import kerastuner as kt
            
            def build_model(hp):
                model = self.model_builder(
                    hp.Choice('learning_rate', [1e-4, 1e-3, 1e-2]),
                    hp.Choice('dropout_rate', [0.2, 0.3, 0.4, 0.5]),
                    hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop'])
                )
                return model
            
            tuner = kt.RandomSearch(
                build_model,
                objective='val_accuracy',
                max_trials=max_trials,
                directory=str(self.output_dir),
                project_name='fruit_grading_tuning'
            )
            
            tuner.search(train_dataset, validation_data=val_dataset, epochs=10)
            
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            
            print("\nBest hyperparameters found:")
            print(f"  Learning rate: {best_hps.get('learning_rate')}")
            print(f"  Dropout rate: {best_hps.get('dropout_rate')}")
            print(f"  Optimizer: {best_hps.get('optimizer')}")
            
            return best_hps
            
        except ImportError:
            print("Keras Tuner not installed. Run: pip install keras-tuner")
            return None
    
    def manual_grid_search(self, train_dataset, val_dataset, param_grid):
        """
        Manual grid search for hyperparameters
        """
        results = []
        
        total_combinations = len(param_grid['learning_rate']) * \
                            len(param_grid['dropout_rate']) * \
                            len(param_grid['batch_size'])
        
        print(f"\n🔍 Manual Grid Search - {total_combinations} combinations")
        
        for lr in param_grid['learning_rate']:
            for dropout in param_grid['dropout_rate']:
                for batch_size in param_grid['batch_size']:
                    print(f"\nTesting: LR={lr}, Dropout={dropout}, Batch={batch_size}")
                    
                    # Create model with these parameters
                    model, _ = self.model_builder(dropout_rate=dropout)
                    
                    # Compile model
                    if param_grid.get('optimizer', 'adam') == 'adam':
                        opt = tf.keras.optimizers.Adam(learning_rate=lr)
                    else:
                        opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
                    
                    model.compile(
                        optimizer=opt,
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    
                    # Create dataset with batch size
                    # (Recreate dataset with new batch size)
                    
                    # Train for few epochs
                    history = model.fit(
                        train_dataset,
                        validation_data=val_dataset,
                        epochs=5,
                        verbose=0
                    )
                    
                    results.append({
                        'learning_rate': lr,
                        'dropout_rate': dropout,
                        'batch_size': batch_size,
                        'val_accuracy': history.history['val_accuracy'][-1],
                        'val_loss': history.history['val_loss'][-1]
                    })
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_dir / 'grid_search_results.csv', index=False)
        
        # Find best combination
        best = results_df.loc[results_df['val_accuracy'].idxmax()]
        
        print("\n" + "=" * 60)
        print("Grid Search Results")
        print("=" * 60)
        print(f"Best validation accuracy: {best['val_accuracy']:.4f}")
        print(f"Best parameters:")
        print(f"  Learning rate: {best['learning_rate']}")
        print(f"  Dropout rate: {best['dropout_rate']}")
        print(f"  Batch size: {best['batch_size']:.0f}")
        
        return best.to_dict()
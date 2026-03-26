"""
Model Evaluation Module for Fruit Grading System
Generates performance metrics and visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json
from pathlib import Path
import tensorflow as tf
from tqdm import tqdm

class ModelEvaluator:
    """Evaluates model performance on test data"""
    
    def __init__(self, model, output_dir="ml/outputs"):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.fruit_types = ['apple', 'mango', 'orange', 'tomato']
        self.grades = ['A', 'B', 'C']
        self.class_names = [f"{fruit}_grade_{grade}" for fruit in self.fruit_types for grade in self.grades]
    
    def evaluate(self, test_dataset, test_df):
        """
        Evaluate model on test dataset
        """
        print("\n" + "=" * 60)
        print("Model Evaluation on Test Set")
        print("=" * 60)
        
        # Get predictions
        y_true = []
        y_pred = []
        
        for images, labels in tqdm(test_dataset, desc="Evaluating"):
            predictions = self.model.predict(images, verbose=0)
            y_true.extend(np.argmax(labels.numpy(), axis=1))
            y_pred.extend(np.argmax(predictions, axis=1))
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Classification report
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        
        # Save classification report
        with open(self.output_dir / 'classification_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\nClassification Report (Macro Avg):")
        print(f"  Precision: {report['macro avg']['precision']:.4f}")
        print(f"  Recall: {report['macro avg']['recall']:.4f}")
        print(f"  F1-Score: {report['macro avg']['f1-score']:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Save results
        self._save_evaluation_results(accuracy, report, cm)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'y_true': y_true,
            'y_pred': y_pred
        }
    
    def _save_evaluation_results(self, accuracy, report, cm):
        """Save evaluation results to files"""
        results = {
            'accuracy': float(accuracy),
            'macro_precision': float(report['macro avg']['precision']),
            'macro_recall': float(report['macro avg']['recall']),
            'macro_f1': float(report['macro avg']['f1-score']),
            'per_class_metrics': {
                class_name: {
                    'precision': float(report[class_name]['precision']),
                    'recall': float(report[class_name]['recall']),
                    'f1-score': float(report[class_name]['f1-score'])
                }
                for class_name in self.class_names
            }
        }
        
        with open(self.output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save confusion matrix
        np.save(self.output_dir / 'confusion_matrix.npy', cm)
        
        print(f"\n✓ Evaluation results saved to {self.output_dir}")
    
    def plot_confusion_matrix(self, cm, normalize=False):
        """
        Plot confusion matrix
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Confusion matrix saved to {self.output_dir / 'confusion_matrix.png'}")
    
    def plot_per_class_metrics(self, report):
        """
        Plot per-class precision, recall, F1-score
        """
        classes = list(report.keys())[:-3]  # Exclude avg metrics
        precisions = [report[c]['precision'] for c in classes]
        recalls = [report[c]['recall'] for c in classes]
        f1_scores = [report[c]['f1-score'] for c in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.bar(x - width, precisions, width, label='Precision', color='#2ecc71')
        ax.bar(x, recalls, width, label='Recall', color='#3498db')
        ax.bar(x + width, f1_scores, width, label='F1-Score', color='#e74c3c')
        
        ax.set_xlabel('Classes', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'per_class_metrics.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Per-class metrics saved to {self.output_dir / 'per_class_metrics.png'}")

class InferenceOptimizer:
    """Optimizes model for inference"""
    
    def __init__(self, model):
        self.model = model
    
    def measure_inference_time(self, test_images, num_iterations=100):
        """
        Measure inference time per image
        """
        import time
        
        # Warm-up
        for _ in range(10):
            _ = self.model.predict(test_images[:1], verbose=0)
        
        # Measure
        times = []
        for i in range(num_iterations):
            start = time.time()
            _ = self.model.predict(test_images[i % len(test_images):(i % len(test_images)) + 1], verbose=0)
            times.append(time.time() - start)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"\n⚡ Inference Time Analysis:")
        print(f"  Average: {avg_time*1000:.2f} ms per image")
        print(f"  Std Dev: {std_time*1000:.2f} ms")
        print(f"  FPS: {1/avg_time:.2f} images/second")
        
        return {'avg_ms': avg_time * 1000, 'std_ms': std_time * 1000, 'fps': 1/avg_time}
    
    def quantize_model(self, representative_dataset):
        """
        Apply post-training quantization
        """
        print("\n📦 Quantizing model...")
        
        def representative_data_gen():
            for images, _ in representative_dataset.take(100):
                yield [images]
        
        # Dynamic range quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_quantized = converter.convert()
        
        print("✓ Quantization complete")
        return tflite_quantized
    
    def prune_model(self, x_train, y_train, x_val, y_val):
        """
        Apply model pruning
        """
        try:
            import tensorflow_model_optimization as tfmot
            
            print("\n✂️ Pruning model...")
            
            # Define pruning schedule
            pruning_params = {
                'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                    initial_sparsity=0.0,
                    final_sparsity=0.5,
                    begin_step=2000,
                    end_step=10000
                )
            }
            
            # Apply pruning
            pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
                self.model, **pruning_params
            )
            
            # Compile pruned model
            pruned_model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Add pruning callbacks
            callbacks = [
                tfmot.sparsity.keras.UpdatePruningStep(),
                tfmot.sparsity.keras.PruningSummaries(log_dir='ml/outputs/logs')
            ]
            
            print("✓ Pruning setup complete")
            return pruned_model, callbacks
            
        except ImportError:
            print("TensorFlow Model Optimization not installed. Run: pip install tensorflow-model-optimization")
            return self.model, []
    
    def export_to_tflite(self, quantize=False):
        """
        Export model to TensorFlow Lite format
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        return tflite_model
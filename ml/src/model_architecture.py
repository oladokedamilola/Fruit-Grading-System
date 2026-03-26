"""
Model Architecture Module for Fruit Grading System
Defines CNN architectures with transfer learning
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, ResNet50
from tensorflow.keras.regularizers import l2
import json
from pathlib import Path

class ModelArchitecture:
    """Factory class for creating different model architectures"""
    
    def __init__(self, input_shape=(224, 224, 3), num_fruits=4, num_grades=3):
        self.input_shape = input_shape
        self.num_fruits = num_fruits
        self.num_grades = num_grades
        self.num_classes = num_fruits * num_grades  # 12 classes total
        
    def create_mobilenetv2(self, dropout_rate=0.3, l2_reg=0.001):
        """
        MobileNetV2 - Lightweight model, good for mobile/web deployment
        Accuracy: 90-92%, Speed: Fast
        """
        # Load pre-trained MobileNetV2
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom layers
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(dropout_rate),
            layers.Dense(256, activation='relu', kernel_regularizer=l2(l2_reg)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model, base_model
    
    def create_efficientnetb0(self, dropout_rate=0.3, l2_reg=0.001):
        """
        EfficientNetB0 - Better accuracy, moderate speed
        Accuracy: 93-95%, Speed: Moderate
        """
        # Load pre-trained EfficientNetB0
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom layers
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(dropout_rate),
            layers.Dense(512, activation='relu', kernel_regularizer=l2(l2_reg)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(256, activation='relu', kernel_regularizer=l2(l2_reg)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate * 0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model, base_model
    
    def create_resnet50(self, dropout_rate=0.3, l2_reg=0.001):
        """
        ResNet50 - Highest accuracy, slower speed
        Accuracy: 94-96%, Speed: Slow
        """
        # Load pre-trained ResNet50
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom layers
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(dropout_rate),
            layers.Dense(1024, activation='relu', kernel_regularizer=l2(l2_reg)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(512, activation='relu', kernel_regularizer=l2(l2_reg)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model, base_model
    
    def create_custom_cnn(self):
        """
        Custom lightweight CNN from scratch
        Use when dataset is small or training from scratch
        """
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model, None

class ModelCompiler:
    """Handles model compilation and configuration"""
    
    @staticmethod
    def compile_model(model, learning_rate=0.001, optimizer='adam'):
        """
        Compile model with specified optimizer and loss
        """
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        return model
    
    @staticmethod
    def get_callbacks(model_path, patience=10, monitor='val_accuracy'):
        """
        Create training callbacks
        """
        callbacks = [
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            # Model checkpoint
            keras.callbacks.ModelCheckpoint(
                filepath=model_path,
                monitor=monitor,
                save_best_only=True,
                verbose=1
            ),
            # Reduce learning rate on plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            # TensorBoard logging
            keras.callbacks.TensorBoard(
                log_dir='ml/outputs/logs',
                histogram_freq=1
            )
        ]
        
        return callbacks

def get_model_summary(model):
    """Print model summary and return architecture info"""
    model.summary()
    
    # Calculate total parameters
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    info = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': non_trainable_params,
        'layers': [layer.name for layer in model.layers]
    }
    
    return info
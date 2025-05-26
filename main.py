#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Age Classification Model using ResNet50V2
Classifies images into Adults, Teenagers, and Toddler categories
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import regularizers
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgeClassifier:
    def __init__(self, data_dir, img_size=(224, 224), batch_size=32):
        """
        Initialize the Age Classifier
        
        Args:
            data_dir (str): Path to data directory containing train, validate, test_out folders
            img_size (tuple): Input image size for the model
            batch_size (int): Batch size for training
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        
        # Set up paths
        self.train_path = os.path.join(data_dir, "train")
        self.validate_path = os.path.join(data_dir, "validate") 
        self.test_path = os.path.join(data_dir, "test_out")
        
        # Initialize label encoder
        self.label_encoder = LabelEncoder()
        self.classes = ['Adults', 'Teenagers', 'Toddler']
        self.label_encoder.fit(self.classes)
        
        # Set seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
        
        logger.info("Age Classifier initialized")
    
    def verify_data_structure(self):
        """Verify that all required directories exist"""
        required_paths = [self.train_path, self.validate_path, self.test_path]
        
        for path in required_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required directory not found: {path}")
        
        logger.info("Data directory structure verified")
        
        # Log dataset statistics
        for split, path in [("Train", self.train_path), ("Validation", self.validate_path)]:
            if os.path.exists(path):
                total_images = sum([len(files) for r, d, files in os.walk(path) if files])
                logger.info(f"{split} dataset contains {total_images} images")
    
    def create_data_generators(self):
        """Create data generators for training, validation, and testing"""
        
        # Enhanced data augmentation for training
        train_datagen = ImageDataGenerator(
            rotation_range=25,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.15,
            zoom_range=0.15,
            horizontal_flip=True,
            brightness_range=[0.85, 1.15],
            channel_shift_range=0.1,
            fill_mode='nearest',
            preprocessing_function=preprocess_input
        )
        
        # No augmentation for validation and test
        valid_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        
        # Create generators
        self.train_generator = train_datagen.flow_from_directory(
            self.train_path,
            target_size=self.img_size,
            class_mode='categorical',
            batch_size=self.batch_size,
            seed=42,
            shuffle=True
        )
        
        self.validation_generator = valid_test_datagen.flow_from_directory(
            self.validate_path,
            target_size=self.img_size,
            class_mode='categorical',
            batch_size=self.batch_size,
            seed=42,
            shuffle=False
        )
        
        self.test_generator = valid_test_datagen.flow_from_directory(
            self.test_path,
            target_size=self.img_size,
            class_mode=None,
            batch_size=1,
            shuffle=False
        )
        
        # Calculate steps
        self.steps_per_epoch = max(1, self.train_generator.samples // self.batch_size)
        self.validation_steps = max(1, self.validation_generator.samples // self.batch_size)
        
        logger.info(f"Data generators created - Steps per epoch: {self.steps_per_epoch}, Validation steps: {self.validation_steps}")
        logger.info(f"Class indices: {self.train_generator.class_indices}")
    
    def build_model(self, fine_tune_layers=100):
        """
        Build the model using ResNet50V2 as base
        
        Args:
            fine_tune_layers (int): Number of layers to fine-tune from the end
        """
        
        # Load pre-trained ResNet50V2
        base_model = ResNet50V2(
            include_top=False,
            weights='imagenet',
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Add custom classification head
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        x = Dropout(0.4)(x)
        x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        x = Dropout(0.3)(x)
        predictions = Dense(3, activation='softmax', name='predictions')(x)
        
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        logger.info("Model built and compiled")
        logger.info(f"Model has {self.model.count_params():,} parameters")
    
    def train_model(self, epochs=30, patience=7):
        """
        Train the model with callbacks
        
        Args:
            epochs (int): Maximum number of epochs
            patience (int): Early stopping patience
        """
        
        # Create callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        logger.info("Starting initial training...")
        
        # Initial training with frozen base
        initial_history = self.model.fit(
            self.train_generator,
            epochs=min(10, epochs),
            steps_per_epoch=self.steps_per_epoch,
            validation_data=self.validation_generator,
            validation_steps=self.validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        # Fine-tuning phase
        logger.info("Starting fine-tuning phase...")
        
        # Unfreeze top layers of base model
        self.model.layers[0].trainable = True
        for layer in self.model.layers[0].layers[:-30]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        # Continue training
        fine_tune_history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            initial_epoch=len(initial_history.history['loss']),
            steps_per_epoch=self.steps_per_epoch,
            validation_data=self.validation_generator,
            validation_steps=self.validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        # Combine histories
        self.history = self._combine_histories(initial_history, fine_tune_history)
        logger.info("Training completed")
    
    def _combine_histories(self, hist1, hist2):
        """Combine two training histories"""
        combined = {}
        for key in hist1.history.keys():
            combined[key] = hist1.history[key] + hist2.history[key]
        
        class CombinedHistory:
            def __init__(self, history_dict):
                self.history = history_dict
        
        return CombinedHistory(combined)
    
    def plot_training_history(self, save_path='training_history.png'):
        """Plot and save training history"""
        if self.history is None:
            logger.warning("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16)
        
        # Accuracy plot
        axes[0, 0].plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[0, 1].plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate plot
        if 'lr' in self.history.history:
            axes[1, 0].plot(self.history.history['lr'], linewidth=2)
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Top-k accuracy plot
        if 'top_k_categorical_accuracy' in self.history.history:
            axes[1, 1].plot(self.history.history['top_k_categorical_accuracy'], 
                          label='Train Top-K Accuracy', linewidth=2)
            axes[1, 1].plot(self.history.history['val_top_k_categorical_accuracy'], 
                          label='Val Top-K Accuracy', linewidth=2)
            axes[1, 1].set_title('Top-K Categorical Accuracy')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        logger.info(f"Training history saved to {save_path}")
    
    def predict_test_set(self, output_file='test.csv'):
        """
        Make predictions on test set and save to CSV
        
        Args:
            output_file (str): Name of output CSV file
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        logger.info("Making predictions on test set...")
        
        # Reset test generator
        self.test_generator.reset()
        
        # Make predictions
        predictions = self.model.predict(
            self.test_generator,
            steps=len(self.test_generator),
            verbose=1
        )
        
        # Get predicted classes
        predicted_classes = np.argmax(predictions, axis=1)
        predicted_categories = self.label_encoder.inverse_transform(predicted_classes)
        
        # Get confidence scores
        confidence_scores = np.max(predictions, axis=1)
        
        # Get filenames
        filenames = [os.path.basename(f) for f in self.test_generator.filenames]
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'filename': filenames,
            'predicted_category': predicted_categories,
            'confidence': confidence_scores
        })
        
        # Add individual class probabilities
        for i, class_name in enumerate(self.classes):
            results_df[f'prob_{class_name.lower()}'] = predictions[:, i]
        
        # Sort by filename for consistency
        results_df = results_df.sort_values('filename').reset_index(drop=True)
        
        # Save to CSV
        results_df.to_csv(output_file, index=False)
        logger.info(f"Predictions saved to {output_file}")
        
        # Print summary
        print(f"\nPrediction Summary:")
        print(f"Total test images: {len(results_df)}")
        print(f"Average confidence: {confidence_scores.mean():.3f}")
        print(f"\nPrediction distribution:")
        print(results_df['predicted_category'].value_counts())
        
        return results_df
    
    def evaluate_model(self):
        """Evaluate model on validation set"""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        logger.info("Evaluating model on validation set...")
        evaluation = self.model.evaluate(
            self.validation_generator,
            steps=self.validation_steps,
            verbose=1
        )
        
        metrics = dict(zip(self.model.metrics_names, evaluation))
        logger.info(f"Validation metrics: {metrics}")
        return metrics


def main():
    """Main execution function"""
    
    # Configuration
    DATA_DIR = "/content/drive/My Drive/Kaggle/data"  # Update this path as needed
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 25
    
    try:
        # Initialize classifier
        classifier = AgeClassifier(
            data_dir=DATA_DIR,
            img_size=IMG_SIZE,
            batch_size=BATCH_SIZE
        )
        
        # Verify data structure
        classifier.verify_data_structure()
        
        # Create data generators
        classifier.create_data_generators()
        
        # Build model
        classifier.build_model()
        
        # Train model
        classifier.train_model(epochs=EPOCHS)
        
        # Plot training history
        classifier.plot_training_history()
        
        # Evaluate model
        classifier.evaluate_model()
        
        # Make predictions and save to CSV
        results_df = classifier.predict_test_set('test.csv')
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    main()
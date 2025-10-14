"""
Deep Learning Model for Multi-Lingual Lip Reading
Combines 3D CNN, Bidirectional LSTM, and Attention mechanisms
Author: AI Assistant
Date: October 2025
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np


class AttentionLayer(layers.Layer):
    """Custom attention layer for temporal sequence"""
    
    def __init__(self, units=128, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        self.W = None
        self.b = None
        self.u = None
    
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_W',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_b',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_u',
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        # Compute attention scores
        uit = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(ait, axis=1)
        attention_weights = tf.expand_dims(attention_weights, -1)
        
        # Apply attention weights
        weighted_input = x * attention_weights
        output = tf.reduce_sum(weighted_input, axis=1)
        
        return output
    
    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({'units': self.units})
        return config


class LipReadingModel:
    """Multi-lingual lip reading model with Bidirectional LSTM + Attention for geometric features"""
    
    def __init__(self,
                 num_classes,
                 sequence_length=75,
                 num_features=None,
                 frame_height=100,
                 frame_width=100,
                 channels=3,
                 lstm_units=[256, 128],
                 dense_units=[512, 256],
                 dropout_rate=0.5):
        """
        Initialize the lip reading model
        
        Args:
            num_classes: Number of output classes (words across all languages)
            sequence_length: Number of frames in input sequence
            num_features: Number of geometric features per frame (computed automatically if None)
            frame_height: Height of each frame (deprecated for geometric features, kept for compatibility)
            frame_width: Width of each frame (deprecated for geometric features, kept for compatibility)
            channels: Number of color channels (deprecated for geometric features, kept for compatibility)
            lstm_units: List of LSTM layer units
            dense_units: List of dense layer units
            dropout_rate: Dropout rate for regularization
        """
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.num_features = num_features  # Will be set dynamically if None
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.channels = channels
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        
        self.model = None
        self.history = None
    
    def build_model(self, num_features=None):
        """
        Build the complete model architecture for geometric features
        
        Args:
            num_features: Number of features per timestep (auto-detected if None)
        """
        if num_features is not None:
            self.num_features = num_features
        
        if self.num_features is None:
            # Default feature count (will be updated dynamically)
            # Estimate: 20 outer lip points * 2 (x,y) + 11 inner lip points * 2 + other features
            # Plus velocity and acceleration features (3x multiplier)
            self.num_features = 200  # Placeholder, will be auto-detected from data
        
        # Input layer for geometric features
        input_shape = (self.sequence_length, self.num_features)
        inputs = layers.Input(shape=input_shape, name='feature_input')
        
        # Dense layers to process features
        x = layers.Dense(256, activation='relu', name='feature_dense_1')(inputs)
        x = layers.BatchNormalization(name='bn_1')(x)
        x = layers.Dropout(0.3, name='dropout_1')(x)
        
        x = layers.Dense(128, activation='relu', name='feature_dense_2')(x)
        x = layers.BatchNormalization(name='bn_2')(x)
        x = layers.Dropout(0.3, name='dropout_2')(x)
        
        # Bidirectional LSTM layers for temporal modeling
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            x = layers.Bidirectional(
                layers.LSTM(units, return_sequences=return_sequences, dropout=0.3),
                name=f'bilstm_{i+1}'
            )(x)
            x = layers.Dropout(self.dropout_rate, name=f'dropout_lstm_{i+1}')(x)
        
        # Dense layers for classification
        for i, units in enumerate(self.dense_units):
            x = layers.Dense(units, activation='relu', name=f'dense_{i+1}')(x)
            x = layers.Dropout(self.dropout_rate, name=f'dropout_dense_{i+1}')(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs, name='LipReadingModel_Geometric')
        
        return self.model
    
    def build_model_with_attention(self, num_features=None):
        """Build model with attention mechanism for geometric features"""
        
        if num_features is not None:
            self.num_features = num_features
        
        if self.num_features is None:
            self.num_features = 200  # Placeholder
        
        # Input layer for geometric features
        input_shape = (self.sequence_length, self.num_features)
        inputs = layers.Input(shape=input_shape, name='feature_input')
        
        # Dense layers to process features
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Bidirectional LSTM with return_sequences=True for attention
        x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.3))(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.3))(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Attention layer
        x = AttentionLayer(128)(x)
        
        # Dense layers
        for units in self.dense_units:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(self.dropout_rate)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name='LipReadingModel_Geometric_Attention')
        
        return self.model
    
    def compile_model(self, learning_rate=0.001, device='GPU'):
        """
        Compile the model with optimizer and loss
        
        Args:
            learning_rate: Initial learning rate
            device: 'GPU' or 'CPU'
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Configure device
        if device.upper() == 'GPU':
            # Check if GPU is available
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # Enable memory growth to avoid OOM errors
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    
                    # Set GPU as visible device
                    tf.config.set_visible_devices(gpus, 'GPU')
                    
                    # Use mixed precision for faster training on GPU
                    policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(policy)
                    
                    print(f"✓ GPU training enabled on {len(gpus)} device(s)")
                    print("✓ Mixed precision enabled for faster training")
                except RuntimeError as e:
                    print(f"✗ GPU configuration failed: {e}")
                    print("  Falling back to CPU")
            else:
                print("✗ No GPU detected. Using CPU.")
        else:
            # Force CPU usage
            tf.config.set_visible_devices([], 'GPU')
            print("✓ CPU training mode selected")
        
        # Optimizer with learning rate schedule
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy'],
            run_eagerly=False  # Set to False for better performance
        )
        
        print("✓ Model compiled successfully")
        return self.model
    
    def get_callbacks(self, checkpoint_path='./models/best_model.h5',
                     log_dir='./logs/tensorboard',
                     early_stopping_patience=15,
                     reduce_lr_patience=5):
        """Get training callbacks"""
        
        callbacks = [
            # Model checkpoint - save best model
            keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=reduce_lr_patience,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard logging
            keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=False,
                update_freq='epoch'
            )
        ]
        
        return callbacks
    
    def train(self, train_data, val_data, epochs=100, batch_size=8, callbacks=None):
        """
        Train the model
        
        Args:
            train_data: Training data generator or dataset
            val_data: Validation data generator or dataset
            epochs: Number of epochs
            batch_size: Batch size (used for calculating steps)
            callbacks: List of callbacks
        
        Returns:
            history: Training history
        """
        if self.model is None:
            raise ValueError("Model not built and compiled. Call build_model() and compile_model() first.")
        
        # Calculate steps per epoch based on training samples and batch size
        steps_per_epoch = None
        validation_steps = None
        num_train_samples = 0
        num_val_samples = 0
        
        # Determine number of training samples
        if hasattr(train_data, '__iter__') and not isinstance(train_data, tuple):
            # It's a generator - try to get length
            if hasattr(train_data, '__len__'):
                steps_per_epoch = len(train_data)
                num_train_samples = steps_per_epoch * batch_size
            elif hasattr(train_data, 'n'):  # Keras ImageDataGenerator style
                num_train_samples = train_data.n
                steps_per_epoch = int(np.ceil(num_train_samples / batch_size))
            elif hasattr(train_data, 'num_samples'):  # Custom generator attribute
                num_train_samples = train_data.num_samples
                steps_per_epoch = int(np.ceil(num_train_samples / batch_size))
            # If generator has no length info, steps_per_epoch will be None
            # This will cause TensorFlow to show "Unknown" as total steps
        else:
            # It's a tuple of (X, y)
            num_train_samples = len(train_data[0])
            steps_per_epoch = int(np.ceil(num_train_samples / batch_size))
        
        # Determine number of validation samples
        if hasattr(val_data, '__iter__') and not isinstance(val_data, tuple):
            # It's a generator
            if hasattr(val_data, '__len__'):
                validation_steps = len(val_data)
                num_val_samples = validation_steps * batch_size
            elif hasattr(val_data, 'n'):
                num_val_samples = val_data.n
                validation_steps = int(np.ceil(num_val_samples / batch_size))
            elif hasattr(val_data, 'num_samples'):  # Custom generator attribute
                num_val_samples = val_data.num_samples
                validation_steps = int(np.ceil(num_val_samples / batch_size))
        else:
            # It's a tuple of (X, y)
            num_val_samples = len(val_data[0])
            validation_steps = int(np.ceil(num_val_samples / batch_size))
        
        print(f"\n{'='*60}")
        print("TRAINING STARTED")
        print(f"{'='*60}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        if num_train_samples > 0:
            print(f"Training samples: {num_train_samples}")
            if steps_per_epoch:
                print(f"Steps per epoch: {steps_per_epoch} (calculated as ⌈{num_train_samples}/{batch_size}⌉)")
        if num_val_samples > 0:
            print(f"Validation samples: {num_val_samples}")
            if validation_steps:
                print(f"Validation steps: {validation_steps} (calculated as ⌈{num_val_samples}/{batch_size}⌉)")
        print(f"{'='*60}\n")
        
        # Check if train_data is a generator or tuple
        if hasattr(train_data, '__iter__') and not isinstance(train_data, tuple):
            # It's a generator - pass steps_per_epoch
            self.history = self.model.fit(
                train_data,
                validation_data=val_data,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
        else:
            # It's a tuple of (X, y) - pass batch_size
            self.history = self.model.fit(
                train_data[0],
                train_data[1],
                validation_data=val_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETED")
        print(f"{'='*60}\n")
        
        return self.history
    
    def predict(self, sequence):
        """
        Predict class for a single sequence
        
        Args:
            sequence: Input sequence of shape (sequence_length, height, width, channels)
        
        Returns:
            predictions: Class probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call build_model() or load_model() first.")
        
        # Add batch dimension if needed
        if len(sequence.shape) == 4:
            sequence = np.expand_dims(sequence, axis=0)
        
        # Predict
        predictions = self.model.predict(sequence, verbose=0)
        
        return predictions[0]
    
    def predict_batch(self, sequences):
        """Predict classes for multiple sequences"""
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        return self.model.predict(sequences, verbose=0)
    
    def save_model(self, filepath):
        """Save model to file"""
        if self.model is None:
            raise ValueError("No model to save.")
        
        self.model.save(filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from file"""
        self.model = keras.models.load_model(
            filepath,
            custom_objects={'AttentionLayer': AttentionLayer}
        )
        print(f"✓ Model loaded from {filepath}")
        return self.model
    
    def summary(self):
        """Print model summary"""
        if self.model is None:
            raise ValueError("Model not built.")
        
        return self.model.summary()
    
    def get_model_info(self):
        """Get model information as dictionary"""
        if self.model is None:
            return None
        
        total_params = self.model.count_params()
        trainable_params = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        
        info = {
            'num_classes': self.num_classes,
            'input_shape': (self.sequence_length, self.num_features) if self.num_features else (self.sequence_length, self.frame_height, self.frame_width, self.channels),
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'model_size_mb': (total_params * 4) / (1024 ** 2),  # Assuming float32
            'num_layers': len(self.model.layers)
        }
        
        return info


def test_model():
    """Test model creation"""
    print("Testing LipReadingModel...")
    
    # Create model
    model = LipReadingModel(
        num_classes=50,
        sequence_length=75,
        frame_height=100,
        frame_width=100
    )
    
    # Build model
    model.build_model()
    print("✓ Model built successfully")
    
    # Print summary
    print("\nModel Summary:")
    model.summary()
    
    # Get model info
    info = model.get_model_info()
    print("\nModel Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test prediction with random data
    print("\nTesting prediction with random data...")
    random_sequence = np.random.rand(1, 75, 100, 100, 3).astype(np.float32)
    model.compile_model(device='CPU')
    predictions = model.predict(random_sequence)
    print(f"✓ Prediction shape: {predictions.shape}")
    print(f"✓ Prediction sum: {predictions.sum():.4f} (should be ~1.0)")


if __name__ == "__main__":
    test_model()

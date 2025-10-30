"""
Deep Learning Models for Fake News Detection
Includes: LSTM, BiLSTM, CNN-LSTM
"""

import os
import sys
from pathlib import Path
import pickle
import numpy as np

# Configure TensorFlow BEFORE importing it
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

# GPU Optimization Settings
USE_GPU = True  # Set to False to force CPU
OPTIMIZE_GPU = True  # Enable GPU optimizations (XLA, mixed precision)

if not USE_GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print("ℹ️  Using CPU for training (GPU disabled)")
else:
    # Enable GPU optimizations
    if OPTIMIZE_GPU:
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
        os.environ['TF_GPU_THREAD_COUNT'] = '2'
    
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    # Try to set XLA flags for GPU
    try:
        os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda'
    except:
        pass

import tensorflow as tf

# Configure GPU for maximum performance
if USE_GPU:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                # Allow memory growth
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Enable mixed precision for faster training on GPU
            if OPTIMIZE_GPU:
                from tensorflow.keras import mixed_precision
                policy = mixed_precision.Policy('mixed_float16')
                mixed_precision.set_global_policy(policy)
                print(f"🚀 GPU Optimizations Enabled:")
                print(f"   ✅ Mixed Precision (FP16) - 2-3x faster training")
                print(f"   ✅ XLA compilation - Optimized GPU kernels")
                print(f"   ✅ Memory growth - Efficient GPU memory usage")
            
            print(f"✅ Found {len(gpus)} GPU(s): {gpus}")
            print(f"   Using GPU for training (GTX 1650 with 4GB VRAM)")
        except RuntimeError as e:
            print(f"⚠️  GPU configuration warning: {e}")
            print(f"   Falling back to CPU")
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        print("ℹ️  No GPU found, using CPU for training")

from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

import os
# Configure TensorFlow to use GPU 0 (GTX 1650)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
# Allow GPU memory growth to avoid OOM errors
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
from pathlib import Path
import pickle
from tqdm import tqdm

# Configure GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"✅ Found {len(physical_devices)} GPU(s): {[d.name for d in physical_devices]}")
        print(f"   Using GPU for training (GTX 1650)")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("⚠️  No GPU found, using CPU")
from tqdm import tqdm
from tqdm.keras import TqdmCallback

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import MODELS_DIR, RANDOM_STATE, BATCH_SIZE, EPOCHS


class LSTMModel:
    """
    LSTM model for fake news detection
    """
    
    def __init__(self, max_features=10000, embedding_dim=128, max_length=500):
        """
        Initialize LSTM model
        
        Args:
            max_features: Maximum number of words to keep
            embedding_dim: Dimension of word embeddings
            max_length: Maximum sequence length
        """
        self.max_features = max_features
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self.history = None
        
    def build_model(self):
        """Build LSTM architecture optimized for GPU"""
        # Add 1 to max_features to account for OOV token and padding
        model = models.Sequential([
            layers.Embedding(self.max_features + 1, self.embedding_dim, 
                           input_length=self.max_length,
                           mask_zero=True),  # Mask padding
            layers.SpatialDropout1D(0.2),
            # Use CuDNN-optimized LSTM (faster on GPU)
            layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            # For mixed precision, use float32 for final layer
            layers.Dense(1, activation='sigmoid', dtype='float32')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), 
                    tf.keras.metrics.Recall()]
        )
        
        self.model = model
        return model
    
    def prepare_data(self, texts, labels=None):
        """
        Prepare text data for training
        
        Args:
            texts: Raw text data
            labels: Labels (optional, for training)
            
        Returns:
            Padded sequences and labels
        """
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words=self.max_features)
            self.tokenizer.fit_on_texts(texts)
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=self.max_length)
        
        if labels is not None:
            return X, np.array(labels)
        return X
    
    def train(self, X_train, y_train, X_val, y_val, 
              batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1):
        """
        Train the model
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            batch_size: Batch size
            epochs: Number of epochs
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
        
        # Add progress bar callback
        tqdm_callback = TqdmCallback(verbose=0)
        
        print(f"Starting training with batch_size={batch_size}, epochs={epochs}")
        print(f"⏳ First epoch may take 1-2 minutes to compile on GPU...")
        print(f"   Please wait, training is in progress...")
        
        # Train with error handling
        try:
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=[early_stop, reduce_lr, tqdm_callback],
                verbose=1  # Changed to 1 to show progress
            )
        except Exception as e:
            print(f"\n⚠️  GPU training failed: {e}")
            print(f"   Retrying with CPU...")
            
            # Force CPU
            with tf.device('/CPU:0'):
                self.history = self.model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=[early_stop, reduce_lr, tqdm_callback],
                    verbose=1
                )
        
        return self.history
    
    def evaluate(self, X, y, verbose=1):
        """
        Evaluate model
        
        Args:
            X: Features
            y: Labels
            verbose: Verbosity level
            
        Returns:
            Evaluation metrics
        """
        results = self.model.evaluate(X, y, verbose=verbose)
        metrics = {
            'loss': results[0],
            'accuracy': results[1],
            'precision': results[2],
            'recall': results[3]
        }
        
        # Calculate F1
        metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / \
                       (metrics['precision'] + metrics['recall'] + 1e-7)
        
        return metrics
    
    def predict(self, X):
        """Make predictions"""
        return (self.model.predict(X) > 0.5).astype(int).flatten()
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict(X).flatten()
    
    def save(self, model_name='lstm_model'):
        """
        Save model and tokenizer
        
        Args:
            model_name: Name for saved files
        """
        # Save model
        model_path = os.path.join(MODELS_DIR, f'{model_name}.h5')
        self.model.save(model_path)
        
        # Save tokenizer
        tokenizer_path = os.path.join(MODELS_DIR, f'{model_name}_tokenizer.pkl')
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        print(f"✅ Model saved to: {model_path}")
        print(f"✅ Tokenizer saved to: {tokenizer_path}")
    
    def load(self, model_name='lstm_model'):
        """
        Load model and tokenizer
        
        Args:
            model_name: Name of saved files
        """
        # Load model
        model_path = os.path.join(MODELS_DIR, f'{model_name}.h5')
        self.model = keras.models.load_model(model_path)
        
        # Load tokenizer
        tokenizer_path = os.path.join(MODELS_DIR, f'{model_name}_tokenizer.pkl')
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        print(f"✅ Model loaded from: {model_path}")


class BiLSTMModel(LSTMModel):
    """
    Bidirectional LSTM model
    """
    
    def build_model(self):
        """Build BiLSTM architecture optimized for GPU"""
        # Add 1 to max_features to account for OOV token and padding
        model = models.Sequential([
            layers.Embedding(self.max_features + 1, self.embedding_dim, 
                           input_length=self.max_length,
                           mask_zero=True),
            layers.SpatialDropout1D(0.2),
            # Bidirectional LSTM for better context understanding
            layers.Bidirectional(layers.LSTM(128, dropout=0.2, 
                                            recurrent_dropout=0.2)),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            # For mixed precision, use float32 for final layer
            layers.Dense(1, activation='sigmoid', dtype='float32')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), 
                    tf.keras.metrics.Recall()]
        )
        
        self.model = model
        return model


class CNNLSTMModel(LSTMModel):
    """
    CNN-LSTM hybrid model
    """
    
    def build_model(self):
        """Build CNN-LSTM architecture optimized for GPU"""
        # Add 1 to max_features to account for OOV token and padding
        model = models.Sequential([
            layers.Embedding(self.max_features + 1, self.embedding_dim, 
                           input_length=self.max_length,
                           mask_zero=True),
            layers.SpatialDropout1D(0.2),
            
            # CNN layers for feature extraction
            layers.Conv1D(64, 5, activation='relu', padding='same'),
            layers.MaxPooling1D(pool_size=4),
            
            # LSTM layer for sequence learning
            layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            # For mixed precision, use float32 for final layer
            layers.Dense(1, activation='sigmoid', dtype='float32')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), 
                    tf.keras.metrics.Recall()]
        )
        
        self.model = model
        return model


def train_deep_learning_models(texts_train, y_train, texts_val, y_val, 
                               texts_test, y_test):
    """
    Train all deep learning models
    
    Args:
        texts_train, y_train: Training data
        texts_val, y_val: Validation data
        texts_test, y_test: Test data
        
    Returns:
        Dictionary of trained models and metrics
    """
    results = {}
    models_to_train = [
        ('lstm', LSTMModel),
        ('bilstm', BiLSTMModel),
        ('cnn_lstm', CNNLSTMModel)
    ]
    
    print("\n" + "="*80)
    print("TRAINING DEEP LEARNING MODELS")
    print("="*80)
    
    # Progress bar for models
    with tqdm(total=len(models_to_train), desc="Overall Progress", position=0) as pbar_models:
        for model_name, ModelClass in models_to_train:
            pbar_models.set_description(f"Training {model_name.upper()}")
            
            print(f"\n{'*'*80}")
            print(f"Training {model_name.upper()} model")
            print(f"{'*'*80}")
            
            # Initialize model
            dl_model = ModelClass(max_features=10000, embedding_dim=128, max_length=500)
            
            # Prepare data
            tqdm.write("Preparing data...")
            X_train, y_train_arr = dl_model.prepare_data(texts_train, y_train)
            X_val = dl_model.prepare_data(texts_val)
            X_test = dl_model.prepare_data(texts_test)
            
            # Build model
            dl_model.build_model()
            tqdm.write(f"\n{model_name.upper()} Model Architecture:")
            dl_model.model.summary()
            
            # Train
            tqdm.write(f"\nTraining {model_name}...")
            # Use larger batch size for GPU (128 for GTX 1650 with 4GB)
            # Adjust down to 64 or 32 if you get OOM errors
            batch_size = 128 if USE_GPU and OPTIMIZE_GPU else 32
            history = dl_model.train(X_train, y_train_arr, X_val, y_val, 
                                    epochs=10, batch_size=batch_size)
            
            # Evaluate
            tqdm.write(f"\nEvaluating {model_name}...")
            train_metrics = dl_model.evaluate(X_train, y_train_arr, verbose=0)
            val_metrics = dl_model.evaluate(X_val, y_val, verbose=0)
            test_metrics = dl_model.evaluate(X_test, y_test, verbose=0)
            
            tqdm.write(f"\n{model_name.upper()} Test Metrics:")
            tqdm.write(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
            tqdm.write(f"  Precision: {test_metrics['precision']:.4f}")
            tqdm.write(f"  Recall:    {test_metrics['recall']:.4f}")
            tqdm.write(f"  F1-Score:  {test_metrics['f1']:.4f}")
            
            # Save model
            dl_model.save(model_name)
            
            # Store results
            results[model_name] = {
                'model': dl_model,
                'history': history,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics
            }
            
            pbar_models.update(1)
    
    return results


if __name__ == '__main__':
    import pandas as pd
    
    print("Loading preprocessed data...")
    
    # Load data
    data_path = os.path.join(project_root, 'data', 'processed', 'processed_news.csv')
    df = pd.read_csv(data_path)
    
    # Split data
    from sklearn.model_selection import train_test_split
    
    train_val, test = train_test_split(df, test_size=0.15, 
                                       random_state=RANDOM_STATE, 
                                       stratify=df['label'])
    train, val = train_test_split(train_val, test_size=0.15/0.85, 
                                  random_state=RANDOM_STATE, 
                                  stratify=train_val['label'])
    
    texts_train = train['cleaned_text'].values
    y_train = train['label'].values
    texts_val = val['cleaned_text'].values
    y_val = val['label'].values
    texts_test = test['cleaned_text'].values
    y_test = test['label'].values
    
    print(f"Training samples: {len(texts_train)}")
    print(f"Validation samples: {len(texts_val)}")
    print(f"Test samples: {len(texts_test)}")
    
    # Train models
    results = train_deep_learning_models(texts_train, y_train, 
                                        texts_val, y_val, 
                                        texts_test, y_test)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF DEEP LEARNING MODELS")
    print("="*80)
    print(f"{'Model':<20} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12} {'Test F1':<12}")
    print("-"*80)
    for model_name, result in results.items():
        print(f"{model_name:<20} "
              f"{result['train_metrics']['accuracy']:<12.4f} "
              f"{result['val_metrics']['accuracy']:<12.4f} "
              f"{result['test_metrics']['accuracy']:<12.4f} "
              f"{result['test_metrics']['f1']:<12.4f}")
    print("="*80)

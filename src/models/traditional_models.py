"""
Traditional Machine Learning Models for Fake News Detection
Includes: Logistic Regression, Naive Bayes, SVM, Random Forest
"""

import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, classification_report, confusion_matrix
)
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import MODELS_DIR, RANDOM_STATE


class TraditionalModelTrainer:
    """
    Trainer for traditional ML models
    """
    
    def __init__(self, model_type='logistic', random_state=RANDOM_STATE):
        """
        Initialize model trainer
        
        Args:
            model_type: Type of model ('logistic', 'naive_bayes', 'svm', 'random_forest')
            random_state: Random state for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the specified model"""
        if self.model_type == 'logistic':
            return LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                n_jobs=-1,
                solver='saga'
            )
        elif self.model_type == 'naive_bayes':
            return MultinomialNB(alpha=1.0)
        elif self.model_type == 'svm':
            return SVC(
                kernel='linear',
                random_state=self.random_state,
                probability=True,
                verbose=True  # Enable verbose output for SVM
            )
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1,
                max_depth=20
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train, verbose=True):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            verbose: Whether to print progress
            
        Returns:
            Trained model
        """
        if verbose:
            print(f"Training {self.model_type} model...")
            print(f"Training samples: {X_train.shape[0]}")
            print(f"Features: {X_train.shape[1]}")
            
            if self.model_type == 'svm':
                print("⚠️  SVM training can take several minutes on large datasets...")
                print("    Progress updates will appear below:")
        
        # Add timer for SVM
        import time
        start_time = time.time()
        
        self.model.fit(X_train, y_train)
        
        elapsed_time = time.time() - start_time
        
        if verbose:
            print(f"✅ Training complete! (Time: {elapsed_time:.2f}s)")
        
        return self.model
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        
        Args:
            X: Features
            
        Returns:
            Prediction probabilities
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For models without predict_proba
            return self.model.decision_function(X)
    
    def evaluate(self, X, y, dataset_name='Test'):
        """
        Evaluate model performance
        
        Args:
            X: Features
            y: True labels
            dataset_name: Name of dataset for display
            
        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='binary'),
            'recall': recall_score(y, y_pred, average='binary'),
            'f1': f1_score(y, y_pred, average='binary')
        }
        
        print(f"\n{'='*60}")
        print(f"{self.model_type.upper()} - {dataset_name} Set Performance")
        print(f"{'='*60}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        print(f"{'='*60}\n")
        
        return metrics
    
    def get_classification_report(self, X, y):
        """
        Get detailed classification report
        
        Args:
            X: Features
            y: True labels
            
        Returns:
            Classification report string
        """
        y_pred = self.predict(X)
        return classification_report(y, y_pred, target_names=['Fake', 'Real'])
    
    def get_confusion_matrix(self, X, y):
        """
        Get confusion matrix
        
        Args:
            X: Features
            y: True labels
            
        Returns:
            Confusion matrix
        """
        y_pred = self.predict(X)
        return confusion_matrix(y, y_pred)
    
    def save(self, filename=None):
        """
        Save model to file
        
        Args:
            filename: Output filename (default: {model_type}_model.pkl)
        """
        if filename is None:
            filename = f"{self.model_type}_model.pkl"
        
        filepath = os.path.join(MODELS_DIR, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"✅ Model saved to: {filepath}")
    
    def load(self, filename):
        """
        Load model from file
        
        Args:
            filename: Input filename
        """
        filepath = os.path.join(MODELS_DIR, filename)
        
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        
        print(f"✅ Model loaded from: {filepath}")
    
    def get_feature_importance(self, feature_names=None, top_n=20):
        """
        Get feature importance (for models that support it)
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            List of (feature, importance) tuples
        """
        if self.model_type == 'logistic':
            # Get coefficients for logistic regression
            coef = self.model.coef_[0]
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(coef))]
            
            # Get top positive and negative features
            top_positive_idx = coef.argsort()[-top_n:][::-1]
            top_negative_idx = coef.argsort()[:top_n]
            
            positive_features = [(feature_names[i], coef[i]) for i in top_positive_idx]
            negative_features = [(feature_names[i], coef[i]) for i in top_negative_idx]
            
            return {
                'positive': positive_features,
                'negative': negative_features
            }
        
        elif self.model_type == 'random_forest':
            # Get feature importances
            importances = self.model.feature_importances_
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(importances))]
            
            top_idx = importances.argsort()[-top_n:][::-1]
            top_features = [(feature_names[i], importances[i]) for i in top_idx]
            
            return top_features
        
        else:
            print(f"Feature importance not available for {self.model_type}")
            return None


def train_all_models(X_train, y_train, X_val, y_val, X_test, y_test, skip_svm=False, svm_sample_size=5000):
    """
    Train all traditional ML models
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        skip_svm: Skip SVM training (recommended for large datasets)
        svm_sample_size: Number of samples to use for SVM (None = use all, default=5000)
        
    Returns:
        Dictionary of trained models and their metrics
    """
    models = ['logistic', 'naive_bayes', 'random_forest']
    if not skip_svm:
        models.insert(2, 'svm')
    
    results = {}
    
    print("\n" + "="*80)
    print("TRAINING TRADITIONAL ML MODELS")
    if skip_svm:
        print("(Skipping SVM for faster training)")
    elif svm_sample_size and svm_sample_size < len(y_train):
        print(f"(Using {svm_sample_size} samples for SVM training)")
    print("="*80)
    
    # Progress bar for models
    with tqdm(total=len(models), desc="Overall Progress", position=0) as pbar_models:
        for model_type in models:
            pbar_models.set_description(f"Training {model_type.upper()}")
            
            print(f"\n{'*'*80}")
            print(f"Training {model_type.upper()} model")
            print(f"{'*'*80}")
            
            # Initialize and train
            trainer = TraditionalModelTrainer(model_type=model_type)
            
            # Use subset for SVM if specified
            if model_type == 'svm' and svm_sample_size and svm_sample_size < len(y_train):
                print(f"⚠️  Using stratified sample of {svm_sample_size} for faster SVM training...")
                from sklearn.model_selection import train_test_split
                from scipy.sparse import issparse
                
                # Create stratified sample
                if issparse(X_train):
                    X_train_svm, _, y_train_svm, _ = train_test_split(
                        X_train, y_train, 
                        train_size=svm_sample_size,
                        stratify=y_train,
                        random_state=42
                    )
                else:
                    indices = np.random.choice(len(y_train), svm_sample_size, replace=False)
                    X_train_svm = X_train[indices]
                    y_train_svm = y_train[indices]
                
                trainer.train(X_train_svm, y_train_svm, verbose=True)
            else:
                trainer.train(X_train, y_train, verbose=True)
            
            # Evaluate on all sets
            train_metrics = trainer.evaluate(X_train, y_train, 'Training')
            val_metrics = trainer.evaluate(X_val, y_val, 'Validation')
            test_metrics = trainer.evaluate(X_test, y_test, 'Test')
            
            # Save model
            trainer.save()
            
            # Store results
            results[model_type] = {
                'model': trainer,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics
            }
            
            pbar_models.update(1)
    
    return results


if __name__ == '__main__':
    # Example usage
    print("Loading TF-IDF features...")
    
    # Load features
    features_dir = os.path.join(project_root, 'data', 'processed', 'features', 'tfidf')
    
    X_train = np.load(os.path.join(features_dir, 'X_train.npy'), allow_pickle=True)
    y_train = np.load(os.path.join(features_dir, 'y_train.npy'), allow_pickle=True)
    X_val = np.load(os.path.join(features_dir, 'X_val.npy'), allow_pickle=True)
    y_val = np.load(os.path.join(features_dir, 'y_val.npy'), allow_pickle=True)
    X_test = np.load(os.path.join(features_dir, 'X_test.npy'), allow_pickle=True)
    y_test = np.load(os.path.join(features_dir, 'y_test.npy'), allow_pickle=True)
    
    # Extract from 0-d arrays if needed
    if X_train.shape == ():
        X_train = X_train.item()
    if X_val.shape == ():
        X_val = X_val.item()
    if X_test.shape == ():
        X_test = X_test.item()
    
    # Convert sparse matrices if needed
    from scipy.sparse import issparse
    if issparse(X_train):
        print("Sparse matrices detected (keeping sparse for efficiency)...")
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Train all models (skip SVM by default for speed)
    print("\nNote: Skipping SVM training due to slow performance on large datasets.")
    print("If you want to train SVM, set skip_svm=False in train_all_models()")
    results = train_all_models(X_train, y_train, X_val, y_val, X_test, y_test, skip_svm=True)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF ALL MODELS")
    print("="*80)
    print(f"{'Model':<20} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12} {'Test F1':<12}")
    print("-"*80)
    for model_type, result in results.items():
        print(f"{model_type:<20} "
              f"{result['train_metrics']['accuracy']:<12.4f} "
              f"{result['val_metrics']['accuracy']:<12.4f} "
              f"{result['test_metrics']['accuracy']:<12.4f} "
              f"{result['test_metrics']['f1']:<12.4f}")
    print("="*80)

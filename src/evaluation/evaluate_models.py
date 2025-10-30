"""
Model Evaluation and Comparison
Comprehensive analysis of all trained models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    average_precision_score
)
import pickle
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import MODELS_DIR, RANDOM_STATE

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')


def load_features(feature_type='tfidf'):
    """Load preprocessed features"""
    features_dir = os.path.join(project_root, 'data', 'processed', 'features', feature_type)
    
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
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def plot_confusion_matrices(models_dict, X_test, y_test, save_path='results/confusion_matrices.png'):
    """Plot confusion matrices for all models"""
    n_models = len(models_dict)
    fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, (model_name, model_info) in enumerate(models_dict.items()):
        model = model_info['model']
        
        # Get predictions
        if hasattr(model, 'predict'):
            if hasattr(model, 'model'):  # Deep learning model
                y_pred = model.predict(X_test)
            else:  # Traditional ML model
                y_pred = model.predict(X_test)
        else:
            continue
        
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['Fake', 'Real'],
                   yticklabels=['Fake', 'Real'])
        axes[idx].set_title(f'{model_name.upper()}\nAccuracy: {model_info["test_metrics"]["accuracy"]:.4f}',
                          fontweight='bold')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Confusion matrices saved to: {save_path}")
    plt.show()


def plot_roc_curves(models_dict, X_test, y_test, save_path='results/roc_curves.png'):
    """Plot ROC curves for all models"""
    plt.figure(figsize=(12, 8))
    
    for model_name, model_info in models_dict.items():
        model = model_info['model']
        
        # Get prediction probabilities
        if hasattr(model, 'predict_proba'):
            if hasattr(model, 'model'):  # Deep learning model
                y_proba = model.predict_proba(X_test)
            else:  # Traditional ML model
                y_proba = model.predict_proba(X_test)
            
            # Ensure 1D array (get probability of positive class)
            if len(y_proba.shape) > 1 and y_proba.shape[1] == 2:
                y_proba = y_proba[:, 1]
        else:
            continue
        
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.plot(fpr, tpr, linewidth=2, 
                label=f'{model_name.upper()} (AUC = {roc_auc:.4f})')
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ ROC curves saved to: {save_path}")
    plt.show()


def plot_precision_recall_curves(models_dict, X_test, y_test, save_path='results/pr_curves.png'):
    """Plot Precision-Recall curves for all models"""
    plt.figure(figsize=(12, 8))
    
    for model_name, model_info in models_dict.items():
        model = model_info['model']
        
        # Get prediction probabilities
        if hasattr(model, 'predict_proba'):
            if hasattr(model, 'model'):  # Deep learning model
                y_proba = model.predict_proba(X_test)
            else:  # Traditional ML model
                y_proba = model.predict_proba(X_test)
            
            # Ensure 1D array (get probability of positive class)
            if len(y_proba.shape) > 1 and y_proba.shape[1] == 2:
                y_proba = y_proba[:, 1]
        else:
            continue
        
        # Compute PR curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        avg_precision = average_precision_score(y_test, y_proba)
        
        # Plot
        plt.plot(recall, precision, linewidth=2,
                label=f'{model_name.upper()} (AP = {avg_precision:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ PR curves saved to: {save_path}")
    plt.show()


def create_comparison_table(models_dict, save_path='results/model_comparison.csv'):
    """Create comprehensive comparison table"""
    comparison_data = []
    
    for model_name, model_info in models_dict.items():
        metrics = model_info['test_metrics']
        comparison_data.append({
            'Model': model_name.upper(),
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1'],
            'Type': 'Traditional ML' if model_name in ['logistic', 'naive_bayes', 'svm', 'random_forest'] else 'Deep Learning'
        })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('F1-Score', ascending=False).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    print(f"\n✅ Comparison table saved to: {save_path}")
    
    # Print formatted table
    print("\n" + "="*90)
    print("MODEL PERFORMANCE COMPARISON (Test Set)")
    print("="*90)
    print(df.to_string(index=False))
    print("="*90)
    
    return df


def plot_metrics_comparison(df, save_path='results/metrics_comparison.png'):
    """Plot comparison of all metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['steelblue', 'coral', 'lightgreen', 'plum']
    
    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        ax = axes[idx // 2, idx % 2]
        
        # Sort by metric value
        df_sorted = df.sort_values(metric, ascending=True)
        
        # Plot
        bars = ax.barh(df_sorted['Model'], df_sorted[metric], color=color, alpha=0.7, edgecolor='black')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, df_sorted[metric])):
            ax.text(val + 0.01, i, f'{val:.4f}', va='center', fontsize=10)
        
        ax.set_xlabel(metric, fontsize=12)
        ax.set_title(f'{metric} Comparison', fontsize=13, fontweight='bold')
        ax.set_xlim([0, 1.1])
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Metrics comparison saved to: {save_path}")
    plt.show()


def analyze_errors(model, model_name, X_test, y_test, texts_test=None, save_path='results/'):
    """Analyze model errors"""
    # Get predictions
    if hasattr(model, 'model'):  # Deep learning
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
    else:  # Traditional ML
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
    
    # Ensure 1D array (get probability of positive class)
    if len(y_proba.shape) > 1 and y_proba.shape[1] == 2:
        y_proba = y_proba[:, 1]
    
    # Find errors
    errors = y_pred != y_test
    error_indices = np.where(errors)[0]
    
    print(f"\n{'='*80}")
    print(f"ERROR ANALYSIS: {model_name.upper()}")
    print(f"{'='*80}")
    print(f"Total errors: {errors.sum()} / {len(y_test)} ({errors.sum()/len(y_test)*100:.2f}%)")
    
    # False positives and false negatives
    fp = ((y_pred == 1) & (y_test == 0)).sum()
    fn = ((y_pred == 0) & (y_test == 1)).sum()
    
    print(f"False Positives (predicted Real, actually Fake): {fp}")
    print(f"False Negatives (predicted Fake, actually Real): {fn}")
    
    # Most confident errors
    if len(error_indices) > 0:
        error_confidence = y_proba[errors]
        most_confident_errors_idx = error_indices[np.argsort(np.abs(error_confidence - 0.5))[-10:]]
        
        print(f"\nTop 10 Most Confident Errors:")
        for idx in most_confident_errors_idx[::-1]:
            print(f"  True: {'Real' if y_test[idx] == 1 else 'Fake'}, "
                  f"Predicted: {'Real' if y_pred[idx] == 1 else 'Fake'}, "
                  f"Confidence: {y_proba[idx]:.4f}")
    
    print(f"{'='*80}\n")


if __name__ == '__main__':
    print("="*80)
    print("MODEL EVALUATION & COMPARISON")
    print("="*80)
    
    # Load test features
    print("\nLoading TF-IDF features for traditional ML models...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_features('tfidf')
    
    # Load traditional ML models
    from src.models.traditional_models import TraditionalModelTrainer
    
    models_dict = {}
    ml_models = ['logistic', 'naive_bayes', 'svm', 'random_forest']
    
    print("\nLoading traditional ML models...")
    for model_name in ml_models:
        try:
            trainer = TraditionalModelTrainer(model_type=model_name)
            trainer.load(f'{model_name}_model.pkl')
            
            # Evaluate
            test_metrics = trainer.evaluate(X_test, y_test, 'Test')
            
            models_dict[model_name] = {
                'model': trainer,
                'test_metrics': test_metrics
            }
            print(f"  ✅ Loaded {model_name}")
        except Exception as e:
            print(f"  ⚠️  Could not load {model_name}: {e}")
    
    # Load deep learning models
    print("\nLoading deep learning models...")
    from src.models.deep_learning_models import LSTMModel, BiLSTMModel, CNNLSTMModel
    
    # Load preprocessed text data for DL models
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    data_path = os.path.join(project_root, 'data', 'processed', 'processed_news.csv')
    df = pd.read_csv(data_path)
    
    train_val, test = train_test_split(df, test_size=0.15, random_state=RANDOM_STATE, stratify=df['label'])
    texts_test = test['cleaned_text'].values
    y_test_dl = test['label'].values
    
    dl_models = [
        ('lstm', LSTMModel),
        ('bilstm', BiLSTMModel),
        ('cnn_lstm', CNNLSTMModel)
    ]
    
    for model_name, ModelClass in dl_models:
        try:
            dl_model = ModelClass(max_features=10000, embedding_dim=128, max_length=500)
            dl_model.load(model_name)
            
            # Prepare data
            X_test_dl = dl_model.prepare_data(texts_test)
            
            # Evaluate
            test_metrics = dl_model.evaluate(X_test_dl, y_test_dl, verbose=0)
            
            models_dict[model_name] = {
                'model': dl_model,
                'test_metrics': test_metrics
            }
            print(f"  ✅ Loaded {model_name}")
        except Exception as e:
            print(f"  ⚠️  Could not load {model_name}: {e}")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Generate evaluation plots and reports
    print("\n" + "="*80)
    print("GENERATING EVALUATION REPORTS")
    print("="*80)
    
    # 1. Comparison table
    df_comparison = create_comparison_table(models_dict)
    
    # 2. Metrics comparison plot
    plot_metrics_comparison(df_comparison)
    
    # 3. Confusion matrices (for traditional ML)
    ml_models_dict = {k: v for k, v in models_dict.items() if k in ml_models}
    if ml_models_dict:
        plot_confusion_matrices(ml_models_dict, X_test, y_test)
    
    # 4. ROC curves (for traditional ML)
    if ml_models_dict:
        plot_roc_curves(ml_models_dict, X_test, y_test)
    
    # 5. Precision-Recall curves (for traditional ML)
    if ml_models_dict:
        plot_precision_recall_curves(ml_models_dict, X_test, y_test)
    
    # 6. Error analysis for best model
    best_model_name = df_comparison.iloc[0]['Model'].lower()
    if best_model_name in models_dict:
        best_model = models_dict[best_model_name]['model']
        if best_model_name in ml_models:
            analyze_errors(best_model, best_model_name, X_test, y_test)
        else:
            X_test_best = models_dict[best_model_name]['model'].prepare_data(texts_test)
            analyze_errors(best_model, best_model_name, X_test_best, y_test_dl, texts_test)
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE!")
    print("="*80)
    print("\nGenerated files in 'results/' directory:")
    print("  - model_comparison.csv")
    print("  - metrics_comparison.png")
    print("  - confusion_matrices.png")
    print("  - roc_curves.png")
    print("  - pr_curves.png")
    print("="*80)

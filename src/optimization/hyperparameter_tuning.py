"""
Hyperparameter Tuning and Model Optimization
Optimize the best performing models using Grid Search and Random Search
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import make_scorer, f1_score
import pickle
import os
import sys
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import MODELS_DIR, RANDOM_STATE


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


def tune_logistic_regression(X_train, y_train, X_val, y_val, search_type='grid'):
    """Tune Logistic Regression hyperparameters"""
    print("\n" + "="*80)
    print("TUNING LOGISTIC REGRESSION")
    print("="*80)
    
    # Parameter grid
    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear', 'saga'],
        'max_iter': [500, 1000]
    }
    
    if search_type == 'random':
        param_grid['C'] = np.logspace(-3, 3, 20)
        search = RandomizedSearchCV(
            LogisticRegression(random_state=RANDOM_STATE),
            param_grid,
            n_iter=20,
            cv=3,
            scoring='f1',
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=1
        )
    else:
        search = GridSearchCV(
            LogisticRegression(random_state=RANDOM_STATE),
            param_grid,
            cv=3,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
    
    print(f"\nRunning {search_type.upper()} search...")
    search.fit(X_train, y_train)
    
    print(f"\n✅ Best parameters: {search.best_params_}")
    print(f"✅ Best CV F1-score: {search.best_score_:.4f}")
    
    # Evaluate on validation set
    val_score = f1_score(y_val, search.best_estimator_.predict(X_val))
    print(f"✅ Validation F1-score: {val_score:.4f}")
    
    return search.best_estimator_, search.best_params_


def tune_random_forest(X_train, y_train, X_val, y_val, search_type='grid'):
    """Tune Random Forest hyperparameters"""
    print("\n" + "="*80)
    print("TUNING RANDOM FOREST")
    print("="*80)
    
    # Parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    if search_type == 'random':
        param_grid['n_estimators'] = [50, 100, 200, 300, 500]
        param_grid['max_depth'] = [10, 20, 30, 40, 50, None]
        search = RandomizedSearchCV(
            RandomForestClassifier(random_state=RANDOM_STATE),
            param_grid,
            n_iter=30,
            cv=3,
            scoring='f1',
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=1
        )
    else:
        search = GridSearchCV(
            RandomForestClassifier(random_state=RANDOM_STATE),
            param_grid,
            cv=3,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
    
    print(f"\nRunning {search_type.upper()} search...")
    search.fit(X_train, y_train)
    
    print(f"\n✅ Best parameters: {search.best_params_}")
    print(f"✅ Best CV F1-score: {search.best_score_:.4f}")
    
    # Evaluate on validation set
    val_score = f1_score(y_val, search.best_estimator_.predict(X_val))
    print(f"✅ Validation F1-score: {val_score:.4f}")
    
    return search.best_estimator_, search.best_params_


def tune_svm(X_train, y_train, X_val, y_val, sample_size=5000):
    """Tune SVM hyperparameters (on subset due to computational cost)"""
    print("\n" + "="*80)
    print("TUNING SVM")
    print("="*80)
    
    # Use subset for faster tuning
    from sklearn.model_selection import StratifiedShuffleSplit
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=RANDOM_STATE)
    for train_idx, _ in splitter.split(X_train, y_train):
        X_train_sub = X_train[train_idx]
        y_train_sub = y_train[train_idx]
    
    print(f"Using {sample_size} samples for SVM tuning...")
    
    # Parameter grid
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    
    search = GridSearchCV(
        SVC(probability=True, random_state=RANDOM_STATE),
        param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    print("\nRunning GRID search...")
    search.fit(X_train_sub, y_train_sub)
    
    print(f"\n✅ Best parameters: {search.best_params_}")
    print(f"✅ Best CV F1-score: {search.best_score_:.4f}")
    
    # Train on full dataset with best params
    print("\nTraining on full dataset with best parameters...")
    best_model = SVC(probability=True, random_state=RANDOM_STATE, **search.best_params_)
    best_model.fit(X_train_sub, y_train_sub)
    
    # Evaluate on validation set
    val_score = f1_score(y_val, best_model.predict(X_val))
    print(f"✅ Validation F1-score: {val_score:.4f}")
    
    return best_model, search.best_params_


def create_ensemble_model(models, X_train, y_train, X_val, y_val):
    """Create ensemble using voting classifier"""
    from sklearn.ensemble import VotingClassifier
    
    print("\n" + "="*80)
    print("CREATING ENSEMBLE MODEL")
    print("="*80)
    
    estimators = [(name, model) for name, model in models.items()]
    
    # Hard voting
    ensemble_hard = VotingClassifier(estimators=estimators, voting='hard')
    ensemble_hard.fit(X_train, y_train)
    
    hard_score = f1_score(y_val, ensemble_hard.predict(X_val))
    print(f"✅ Hard Voting F1-score: {hard_score:.4f}")
    
    # Soft voting
    ensemble_soft = VotingClassifier(estimators=estimators, voting='soft')
    ensemble_soft.fit(X_train, y_train)
    
    soft_score = f1_score(y_val, ensemble_soft.predict(X_val))
    print(f"✅ Soft Voting F1-score: {soft_score:.4f}")
    
    # Return best
    if soft_score > hard_score:
        print(f"\n🏆 Soft voting performs better!")
        return ensemble_soft, 'soft'
    else:
        print(f"\n🏆 Hard voting performs better!")
        return ensemble_hard, 'hard'


def optimize_deep_learning_model(model_type='lstm', epochs_range=[10, 20, 30], 
                                 batch_sizes=[64, 128, 256], learning_rates=[0.001, 0.0001]):
    """Optimize deep learning model hyperparameters"""
    print("\n" + "="*80)
    print(f"OPTIMIZING {model_type.upper()} MODEL")
    print("="*80)
    
    from src.models.deep_learning_models import LSTMModel, BiLSTMModel, CNNLSTMModel
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    # Load data
    data_path = os.path.join(project_root, 'data', 'processed', 'processed_news.csv')
    df = pd.read_csv(data_path)
    
    train_val, test = train_test_split(df, test_size=0.15, random_state=RANDOM_STATE, stratify=df['label'])
    train, val = train_test_split(train_val, test_size=0.15, random_state=RANDOM_STATE, stratify=train_val['label'])
    
    X_train = train['cleaned_text'].values
    y_train = train['label'].values
    X_val = val['cleaned_text'].values
    y_val = val['label'].values
    
    # Model class
    ModelClass = {'lstm': LSTMModel, 'bilstm': BiLSTMModel, 'cnn_lstm': CNNLSTMModel}[model_type]
    
    best_score = 0
    best_params = {}
    best_model = None
    
    results = []
    
    total_combinations = len(epochs_range) * len(batch_sizes) * len(learning_rates)
    print(f"\nTesting {total_combinations} hyperparameter combinations...")
    
    with tqdm(total=total_combinations, desc="Hyperparameter Search") as pbar:
        for epochs in epochs_range:
            for batch_size in batch_sizes:
                for lr in learning_rates:
                    print(f"\nTrying: epochs={epochs}, batch_size={batch_size}, lr={lr}")
                    
                    # Create and train model
                    model = ModelClass(max_features=10000, embedding_dim=128, max_length=500)
                    model.build_model(learning_rate=lr)
                    
                    # Train
                    model.train(
                        X_train, y_train,
                        X_val, y_val,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=0
                    )
                    
                    # Evaluate
                    X_val_prep = model.prepare_data(X_val)
                    metrics = model.evaluate(X_val_prep, y_val, verbose=0)
                    val_f1 = metrics['f1']
                    
                    results.append({
                        'epochs': epochs,
                        'batch_size': batch_size,
                        'learning_rate': lr,
                        'val_f1': val_f1,
                        'val_accuracy': metrics['accuracy']
                    })
                    
                    print(f"Validation F1: {val_f1:.4f}")
                    
                    if val_f1 > best_score:
                        best_score = val_f1
                        best_params = {'epochs': epochs, 'batch_size': batch_size, 'learning_rate': lr}
                        best_model = model
                    
                    pbar.update(1)
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('val_f1', ascending=False)
    results_path = f'results/{model_type}_tuning_results.csv'
    df_results.to_csv(results_path, index=False)
    
    print("\n" + "="*80)
    print(f"✅ Best parameters: {best_params}")
    print(f"✅ Best validation F1: {best_score:.4f}")
    print(f"✅ Results saved to: {results_path}")
    print("="*80)
    
    return best_model, best_params, df_results


if __name__ == '__main__':
    print("="*80)
    print("MODEL OPTIMIZATION & HYPERPARAMETER TUNING")
    print("="*80)
    
    # Create optimization results directory
    os.makedirs('results/optimization', exist_ok=True)
    
    # Load features
    print("\nLoading TF-IDF features...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_features('tfidf')
    
    # Combine train and val for final training
    from scipy.sparse import vstack
    X_train_full = vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])
    
    optimized_models = {}
    
    # 1. Tune Logistic Regression
    lr_model, lr_params = tune_logistic_regression(X_train, y_train, X_val, y_val, search_type='grid')
    optimized_models['logistic'] = lr_model
    
    # Save parameters
    with open('results/optimization/logistic_best_params.pkl', 'wb') as f:
        pickle.dump(lr_params, f)
    
    # 2. Tune Random Forest
    rf_model, rf_params = tune_random_forest(X_train, y_train, X_val, y_val, search_type='random')
    optimized_models['random_forest'] = rf_model
    
    with open('results/optimization/random_forest_best_params.pkl', 'wb') as f:
        pickle.dump(rf_params, f)
    
    # 3. Tune SVM (optional - very slow)
    tune_svm_flag = input("\nTune SVM? (takes ~15-30 min) [y/N]: ").lower() == 'y'
    if tune_svm_flag:
        svm_model, svm_params = tune_svm(X_train, y_train, X_val, y_val)
        optimized_models['svm'] = svm_model
        
        with open('results/optimization/svm_best_params.pkl', 'wb') as f:
            pickle.dump(svm_params, f)
    
    # 4. Create Ensemble
    if len(optimized_models) >= 2:
        ensemble_model, voting_type = create_ensemble_model(optimized_models, X_train, y_train, X_val, y_val)
        
        # Save ensemble
        with open(os.path.join(MODELS_DIR, 'ensemble_model.pkl'), 'wb') as f:
            pickle.dump(ensemble_model, f)
        print(f"\n✅ Ensemble model saved!")
    
    # 5. Evaluate optimized models on test set
    print("\n" + "="*80)
    print("FINAL EVALUATION ON TEST SET")
    print("="*80)
    
    from sklearn.metrics import classification_report
    
    for name, model in optimized_models.items():
        y_pred = model.predict(X_test)
        print(f"\n{name.upper()}:")
        print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
    
    if 'ensemble_model' in locals():
        y_pred = ensemble_model.predict(X_test)
        print(f"\nENSEMBLE ({voting_type.upper()} VOTING):")
        print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
    
    print("\n" + "="*80)
    print("✅ OPTIMIZATION COMPLETE!")
    print("="*80)
    print("\nOptimized models saved to:", MODELS_DIR)
    print("Tuning results saved to: results/optimization/")
    print("="*80)

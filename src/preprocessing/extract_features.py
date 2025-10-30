"""
Feature Engineering Pipeline
Extracts features using TF-IDF, Word2Vec, and BERT
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
import pickle

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE
from src.preprocessing.feature_extractor import TFIDFExtractor, Word2VecExtractor, BERTExtractor, reduce_dimensions


def load_processed_data():
    """Load preprocessed data"""
    print("📥 Loading preprocessed data...")
    
    data_path = PROCESSED_DATA_DIR / "processed_news.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found at {data_path}\n"
            "Please run: python src/preprocessing/preprocess.py"
        )
    
    df = pd.read_csv(data_path)
    print(f"   ✅ Loaded {len(df):,} samples")
    
    return df


def split_data(df, test_size=TEST_SIZE, val_size=VALIDATION_SIZE, random_state=RANDOM_STATE):
    """
    Split data into train, validation, and test sets
    
    Args:
        df: Input DataFrame
        test_size: Test set proportion
        val_size: Validation set proportion
        random_state: Random seed
        
    Returns:
        Train, validation, and test DataFrames
    """
    print(f"\n📊 Splitting data (train/val/test)...")
    
    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['label']
    )
    
    # Second split: train vs val
    val_proportion = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_proportion,
        random_state=random_state,
        stratify=train_val_df['label']
    )
    
    print(f"   Train set: {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   Val set:   {len(val_df):,} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   Test set:  {len(test_df):,} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df


def extract_tfidf_features(train_df, val_df, test_df, text_col='cleaned_text'):
    """
    Extract TF-IDF features
    
    Returns:
        Feature matrices and labels for train, val, test
    """
    print("\n" + "="*70)
    print("EXTRACTING TF-IDF FEATURES")
    print("="*70)
    
    # Initialize extractor
    tfidf = TFIDFExtractor(max_features=10000, ngram_range=(1, 2))
    
    # Fit on train, transform all sets
    X_train = tfidf.fit_transform(train_df[text_col].values)
    X_val = tfidf.transform(val_df[text_col].values)
    X_test = tfidf.transform(test_df[text_col].values)
    
    # Get labels
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values
    
    # Save vectorizer
    tfidf.save('tfidf_vectorizer.pkl')
    
    # Save features
    save_features('tfidf', X_train, X_val, X_test, y_train, y_val, y_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def extract_word2vec_features(train_df, val_df, test_df, text_col='cleaned_text'):
    """
    Extract Word2Vec features
    
    Returns:
        Feature matrices and labels for train, val, test
    """
    print("\n" + "="*70)
    print("EXTRACTING WORD2VEC FEATURES")
    print("="*70)
    
    # Initialize extractor
    w2v = Word2VecExtractor(vector_size=300, window=5, min_count=5)
    
    # Fit on train
    w2v.fit(train_df[text_col].values)
    
    # Transform all sets
    X_train = w2v.transform(train_df[text_col].values)
    X_val = w2v.transform(val_df[text_col].values)
    X_test = w2v.transform(test_df[text_col].values)
    
    # Get labels
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values
    
    # Save model
    w2v.save('word2vec_model.pkl')
    
    # Save features
    save_features('word2vec', X_train, X_val, X_test, y_train, y_val, y_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def extract_bert_features(train_df, val_df, test_df, text_col='cleaned_text'):
    """
    Extract BERT features (optional, computationally expensive)
    
    Returns:
        Feature matrices and labels for train, val, test
    """
    print("\n" + "="*70)
    print("EXTRACTING BERT FEATURES")
    print("="*70)
    print("⚠️  Warning: This may take a long time and requires significant memory!")
    
    # Initialize extractor
    bert = BERTExtractor(model_name='bert-base-uncased', max_length=512)
    bert.load_model()
    
    # Transform all sets
    X_train = bert.transform(train_df[text_col].values[:1000], batch_size=8)  # Limit for demo
    X_val = bert.transform(val_df[text_col].values[:200], batch_size=8)
    X_test = bert.transform(test_df[text_col].values[:200], batch_size=8)
    
    # Get labels
    y_train = train_df['label'].values[:1000]
    y_val = val_df['label'].values[:200]
    y_test = test_df['label'].values[:200]
    
    # Save features
    save_features('bert', X_train, X_val, X_test, y_train, y_val, y_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_features(feature_type, X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Save feature matrices to disk
    
    Args:
        feature_type: Type of features ('tfidf', 'word2vec', 'bert')
        X_train, X_val, X_test: Feature matrices
        y_train, y_val, y_test: Labels
    """
    features_dir = PROCESSED_DATA_DIR / 'features' / feature_type
    features_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as numpy arrays
    np.save(features_dir / 'X_train.npy', X_train)
    np.save(features_dir / 'X_val.npy', X_val)
    np.save(features_dir / 'X_test.npy', X_test)
    np.save(features_dir / 'y_train.npy', y_train)
    np.save(features_dir / 'y_val.npy', y_val)
    np.save(features_dir / 'y_test.npy', y_test)
    
    print(f"💾 {feature_type.upper()} features saved to {features_dir}")


def main():
    """
    Main feature engineering pipeline
    """
    print("="*70)
    print("FAKE NEWS DETECTION - FEATURE ENGINEERING")
    print("="*70)
    
    try:
        # Load data
        df = load_processed_data()
        
        # Split data
        train_df, val_df, test_df = split_data(df)
        
        # Extract TF-IDF features
        print("\n🔧 Starting feature extraction...")
        X_train_tfidf, X_val_tfidf, X_test_tfidf, y_train, y_val, y_test = extract_tfidf_features(
            train_df, val_df, test_df
        )
        
        # Extract Word2Vec features
        X_train_w2v, X_val_w2v, X_test_w2v, _, _, _ = extract_word2vec_features(
            train_df, val_df, test_df
        )
        
        # BERT features (optional - commented out by default due to computational cost)
        # Uncomment the following lines to extract BERT features:
        # print("\n⚠️  Extracting BERT features (this may take a while)...")
        # X_train_bert, X_val_bert, X_test_bert, _, _, _ = extract_bert_features(
        #     train_df, val_df, test_df
        # )
        
        print("\n" + "="*70)
        print("✅ FEATURE ENGINEERING COMPLETE!")
        print("="*70)
        print("\n📊 Feature Summary:")
        print(f"   TF-IDF:   {X_train_tfidf.shape[1]:,} features")
        print(f"   Word2Vec: {X_train_w2v.shape[1]:,} features")
        print(f"\n📁 Features saved in: {PROCESSED_DATA_DIR / 'features'}")
        print("\n📝 Next steps:")
        print("   1. Open notebooks/03_feature_engineering.ipynb for visualization")
        print("   2. Proceed to Phase 4: Model Development")
        print("\n" + "="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

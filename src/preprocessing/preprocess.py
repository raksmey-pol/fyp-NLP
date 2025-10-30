"""
Data Loading and Preprocessing Pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, RANDOM_STATE
from src.preprocessing.text_cleaner import TextCleaner, remove_short_texts, combine_text_columns


def load_raw_data():
    """
    Load raw fake and real news datasets
    
    Returns:
        Combined DataFrame with labels
    """
    print("📥 Loading raw data...")
    
    # Load datasets
    fake_path = RAW_DATA_DIR / "Fake.csv"
    true_path = RAW_DATA_DIR / "True.csv"
    
    if not fake_path.exists() or not true_path.exists():
        raise FileNotFoundError(
            f"Dataset files not found in {RAW_DATA_DIR}\n"
            "Please run: python src/utils/download_data.py"
        )
    
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    
    # Add labels
    fake_df['label'] = 0  # Fake news
    true_df['label'] = 1  # Real news
    
    # Add source column for tracking
    fake_df['source'] = 'fake'
    true_df['source'] = 'real'
    
    # Combine datasets
    df = pd.concat([fake_df, true_df], ignore_index=True)
    
    print(f"  ✅ Loaded {len(fake_df):,} fake news articles")
    print(f"  ✅ Loaded {len(true_df):,} real news articles")
    print(f"  ✅ Total: {len(df):,} articles")
    
    return df


def preprocess_data(df, save_intermediate=True):
    """
    Complete preprocessing pipeline
    
    Args:
        df: Input DataFrame
        save_intermediate: Save intermediate results
        
    Returns:
        Preprocessed DataFrame
    """
    print("\n🔧 Starting preprocessing pipeline...")
    
    # 1. Handle missing values
    print("\n1️⃣ Handling missing values...")
    initial_count = len(df)
    df = df.dropna(subset=['title', 'text'])
    print(f"   Removed {initial_count - len(df):,} rows with missing values")
    
    # 2. Combine title and text
    print("\n2️⃣ Combining title and text...")
    df = combine_text_columns(df, title_col='title', text_col='text', output_col='full_text')
    
    # 3. Clean text
    print("\n3️⃣ Cleaning text (this may take a while)...")
    cleaner = TextCleaner(
        lowercase=True,
        remove_urls=True,
        remove_html=True,
        remove_emails=True,
        remove_numbers=False,  # Keep numbers for now
        remove_punctuation=True,
        remove_stopwords=True,
        lemmatize=True,
        stem=False
    )
    
    df = cleaner.clean_dataframe(df, text_column='full_text', output_column='cleaned_text')
    print("   ✅ Text cleaning complete")
    
    # 4. Remove short texts
    print("\n4️⃣ Removing very short texts...")
    before_count = len(df)
    df = remove_short_texts(df, text_column='cleaned_text', min_words=10)
    print(f"   Removed {before_count - len(df):,} short texts")
    
    # 5. Add text statistics
    print("\n5️⃣ Adding text statistics...")
    df['char_count'] = df['cleaned_text'].str.len()
    df['word_count'] = df['cleaned_text'].str.split().str.len()
    df['avg_word_length'] = df['cleaned_text'].apply(
        lambda x: np.mean([len(word) for word in x.split()]) if x else 0
    )
    
    # 6. Shuffle the data
    print("\n6️⃣ Shuffling data...")
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    
    print(f"\n✅ Preprocessing complete! Final dataset: {len(df):,} articles")
    
    return df


def save_processed_data(df, filename='processed_news.csv'):
    """
    Save processed data to CSV
    
    Args:
        df: Processed DataFrame
        filename: Output filename
    """
    output_path = PROCESSED_DATA_DIR / filename
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved processed data to: {output_path}")
    
    # Print summary
    print(f"\n📊 Dataset Summary:")
    print(f"   Total samples: {len(df):,}")
    print(f"   Fake news: {(df['label']==0).sum():,}")
    print(f"   Real news: {(df['label']==1).sum():,}")
    print(f"   Columns: {list(df.columns)}")


def main():
    """
    Main preprocessing pipeline
    """
    print("="*70)
    print("FAKE NEWS DETECTION - DATA PREPROCESSING")
    print("="*70)
    
    try:
        # Load data
        df = load_raw_data()
        
        # Preprocess
        df_processed = preprocess_data(df)
        
        # Save
        save_processed_data(df_processed)
        
        print("\n" + "="*70)
        print("✅ PREPROCESSING COMPLETE!")
        print("="*70)
        print("\n📝 Next steps:")
        print("   1. Open notebooks/02_preprocessing_eda.ipynb")
        print("   2. Explore the preprocessed data")
        print("   3. Move to feature engineering (Phase 3)")
        print("\n" + "="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

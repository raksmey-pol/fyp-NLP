"""
Data Download Script for Fake News Detection Project
Downloads the Fake and Real News Dataset from Kaggle
"""

import os
import sys
import zipfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import RAW_DATA_DIR


def download_kaggle_dataset():
    """
    Download the Fake and Real News dataset from Kaggle
    Requires kaggle.json in ~/.kaggle/
    """
    try:
        import kaggle
        
        print("📥 Downloading Fake and Real News Dataset from Kaggle...")
        
        # Download dataset
        kaggle.api.dataset_download_files(
            'clmentbisaillon/fake-and-real-news-dataset',
            path=str(RAW_DATA_DIR),
            unzip=True
        )
        
        print(f"✅ Dataset downloaded successfully to {RAW_DATA_DIR}")
        
        # List downloaded files
        files = list(RAW_DATA_DIR.glob("*.csv"))
        print(f"\n📁 Downloaded files:")
        for file in files:
            print(f"  - {file.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\n💡 Alternative: Download manually from:")
        print("   https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
        print(f"   and place files in: {RAW_DATA_DIR}")
        return False


def manual_download_instructions():
    """
    Print instructions for manual download
    """
    print("\n" + "="*70)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*70)
    print("\n1. Go to: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
    print("2. Click 'Download' button")
    print("3. Extract the ZIP file")
    print(f"4. Place 'Fake.csv' and 'True.csv' in: {RAW_DATA_DIR}")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    print("="*70)
    print("FAKE NEWS DETECTION - DATA DOWNLOAD")
    print("="*70 + "\n")
    
    # Check if kaggle.json exists
    kaggle_config = Path.home() / ".kaggle" / "kaggle.json"
    
    if not kaggle_config.exists():
        print("⚠️  Kaggle API credentials not found!")
        print(f"\n📝 To use Kaggle API:")
        print("   1. Go to https://www.kaggle.com/settings/account")
        print("   2. Click 'Create New API Token'")
        print("   3. Save kaggle.json to ~/.kaggle/")
        print("   4. Run: chmod 600 ~/.kaggle/kaggle.json")
        manual_download_instructions()
    else:
        success = download_kaggle_dataset()
        if not success:
            manual_download_instructions()

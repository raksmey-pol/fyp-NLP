"""
Setup script for Fake News Detection Project
Run this after installing requirements.txt
"""

import sys
import subprocess
from pathlib import Path


def download_nltk_data():
    """Download required NLTK data"""
    print("\n📦 Downloading NLTK data...")
    try:
        import nltk
        
        resources = [
            'punkt',
            'stopwords', 
            'wordnet',
            'averaged_perceptron_tagger',
            'omw-1.4'
        ]
        
        for resource in resources:
            print(f"  Downloading {resource}...")
            nltk.download(resource, quiet=True)
        
        print("✅ NLTK data downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error downloading NLTK data: {e}")
        return False


def verify_installation():
    """Verify all required packages are installed"""
    print("\n🔍 Verifying installation...")
    
    packages = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'sklearn': 'Scikit-learn',
        'nltk': 'NLTK',
        'transformers': 'Transformers',
        'tensorflow': 'TensorFlow',
        'torch': 'PyTorch',
    }
    
    failed = []
    
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - NOT INSTALLED")
            failed.append(name)
    
    if failed:
        print(f"\n⚠️  Missing packages: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages installed successfully!")
        return True


def check_dataset():
    """Check if dataset files exist"""
    print("\n📊 Checking dataset...")
    
    project_root = Path(__file__).parent
    data_dir = project_root / "data" / "raw"
    
    fake_file = data_dir / "Fake.csv"
    true_file = data_dir / "True.csv"
    
    if fake_file.exists() and true_file.exists():
        print(f"  ✅ Dataset found in {data_dir}")
        return True
    else:
        print(f"  ⚠️  Dataset not found in {data_dir}")
        print("\n  Run: python src/utils/download_data.py")
        print("  Or download manually from Kaggle")
        return False


def main():
    """Main setup function"""
    print("="*70)
    print("FAKE NEWS DETECTION - PROJECT SETUP")
    print("="*70)
    
    # Verify installation
    installation_ok = verify_installation()
    
    if installation_ok:
        # Download NLTK data
        download_nltk_data()
        
        # Check dataset
        check_dataset()
        
        print("\n" + "="*70)
        print("✅ SETUP COMPLETE!")
        print("="*70)
        print("\n📝 Next steps:")
        print("  1. Ensure dataset is in data/raw/")
        print("  2. Start with: jupyter notebook")
        print("  3. Open notebooks/ for exploratory analysis")
        print("\n" + "="*70 + "\n")
    else:
        print("\n⚠️  Please install missing packages first:")
        print("   pip install -r requirements.txt\n")


if __name__ == "__main__":
    main()

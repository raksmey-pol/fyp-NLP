"""
Feature Extraction Module
Implements TF-IDF, Word2Vec, and BERT embeddings
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import MAX_FEATURES, MODELS_DIR


class TFIDFExtractor:
    """
    TF-IDF Feature Extractor
    """
    
    def __init__(self, max_features=MAX_FEATURES, ngram_range=(1, 2)):
        """
        Initialize TF-IDF vectorizer
        
        Args:
            max_features: Maximum number of features
            ngram_range: N-gram range (default: unigrams and bigrams)
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=5,
            max_df=0.8,
            sublinear_tf=True
        )
        
    def fit_transform(self, texts):
        """
        Fit vectorizer and transform texts
        
        Args:
            texts: List of text documents
            
        Returns:
            TF-IDF matrix
        """
        print(f"📊 Fitting TF-IDF vectorizer (max_features={self.max_features})...")
        X = self.vectorizer.fit_transform(texts)
        print(f"   ✅ TF-IDF matrix shape: {X.shape}")
        return X
    
    def transform(self, texts):
        """
        Transform texts using fitted vectorizer
        
        Args:
            texts: List of text documents
            
        Returns:
            TF-IDF matrix
        """
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self):
        """Get feature names"""
        return self.vectorizer.get_feature_names_out()
    
    def save(self, filename='tfidf_vectorizer.pkl'):
        """Save vectorizer to file"""
        filepath = MODELS_DIR / filename
        with open(filepath, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print(f"💾 TF-IDF vectorizer saved to {filepath}")
    
    def load(self, filename='tfidf_vectorizer.pkl'):
        """Load vectorizer from file"""
        filepath = MODELS_DIR / filename
        with open(filepath, 'rb') as f:
            self.vectorizer = pickle.load(f)
        print(f"📥 TF-IDF vectorizer loaded from {filepath}")


class Word2VecExtractor:
    """
    Word2Vec Feature Extractor using Gensim
    """
    
    def __init__(self, vector_size=300, window=5, min_count=5, workers=4):
        """
        Initialize Word2Vec parameters
        
        Args:
            vector_size: Dimensionality of word vectors
            window: Maximum distance between current and predicted word
            min_count: Minimum word frequency
            workers: Number of worker threads
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None
        
    def fit(self, texts):
        """
        Train Word2Vec model
        
        Args:
            texts: List of text documents (should be tokenized)
        """
        from gensim.models import Word2Vec
        
        print(f"🔤 Training Word2Vec model (vector_size={self.vector_size})...")
        
        # Tokenize texts
        tokenized_texts = [text.split() for text in texts]
        
        # Train model
        self.model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=10
        )
        
        print(f"   ✅ Word2Vec model trained with {len(self.model.wv)} words")
        
    def transform(self, texts):
        """
        Transform texts to document vectors (average of word vectors)
        
        Args:
            texts: List of text documents
            
        Returns:
            Document vectors matrix
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        doc_vectors = []
        for text in texts:
            words = text.split()
            word_vecs = []
            
            for word in words:
                if word in self.model.wv:
                    word_vecs.append(self.model.wv[word])
            
            if word_vecs:
                # Average of word vectors
                doc_vec = np.mean(word_vecs, axis=0)
            else:
                # Zero vector if no words found
                doc_vec = np.zeros(self.vector_size)
            
            doc_vectors.append(doc_vec)
        
        return np.array(doc_vectors)
    
    def fit_transform(self, texts):
        """Fit and transform in one step"""
        self.fit(texts)
        return self.transform(texts)
    
    def save(self, filename='word2vec_model.pkl'):
        """Save model to file"""
        filepath = MODELS_DIR / filename
        self.model.save(str(filepath))
        print(f"💾 Word2Vec model saved to {filepath}")
    
    def load(self, filename='word2vec_model.pkl'):
        """Load model from file"""
        from gensim.models import Word2Vec
        filepath = MODELS_DIR / filename
        self.model = Word2Vec.load(str(filepath))
        print(f"📥 Word2Vec model loaded from {filepath}")


class BERTExtractor:
    """
    BERT Feature Extractor using Transformers
    """
    
    def __init__(self, model_name='bert-base-uncased', max_length=512):
        """
        Initialize BERT model
        
        Args:
            model_name: Pretrained BERT model name
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        """Load BERT model and tokenizer"""
        from transformers import BertTokenizer, BertModel
        import torch
        
        print(f"🤖 Loading BERT model: {self.model_name}...")
        
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name)
        self.model.eval()
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"   ✅ BERT model loaded on {self.device}")
    
    def transform(self, texts, batch_size=16):
        """
        Transform texts to BERT embeddings
        
        Args:
            texts: List of text documents
            batch_size: Batch size for processing
            
        Returns:
            BERT embeddings matrix
        """
        import torch
        from tqdm import tqdm
        
        if self.model is None:
            self.load_model()
        
        print(f"🔄 Extracting BERT embeddings (batch_size={batch_size})...")
        
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Move to device
            encoded = {key: val.to(self.device) for key, val in encoded.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**encoded)
                # Use [CLS] token embedding
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings)
        print(f"   ✅ BERT embeddings shape: {embeddings.shape}")
        
        return embeddings


def reduce_dimensions(X, n_components=300, method='svd'):
    """
    Reduce dimensionality of feature matrix
    
    Args:
        X: Feature matrix
        n_components: Number of components
        method: Reduction method ('svd', 'pca')
        
    Returns:
        Reduced feature matrix
    """
    print(f"📉 Reducing dimensions to {n_components} using {method.upper()}...")
    
    if method == 'svd':
        reducer = TruncatedSVD(n_components=n_components, random_state=42)
    else:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components, random_state=42)
    
    X_reduced = reducer.fit_transform(X)
    print(f"   ✅ Reduced shape: {X_reduced.shape}")
    print(f"   Explained variance: {reducer.explained_variance_ratio_.sum():.2%}")
    
    return X_reduced, reducer


if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "this is a sample news article about politics",
        "another article discussing technology and innovation",
        "sports news about the latest championship game"
    ]
    
    # TF-IDF
    print("\n" + "="*70)
    print("TF-IDF EXAMPLE")
    print("="*70)
    tfidf = TFIDFExtractor(max_features=100)
    X_tfidf = tfidf.fit_transform(sample_texts)
    print(f"TF-IDF matrix shape: {X_tfidf.shape}")
    
    # Word2Vec
    print("\n" + "="*70)
    print("WORD2VEC EXAMPLE")
    print("="*70)
    w2v = Word2VecExtractor(vector_size=50)
    X_w2v = w2v.fit_transform(sample_texts)
    print(f"Word2Vec matrix shape: {X_w2v.shape}")
    
    print("\n✅ Feature extraction examples completed!")

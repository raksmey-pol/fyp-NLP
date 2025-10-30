"""
Flask Web Application for Fake News Detection
Simple web interface for the trained models
"""

from flask import Flask, render_template, request, jsonify
import os
import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

app = Flask(__name__)

# Global model variable
model = None
model_name = None
predict_fn = None


def load_model():
    """Load the best performing model"""
    global model, model_name, predict_fn
    
    import pandas as pd
    
    # Load best model from evaluation results
    results_path = 'results/model_comparison.csv'
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        model_name = df.iloc[0]['Model'].lower()
        print(f"Loading best model: {model_name.upper()}")
    else:
        model_name = 'lstm'
        print("No evaluation results, using LSTM")
    
    # Load the appropriate model
    if model_name in ['lstm', 'bilstm', 'cnn_lstm']:
        from src.models.deep_learning_models import LSTMModel, BiLSTMModel, CNNLSTMModel
        
        ModelClass = {
            'lstm': LSTMModel,
            'bilstm': BiLSTMModel,
            'cnn_lstm': CNNLSTMModel
        }[model_name]
        
        model = ModelClass(max_features=10000, embedding_dim=128, max_length=500)
        model.load(model_name)
        
        def predict_fn(text):
            X = model.prepare_data([text])
            proba = model.predict_proba(X)[0]
            return proba
    else:
        from src.models.traditional_models import TraditionalModelTrainer
        import pickle
        
        model = TraditionalModelTrainer(model_type=model_name)
        model.load(f'{model_name}_model.pkl')
        
        vectorizer_path = 'data/processed/features/tfidf/tfidf_vectorizer.pkl'
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        def predict_fn(text):
            X = vectorizer.transform([text])
            proba = model.model.predict_proba(X)[0]
            return proba[1] if len(proba) > 1 else proba[0]
    
    print(f"✅ Model loaded successfully!")
    return predict_fn


@app.route('/')
def home():
    """Home page"""
    return render_template('index.html', model_name=model_name.upper() if model_name else 'Unknown')


@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction on submitted text"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Make prediction
        proba = predict_fn(text)
        prediction = 'REAL' if proba >= 0.5 else 'FAKE'
        confidence = float(proba if proba >= 0.5 else 1 - proba)
        
        # Determine confidence level
        if confidence >= 0.9:
            confidence_level = 'Very High'
        elif confidence >= 0.75:
            confidence_level = 'High'
        elif confidence >= 0.6:
            confidence_level = 'Moderate'
        else:
            confidence_level = 'Low'
        
        return jsonify({
            'prediction': prediction,
            'confidence': confidence,
            'probability_real': float(proba),
            'confidence_level': confidence_level,
            'model': model_name.upper()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': model_name.upper() if model_name else 'Not loaded'
    })


if __name__ == '__main__':
    print("="*70)
    print("🚀 FAKE NEWS DETECTION WEB APP")
    print("="*70)
    
    # Load model
    predict_fn = load_model()
    
    print("\n" + "="*70)
    print("Starting Flask server...")
    print("Access the app at: http://localhost:5000")
    print("="*70 + "\n")
    
    # Run app
    app.run(debug=True, host='0.0.0.0', port=5000)

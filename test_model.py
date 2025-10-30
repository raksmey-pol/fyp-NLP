"""
Interactive Fake News Detector - Test Your Trained Models
Quick testing interface for the fake news detection system
"""

import os
import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def load_best_model():
    """Load the best performing model based on evaluation results"""
    import pandas as pd
    
    # Check if evaluation results exist
    results_path = 'results/model_comparison.csv'
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        best_model_name = df.iloc[0]['Model'].lower()
        print(f"🏆 Loading best model: {best_model_name.upper()}")
        print(f"   F1-Score: {df.iloc[0]['F1-Score']:.4f}")
        print(f"   Accuracy: {df.iloc[0]['Accuracy']:.4f}\n")
    else:
        # Default to LSTM if no evaluation results
        best_model_name = 'lstm'
        print("⚠️  No evaluation results found, using LSTM model\n")
    
    # Load the model
    if best_model_name in ['lstm', 'bilstm', 'cnn_lstm']:
        from src.models.deep_learning_models import LSTMModel, BiLSTMModel, CNNLSTMModel
        
        ModelClass = {
            'lstm': LSTMModel,
            'bilstm': BiLSTMModel,
            'cnn_lstm': CNNLSTMModel
        }[best_model_name]
        
        model = ModelClass(max_features=10000, embedding_dim=128, max_length=500)
        model.load(best_model_name)
        
        def predict_fn(text):
            X = model.prepare_data([text])
            proba = model.predict_proba(X)[0]
            return proba
        
    else:
        from src.models.traditional_models import TraditionalModelTrainer
        import pickle
        
        # Load model
        model = TraditionalModelTrainer(model_type=best_model_name)
        model.load(f'{best_model_name}_model.pkl')
        
        # Load vectorizer
        vectorizer_path = 'data/processed/features/tfidf/tfidf_vectorizer.pkl'
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        def predict_fn(text):
            X = vectorizer.transform([text])
            proba = model.model.predict_proba(X)[0]
            return proba[1] if len(proba) > 1 else proba[0]
    
    return predict_fn, best_model_name


def predict_news(text, predict_fn):
    """Make prediction on a news article"""
    proba = predict_fn(text)
    prediction = 'REAL' if proba >= 0.5 else 'FAKE'
    confidence = proba if proba >= 0.5 else 1 - proba
    
    return prediction, confidence, proba


def print_result(prediction, confidence, proba):
    """Print formatted prediction result"""
    print("\n" + "="*70)
    print("🔍 ANALYSIS RESULT")
    print("="*70)
    
    if prediction == 'FAKE':
        emoji = "🚨"
        color = "RED"
    else:
        emoji = "✅"
        color = "GREEN"
    
    print(f"{emoji} Prediction: {prediction}")
    print(f"📊 Confidence: {confidence:.2%}")
    print(f"📈 Probability (Real): {proba:.4f}")
    print("="*70)
    
    # Confidence level interpretation
    if confidence >= 0.9:
        print("💪 Very High Confidence")
    elif confidence >= 0.75:
        print("👍 High Confidence")
    elif confidence >= 0.6:
        print("🤔 Moderate Confidence")
    else:
        print("⚠️  Low Confidence - treat with caution")
    print("="*70 + "\n")


def test_examples(predict_fn):
    """Test with predefined examples"""
    examples = [
        {
            'text': "Scientists discover new planet in habitable zone of nearby star system.",
            'expected': 'REAL'
        },
        {
            'text': "BREAKING: Aliens land in New York City and demand to speak with president!",
            'expected': 'FAKE'
        },
        {
            'text': "Stock market reaches record high amid strong economic recovery and positive jobs report.",
            'expected': 'REAL'
        },
        {
            'text': "You won't believe what this celebrity did! Doctors hate them! Click here now!",
            'expected': 'FAKE'
        },
        {
            'text': "President announces new infrastructure bill passed by Congress with bipartisan support.",
            'expected': 'REAL'
        }
    ]
    
    print("\n" + "="*70)
    print("📝 TESTING WITH EXAMPLE ARTICLES")
    print("="*70 + "\n")
    
    correct = 0
    for i, example in enumerate(examples, 1):
        print(f"Example {i}:")
        print(f"Text: {example['text'][:60]}...")
        
        prediction, confidence, proba = predict_news(example['text'], predict_fn)
        
        print(f"Prediction: {prediction} (Expected: {example['expected']})")
        print(f"Confidence: {confidence:.2%}")
        
        if prediction == example['expected']:
            print("✅ CORRECT")
            correct += 1
        else:
            print("❌ INCORRECT")
        
        print("-" * 70 + "\n")
    
    print(f"Accuracy on examples: {correct}/{len(examples)} ({correct/len(examples)*100:.1f}%)")
    print("="*70 + "\n")


def interactive_mode(predict_fn):
    """Interactive testing mode"""
    print("\n" + "="*70)
    print("🎯 INTERACTIVE MODE")
    print("="*70)
    print("Enter news articles to analyze (type 'quit' to exit)")
    print("="*70 + "\n")
    
    while True:
        print("Enter news article text:")
        text = input("> ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not text:
            print("⚠️  Please enter some text\n")
            continue
        
        prediction, confidence, proba = predict_news(text, predict_fn)
        print_result(prediction, confidence, proba)


def main():
    """Main testing interface"""
    print("\n" + "="*70)
    print("🔍 FAKE NEWS DETECTION SYSTEM - TEST INTERFACE")
    print("="*70 + "\n")
    
    # Load model
    try:
        predict_fn, model_name = load_best_model()
        print(f"✅ Model loaded successfully!\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nMake sure you have trained models in the 'models/' directory")
        return
    
    # Show menu
    while True:
        print("\n" + "="*70)
        print("SELECT TEST MODE:")
        print("="*70)
        print("1. Test with predefined examples")
        print("2. Interactive testing (enter your own text)")
        print("3. Quick single test")
        print("4. Exit")
        print("="*70)
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            test_examples(predict_fn)
        
        elif choice == '2':
            interactive_mode(predict_fn)
        
        elif choice == '3':
            print("\nEnter news article text:")
            text = input("> ").strip()
            if text:
                prediction, confidence, proba = predict_news(text, predict_fn)
                print_result(prediction, confidence, proba)
            else:
                print("⚠️  No text entered\n")
        
        elif choice == '4':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("⚠️  Invalid choice, please try again")


if __name__ == '__main__':
    main()

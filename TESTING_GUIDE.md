# 🧪 Testing Your Fake News Detection Models

## Quick Start - 3 Ways to Test

### 1️⃣ Interactive Testing Interface (Recommended)

```bash
# Activate environment
source env/bin/activate

# Run interactive tester
python test_model.py
```

**Features:**

- ✅ Automatically loads the best performing model
- ✅ Test with predefined examples
- ✅ Interactive mode - enter your own news articles
- ✅ Get confidence scores and detailed analysis

---

### 2️⃣ Jupyter Notebook Testing

**Option A: Use the Evaluation Notebook**

```bash
jupyter notebook notebooks/05_model_evaluation.ipynb
```

- Run all cells to see comprehensive evaluation
- Scroll to "Interactive Prediction Testing" section
- Modify the `custom_text` variable with your own news

**Option B: Quick Testing in Any Notebook**

```python
# In any notebook cell:
from src.models.deep_learning_models import LSTMModel

# Load model
model = LSTMModel(max_features=10000, embedding_dim=128, max_length=500)
model.load('lstm')

# Test with your text
text = "Your news article here..."
X = model.prepare_data([text])
proba = model.predict_proba(X)[0]

prediction = 'REAL' if proba >= 0.5 else 'FAKE'
confidence = proba if proba >= 0.5 else 1 - proba

print(f"Prediction: {prediction}")
print(f"Confidence: {confidence:.2%}")
```

---

### 3️⃣ Python Script Testing

Create a simple test script:

```python
# test_single.py
from src.models.deep_learning_models import LSTMModel

# Load your best model
model = LSTMModel(max_features=10000, embedding_dim=128, max_length=500)
model.load('lstm')  # or 'bilstm', 'cnn_lstm'

# Your news article
news_text = """
Scientists at MIT have developed a new battery technology
that could revolutionize electric vehicles with 10x longer range.
"""

# Predict
X = model.prepare_data([news_text])
proba = model.predict_proba(X)[0]

print(f"Probability (Real): {proba:.4f}")
print(f"Prediction: {'REAL' if proba >= 0.5 else 'FAKE'}")
```

Run it:

```bash
python test_single.py
```

---

## 📊 Example Test Cases

### ✅ Real News Examples

```
1. "President announces new climate initiative at international summit."
2. "Stock market closes higher after positive economic data release."
3. "Scientists publish peer-reviewed study on vaccine efficacy."
```

### 🚨 Fake News Examples

```
1. "BREAKING: Aliens confirmed by NASA! Government hiding truth!"
2. "This one weird trick will make you rich overnight! Doctors hate it!"
3. "Celebrity reveals shocking secret that changes EVERYTHING!"
```

---

## 🎯 Testing Different Models

### Test Traditional ML Models (Logistic Regression, etc.)

```python
from src.models.traditional_models import TraditionalModelTrainer
import pickle

# Load model
model = TraditionalModelTrainer(model_type='logistic')
model.load('logistic_model.pkl')

# Load vectorizer
with open('data/processed/features/tfidf/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Predict
text = "Your news article here..."
X = vectorizer.transform([text])
proba = model.model.predict_proba(X)[0, 1]
prediction = 'REAL' if proba >= 0.5 else 'FAKE'
```

### Test Deep Learning Models

```python
from src.models.deep_learning_models import BiLSTMModel

# Load model
model = BiLSTMModel(max_features=10000, embedding_dim=128, max_length=500)
model.load('bilstm')

# Predict
text = "Your news article here..."
X = model.prepare_data([text])
proba = model.predict_proba(X)[0]
prediction = 'REAL' if proba >= 0.5 else 'FAKE'
```

---

## 🔍 Understanding the Output

### Confidence Levels

- **90-100%**: Very High Confidence - Strong prediction
- **75-90%**: High Confidence - Reliable prediction
- **60-75%**: Moderate Confidence - Good prediction
- **50-60%**: Low Confidence - Uncertain, be cautious

### Interpreting Probability

- **Probability > 0.5**: Predicted as REAL news
- **Probability < 0.5**: Predicted as FAKE news
- **Probability ≈ 0.5**: Model is uncertain

Example:

```
Probability: 0.92 → REAL news (92% confidence)
Probability: 0.15 → FAKE news (85% confidence)
Probability: 0.52 → REAL news (52% confidence - low confidence!)
```

---

## 📈 Performance Metrics

Check your model's performance:

```bash
# View evaluation results
cat results/model_comparison.csv

# Or in Python:
import pandas as pd
df = pd.read_csv('results/model_comparison.csv')
print(df)
```

---

## 🚀 Advanced Testing

### Batch Testing Multiple Articles

```python
articles = [
    "Article 1 text...",
    "Article 2 text...",
    "Article 3 text..."
]

for i, article in enumerate(articles, 1):
    X = model.prepare_data([article])
    proba = model.predict_proba(X)[0]
    prediction = 'REAL' if proba >= 0.5 else 'FAKE'
    print(f"{i}. {prediction} ({proba:.2%})")
```

### Test from CSV File

```python
import pandas as pd

# Load your test data
df = pd.read_csv('my_test_data.csv')

# Predict
results = []
for text in df['text']:
    X = model.prepare_data([text])
    proba = model.predict_proba(X)[0]
    results.append(proba)

df['prediction'] = ['REAL' if p >= 0.5 else 'FAKE' for p in results]
df['confidence'] = [p if p >= 0.5 else 1-p for p in results]

# Save results
df.to_csv('predictions.csv', index=False)
```

---

## ❓ Troubleshooting

### Model Not Found Error

```bash
# Make sure you've trained the models first
python src/models/deep_learning_models.py
```

### Import Errors

```bash
# Make sure you're in the project directory
cd /path/to/fyp-NLP

# Activate virtual environment
source env/bin/activate
```

### GPU Issues (for Deep Learning models)

- Models automatically fall back to CPU if GPU unavailable
- For CPU-only testing, models will still work but may be slower

---

## 💡 Tips for Best Results

1. **Input Quality**: Feed complete sentences/paragraphs, not just keywords
2. **Length**: Articles with 50-500 words work best
3. **Language**: Models are trained on English news
4. **Context**: Include full context, not just headlines
5. **Clean Text**: Remove special characters, URLs if possible

---

## 📞 Need Help?

- Check `results/model_comparison.csv` for model performance
- Review `notebooks/05_model_evaluation.ipynb` for examples
- See training logs in `models/` directory

Happy Testing! 🎉

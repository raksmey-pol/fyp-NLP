# Fake News Detection using NLP

A machine learning project to detect and classify fake news articles using Natural Language Processing techniques and deep learning models.

## 📋 Project Overview

This project implements multiple machine learning and deep learning models to identify fake news using various NLP techniques including TF-IDF, Word2Vec, and BERT embeddings.

## 🎯 Objectives

- Build an accurate fake news detection system
- Compare performance of different ML/DL models
- Explore various text representation techniques
- Create a deployable solution for real-world use

## 📁 Project Structure

```
fyp-NLP/
├── data/
│   ├── raw/              # Original dataset
│   └── processed/        # Cleaned and preprocessed data
├── notebooks/            # Jupyter notebooks for exploration
├── src/
│   ├── preprocessing/    # Data cleaning and preprocessing
│   ├── models/          # Model implementations
│   └── utils/           # Utility functions
├── models/              # Saved trained models
├── results/             # Evaluation results and visualizations
├── docs/                # Documentation
├── requirements.txt     # Project dependencies
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/raksmey-pol/fyp-NLP.git
cd fyp-NLP
```

2. Create and activate virtual environment:

```bash
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download NLTK data:

```python
python -c "import nltk; nltk.download('all')"
```

## 📊 Dataset

We use the **Fake and Real News Dataset** from Kaggle:

- Source: [Kaggle Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Contains labeled real and fake news articles
- Features: Title, Text, Subject, Date

### Download Dataset

```bash
# Using Kaggle API (requires kaggle.json)
kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset
```

Or download manually from Kaggle and place in `data/raw/`

## 🔬 Models Implemented

### Traditional ML Models

- Logistic Regression
- Naive Bayes
- Support Vector Machine (SVM)
- Random Forest

### Deep Learning Models

- LSTM (Long Short-Term Memory)
- BiLSTM with Attention
- Fine-tuned BERT

## 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC Curve

## 🛠️ Tech Stack

- **Language:** Python 3.8+
- **ML Frameworks:** Scikit-learn, TensorFlow, PyTorch
- **NLP Libraries:** NLTK, spaCy, Transformers
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly

## 📝 Development Phases

- [x] Phase 1: Project Setup & Data Collection
- [x] Phase 2: Data Preprocessing & EDA
- [x] Phase 3: Feature Engineering (TF-IDF, Word2Vec)
- [x] Phase 4: Model Development (7 models trained with GPU optimization)
- [x] Phase 5: Model Evaluation & Comparison
- [x] Phase 6: Model Optimization & Hyperparameter Tuning
- [ ] Phase 7: Deployment & API Development

## 👥 Authors

- **Student Name** - ITM-454 NLP Final Project
- Raksmey POL
- Virakyuth SRUN
- Henglong LY
- Sokati KEO

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- AUPP - ITM-454 Natural Language Processing Course
- Kaggle for the dataset
- Open-source NLP community
- Claude Sonnet 4.5, ChatGPT, Gemini, DeepSeek for code reviews, debugging process and co-write a comprehensive overviews.

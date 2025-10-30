"""
Fake News Detection Project Configuration
"""

import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

# Data file paths
FAKE_NEWS_PATH = RAW_DATA_DIR / "Fake.csv"
TRUE_NEWS_PATH = RAW_DATA_DIR / "True.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "processed_news.csv"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15

# Text preprocessing parameters
MAX_FEATURES = 10000
MAX_SEQUENCE_LENGTH = 500
EMBEDDING_DIM = 300

# Training parameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

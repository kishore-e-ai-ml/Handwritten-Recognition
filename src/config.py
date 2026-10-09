"""
Configuration settings for Handwritten Character Recognition System
"""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EXPLAINABILITY_DIR = FIGURES_DIR / "explainability"

# Ensure directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, EXPLAINABILITY_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model paths
MNIST_MODEL_PATH = MODELS_DIR / "mnist_cnn.pt"
EMNIST_MODEL_PATH = MODELS_DIR / "emnist_cnn.pt"
MNIST_KERAS_PATH = MODELS_DIR / "mnist_cnn.keras"
EMNIST_KERAS_PATH = MODELS_DIR / "emnist_cnn.keras"

# Dataset parameters
IMAGE_SIZE = (28, 28)
IMAGE_CHANNELS = 1
MNIST_NUM_CLASSES = 10
EMNIST_NUM_CLASSES = 47
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.1

# Training parameters
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
PATIENCE = 4

# Class Mappings
MNIST_CLASSES = [str(i) for i in range(10)]

# EMNIST Balanced 47 classes mapping standard ASCII ASCII mapping:
# 0-9: Digits 0-9
# 10-35: Capital letters A-Z
# 36-46: Lowercase letters (a, b, d, e, f, g, h, n, q, r, t)
EMNIST_BALANCED_CLASSES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
]

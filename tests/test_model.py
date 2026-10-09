"""
Tests for model architecture: build, parameter count, output shape.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import torch
from src.model import build_model, count_parameters
from src.config import MNIST_NUM_CLASSES, EMNIST_NUM_CLASSES, MODELS_DIR


def test_mnist_model_builds():
    """MNIST model should build without errors."""
    model = build_model(num_classes=MNIST_NUM_CLASSES)
    assert model is not None


def test_emnist_model_builds():
    """EMNIST model should build without errors."""
    model = build_model(num_classes=EMNIST_NUM_CLASSES)
    assert model is not None


def test_mnist_output_shape():
    """MNIST model forward pass should return (batch, 10) logits."""
    model = build_model(num_classes=MNIST_NUM_CLASSES)
    model.eval()
    dummy = torch.zeros(4, 1, 28, 28)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (4, MNIST_NUM_CLASSES)


def test_emnist_output_shape():
    """EMNIST model forward pass should return (batch, 47) logits."""
    model = build_model(num_classes=EMNIST_NUM_CLASSES)
    model.eval()
    dummy = torch.zeros(4, 1, 28, 28)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (4, EMNIST_NUM_CLASSES)


def test_parameter_count():
    """Model should have a reasonable parameter count (between 100K and 2M)."""
    model = build_model(num_classes=MNIST_NUM_CLASSES)
    params = count_parameters(model)
    assert 100_000 < params < 2_000_000


def test_mnist_model_loads():
    """Saved MNIST model checkpoint should load correctly."""
    model_path = MODELS_DIR / "mnist_cnn.pt"
    assert model_path.exists(), "mnist_cnn.pt not found – run training first"
    model = build_model(num_classes=MNIST_NUM_CLASSES)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    assert checkpoint['val_acc'] > 0.95, "Expected val_acc > 95% for MNIST"


def test_emnist_model_loads():
    """Saved EMNIST model checkpoint should load correctly."""
    model_path = MODELS_DIR / "emnist_cnn.pt"
    assert model_path.exists(), "emnist_cnn.pt not found – run training first"
    model = build_model(num_classes=EMNIST_NUM_CLASSES)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    assert checkpoint['val_acc'] > 0.80, "Expected val_acc > 80% for EMNIST"

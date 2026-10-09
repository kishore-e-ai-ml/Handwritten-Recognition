"""
Tests for preprocessing and prediction pipeline.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
import torch
from PIL import Image

from src.preprocessing import preprocess_image_for_model
from src.predict import Predictor
from src.config import MNIST_NUM_CLASSES, EMNIST_NUM_CLASSES


# ─── Preprocessing tests ────────────────────────────────────────────────────

def test_preprocess_pil_image_mnist():
    """PIL grayscale image should produce correct tensor shape for MNIST."""
    img = Image.fromarray(np.random.randint(0, 255, (28, 28), dtype=np.uint8), mode='L')
    tensor, np_arr = preprocess_image_for_model(img, dataset_type='mnist')
    assert tensor.shape == (1, 1, 28, 28)
    assert np_arr.shape == (28, 28)


def test_preprocess_numpy_rgba():
    """RGBA numpy array (canvas format) should be handled correctly."""
    rgba = np.zeros((280, 280, 4), dtype=np.uint8)
    rgba[100:180, 100:180, :] = 255  # Simulate a drawn square
    tensor, np_arr = preprocess_image_for_model(rgba, dataset_type='mnist')
    assert tensor.shape == (1, 1, 28, 28)


def test_preprocess_opaque_canvas_uses_rgb_pixels_not_alpha():
    """Opaque canvas alpha must not turn a visible drawing into a blank image."""
    rgba = np.zeros((280, 280, 4), dtype=np.uint8)
    rgba[:, :, 3] = 255
    rgba[80:200, 130:150, :3] = 255

    _, np_arr = preprocess_image_for_model(rgba, dataset_type='mnist')

    assert np_arr.max() == 1.0
    assert np.count_nonzero(np_arr) > 0


def test_preprocess_blank_opaque_canvas_stays_blank():
    """An opaque black canvas with no strokes must remain blank."""
    rgba = np.zeros((280, 280, 4), dtype=np.uint8)
    rgba[:, :, 3] = 255

    _, np_arr = preprocess_image_for_model(rgba, dataset_type='mnist')

    assert np_arr.max() == 0.0


def test_preprocess_pil_rgb():
    """RGB PIL image should be converted to grayscale and preprocessed."""
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8), mode='RGB')
    tensor, np_arr = preprocess_image_for_model(img, dataset_type='emnist')
    assert tensor.shape == (1, 1, 28, 28)


def test_preprocess_invalid_type_raises():
    """Non-image input should raise ValueError."""
    with pytest.raises((ValueError, AttributeError)):
        preprocess_image_for_model(12345, dataset_type='mnist')


# ─── Prediction tests ────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def mnist_predictor():
    return Predictor(dataset_type='mnist')


@pytest.fixture(scope='module')
def emnist_predictor():
    return Predictor(dataset_type='emnist')


def test_predictor_mnist_loads(mnist_predictor):
    """MNIST predictor should load without errors."""
    assert mnist_predictor.model is not None
    assert mnist_predictor.num_classes == MNIST_NUM_CLASSES


def test_predictor_emnist_loads(emnist_predictor):
    """EMNIST predictor should load without errors."""
    assert emnist_predictor.model is not None
    assert emnist_predictor.num_classes == EMNIST_NUM_CLASSES


def test_mnist_prediction_output_keys(mnist_predictor):
    """Prediction result dict should contain all expected keys."""
    img = Image.fromarray(np.zeros((28, 28), dtype=np.uint8), mode='L')
    result = mnist_predictor.predict(img)
    assert 'label' in result
    assert 'confidence' in result
    assert 'top_k' in result
    assert 'all_probs' in result
    assert 'preprocessed_image' in result


def test_mnist_probability_sum(mnist_predictor):
    """Output probabilities should sum to ~1.0."""
    img = Image.fromarray(np.random.randint(0, 255, (28, 28), dtype=np.uint8))
    result = mnist_predictor.predict(img)
    assert abs(result['all_probs'].sum() - 1.0) < 1e-4


def test_emnist_probability_shape(emnist_predictor):
    """EMNIST all_probs should have 47 elements."""
    img = Image.fromarray(np.random.randint(0, 255, (28, 28), dtype=np.uint8))
    result = emnist_predictor.predict(img)
    assert result['all_probs'].shape == (EMNIST_NUM_CLASSES,)


def test_top_k_count(mnist_predictor):
    """top_k=3 should return exactly 3 predictions."""
    img = Image.fromarray(np.zeros((28, 28), dtype=np.uint8), mode='L')
    result = mnist_predictor.predict(img, top_k=3)
    assert len(result['top_k']) == 3


def test_confidence_range(mnist_predictor):
    """Confidence should be between 0 and 100."""
    img = Image.fromarray(np.zeros((28, 28), dtype=np.uint8), mode='L')
    result = mnist_predictor.predict(img)
    assert 0.0 <= result['confidence'] <= 100.0


def test_label_is_string(mnist_predictor):
    """Predicted label should be a string."""
    img = Image.fromarray(np.zeros((28, 28), dtype=np.uint8), mode='L')
    result = mnist_predictor.predict(img)
    assert isinstance(result['label'], str)

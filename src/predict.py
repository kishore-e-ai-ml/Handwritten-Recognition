"""
Prediction module for Handwritten Character Recognition.
Provides a clean API to load models and predict from images.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Union, Dict, List, Tuple

from src.config import (
    MODELS_DIR,
    MNIST_CLASSES,
    EMNIST_BALANCED_CLASSES,
    MNIST_NUM_CLASSES,
    EMNIST_NUM_CLASSES
)
from src.model import build_model
from src.preprocessing import preprocess_image_for_model


class Predictor:
    """
    Wraps a trained CNN model for single-image prediction with confidence scores.
    """
    def __init__(self, dataset_type: str = 'mnist'):
        """
        Args:
            dataset_type: 'mnist' or 'emnist'
        """
        self.dataset_type = dataset_type.lower()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.dataset_type == 'mnist':
            self.num_classes = MNIST_NUM_CLASSES
            self.class_names = MNIST_CLASSES
            self.model_path = MODELS_DIR / "mnist_cnn.pt"
        else:
            self.num_classes = EMNIST_NUM_CLASSES
            self.class_names = EMNIST_BALANCED_CLASSES
            self.model_path = MODELS_DIR / "emnist_cnn.pt"

        self.model = self._load_model()

    def _load_model(self) -> torch.nn.Module:
        """Loads trained model checkpoint from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        model = build_model(num_classes=self.num_classes).to(self.device)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def predict(
        self,
        image_input: Union[Image.Image, np.ndarray, str],
        top_k: int = 3
    ) -> Dict:
        """
        Predicts character from image input.

        Args:
            image_input: PIL Image, numpy array (H,W) or (H,W,C), or file path string.
            top_k: Number of top predictions to return.

        Returns:
            dict with keys:
              - 'label': predicted class name (str)
              - 'class_index': predicted class index (int)
              - 'confidence': confidence in percent (float)
              - 'top_k': list of (label, confidence%) tuples
              - 'all_probs': numpy array of probabilities for all classes
              - 'preprocessed_image': normalized 28x28 numpy array for display
        """
        if image_input is None:
            raise ValueError("image_input cannot be None")

        tensor_input, preprocessed_np = preprocess_image_for_model(image_input, self.dataset_type)
        tensor_input = tensor_input.to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor_input)
            probs = F.softmax(outputs, dim=1)

        probs_np = probs.cpu().numpy()[0]
        pred_idx = int(np.argmax(probs_np))
        confidence = float(probs_np[pred_idx]) * 100

        top_k_indices = np.argsort(probs_np)[::-1][:top_k]
        top_k_results = [(self.class_names[i], float(probs_np[i]) * 100) for i in top_k_indices]

        return {
            'label': self.class_names[pred_idx],
            'class_index': pred_idx,
            'confidence': confidence,
            'top_k': top_k_results,
            'all_probs': probs_np,
            'preprocessed_image': preprocessed_np
        }


def predict_image(
    image_input: Union[Image.Image, np.ndarray, str],
    model: torch.nn.Module,
    class_names: List[str],
    dataset_type: str = 'mnist',
    top_k: int = 3,
    device: torch.device = None
) -> Dict:
    """
    Standalone prediction function using a pre-loaded model.

    Args:
        image_input: PIL Image, numpy array, or file path.
        model: Loaded PyTorch model in eval mode.
        class_names: List of class name strings.
        dataset_type: 'mnist' or 'emnist' for correct normalization.
        top_k: Number of top predictions.
        device: Torch device.

    Returns:
        Prediction result dict (same schema as Predictor.predict).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tensor_input, preprocessed_np = preprocess_image_for_model(image_input, dataset_type)
    tensor_input = tensor_input.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(tensor_input)
        probs = F.softmax(outputs, dim=1)

    probs_np = probs.cpu().numpy()[0]
    pred_idx = int(np.argmax(probs_np))
    confidence = float(probs_np[pred_idx]) * 100

    top_k_indices = np.argsort(probs_np)[::-1][:top_k]
    top_k_results = [(class_names[i], float(probs_np[i]) * 100) for i in top_k_indices]

    return {
        'label': class_names[pred_idx],
        'class_index': pred_idx,
        'confidence': confidence,
        'top_k': top_k_results,
        'all_probs': probs_np,
        'preprocessed_image': preprocessed_np
    }

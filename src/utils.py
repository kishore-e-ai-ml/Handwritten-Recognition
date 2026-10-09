"""
Utility functions: label mapping, plotting helpers, and reproducibility seed setup.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List

from src.config import MNIST_CLASSES, EMNIST_BALANCED_CLASSES


def set_seed(seed: int = 42):
    """Sets random seed for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_class_names(dataset_type: str) -> List[str]:
    """Returns the class name list for a given dataset type."""
    if dataset_type.lower() == 'mnist':
        return MNIST_CLASSES
    elif dataset_type.lower() == 'emnist':
        return EMNIST_BALANCED_CLASSES
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def plot_sample_grid(images: np.ndarray, labels: np.ndarray, class_names: List[str], dataset_name: str, save_path: Path = None, n: int = 25):
    """Plots a grid of sample images from the dataset with labels."""
    n = min(n, len(images))
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2 * rows))
    axes = axes.flatten()

    for i in range(n):
        axes[i].imshow(images[i].squeeze(), cmap='gray')
        axes[i].set_title(class_names[labels[i]], fontsize=9)
        axes[i].axis('off')
    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f"{dataset_name.upper()} Sample Images", fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

"""
Tests for data loading: dataset shapes, splits, class counts.
(No sklearn/pandas – uses only torch/numpy to avoid DLL issues on restricted systems)
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import torch
from src.data_loader import load_mnist_data, load_emnist_data
from src.config import MNIST_NUM_CLASSES, EMNIST_NUM_CLASSES


def test_mnist_data_loads():
    train_loader, val_loader, test_loader = load_mnist_data(batch_size=256, seed=42)
    assert train_loader is not None and val_loader is not None and test_loader is not None


def test_mnist_dataset_sizes():
    train_loader, val_loader, test_loader = load_mnist_data(batch_size=256, seed=42)
    assert len(train_loader.dataset) == 54000
    assert len(val_loader.dataset) == 6000
    assert len(test_loader.dataset) == 10000


def test_mnist_batch_shape():
    train_loader, _, _ = load_mnist_data(batch_size=64, seed=42)
    images, labels = next(iter(train_loader))
    assert images.shape == (64, 1, 28, 28)
    assert labels.min().item() >= 0
    assert labels.max().item() < MNIST_NUM_CLASSES


def test_emnist_data_loads():
    train_loader, val_loader, test_loader = load_emnist_data(split='balanced', batch_size=256, seed=42)
    assert train_loader is not None and test_loader is not None


def test_emnist_batch_shape():
    train_loader, _, _ = load_emnist_data(split='balanced', batch_size=64, seed=42)
    images, labels = next(iter(train_loader))
    assert images.shape == (64, 1, 28, 28)
    assert labels.min().item() >= 0
    assert labels.max().item() < EMNIST_NUM_CLASSES

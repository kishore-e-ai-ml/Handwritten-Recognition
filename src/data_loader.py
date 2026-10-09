"""
Data Loader Module for MNIST and EMNIST Datasets
Handles dataset download, train/validation/test splitting, and DataLoader creation.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.datasets import MNIST, EMNIST
import torchvision.transforms as transforms
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    BATCH_SIZE,
    RANDOM_SEED,
    VALIDATION_SPLIT,
    MNIST_NUM_CLASSES,
    EMNIST_NUM_CLASSES
)


class EMNISTTransposeTransform:
    """
    Custom transform to fix EMNIST default rotation/transposition issue.
    EMNIST images in raw datasets are rotated by 90 degrees / transposed.
    """
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            return img.transpose(-2, -1)
        elif isinstance(img, np.ndarray):
            return np.transpose(img, (1, 0))
        return img


def get_transforms(augment: bool = False):
    """
    Returns data transformations for training and testing.
    
    Args:
        augment: Whether to apply data augmentation for training.
        
    Returns:
        transform pipeline
    """
    if augment:
        return transforms.Compose([
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])


def load_mnist_data(
    data_dir: Path = RAW_DATA_DIR,
    batch_size: int = BATCH_SIZE,
    val_split: float = VALIDATION_SPLIT,
    seed: int = RANDOM_SEED
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads and prepares MNIST dataset (digits 0-9).
    
    Args:
        data_dir: Path to raw data directory
        batch_size: DataLoader batch size
        val_split: Proportion of training data for validation
        seed: Random seed for reproducibility
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_transform = get_transforms(augment=True)
    test_transform = get_transforms(augment=False)
    
    # Load raw datasets
    full_train_dataset = MNIST(root=str(data_dir), train=True, download=True, transform=train_transform)
    test_dataset = MNIST(root=str(data_dir), train=False, download=True, transform=test_transform)
    
    # Validation split
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size], generator=generator
    )
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"MNIST Data Loaded successfully:", flush=True)
    print(f"  Train samples: {len(train_dataset)}", flush=True)
    print(f"  Validation samples: {len(val_dataset)}", flush=True)
    print(f"  Test samples: {len(test_dataset)}", flush=True)
    
    return train_loader, val_loader, test_loader


def load_emnist_data(
    data_dir: Path = RAW_DATA_DIR,
    split: str = 'balanced',
    batch_size: int = BATCH_SIZE,
    val_split: float = VALIDATION_SPLIT,
    seed: int = RANDOM_SEED
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads and prepares EMNIST dataset with transpose transformation.
    
    Args:
        data_dir: Path to raw data directory
        split: EMNIST split variant ('balanced', 'digits', 'letters', 'byclass', 'bymerge')
        batch_size: DataLoader batch size
        val_split: Proportion of training data for validation
        seed: Random seed for reproducibility
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_transform = transforms.Compose([
        EMNISTTransposeTransform(),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.1751,), (0.3267,))
    ])
    
    test_transform = transforms.Compose([
        EMNISTTransposeTransform(),
        transforms.ToTensor(),
        transforms.Normalize((0.1751,), (0.3267,))
    ])
    
    # Load raw EMNIST datasets
    full_train_dataset = EMNIST(root=str(data_dir), split=split, train=True, download=True, transform=train_transform)
    test_dataset = EMNIST(root=str(data_dir), split=split, train=False, download=True, transform=test_transform)
    
    # Validation split
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size], generator=generator
    )
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"EMNIST ({split}) Data Loaded successfully:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

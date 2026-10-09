"""
Convolutional Neural Network Architecture for Handwritten Character Recognition.
Includes feature extraction blocks with BatchNorm, MaxPool, Dropout, and Dense layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class HandwrittenCNN(nn.Module):
    """
    CNN Architecture for Handwritten Digits (MNIST) and Characters (EMNIST).
    """
    def __init__(self, num_classes: int = 10):
        super(HandwrittenCNN, self).__init__()
        self.num_classes = num_classes
        
        # Conv Block 1
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        
        # Conv Block 2
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)
        
        # Fully Connected Block
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.5)
        self.out = nn.Linear(128, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Returns raw logits.
        """
        # Block 1
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Dense Block
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout3(x)
        logits = self.out(x)
        
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Softmax activation for probabilities.
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


def build_model(num_classes: int = 10) -> HandwrittenCNN:
    """
    Factory function to build HandwrittenCNN model instance.
    
    Args:
        num_classes: Number of target classes (10 for MNIST, 47 for EMNIST Balanced)
        
    Returns:
        Instantiated PyTorch model
    """
    model = HandwrittenCNN(num_classes=num_classes)
    return model


def count_parameters(model: nn.Module) -> int:
    """
    Counts trainable parameters in model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

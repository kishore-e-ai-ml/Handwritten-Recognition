"""
Model Training Script for MNIST and EMNIST CNNs.
Implements training loop, learning rate decay, early stopping, checkpointing, and metrics tracking.
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Tuple, Dict, Any

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.config import (
    RAW_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PATIENCE,
    RANDOM_SEED,
    MNIST_NUM_CLASSES,
    EMNIST_NUM_CLASSES
)
from src.data_loader import load_mnist_data, load_emnist_data
from src.model import build_model, count_parameters


class EarlyStopping:
    """Early stopping handler based on validation loss."""
    def __init__(self, patience: int = PATIENCE, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def train_epoch(model, loader, criterion, optimizer, device):
    """Runs single training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total += targets.size(0)
        correct += preds.eq(targets).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate_epoch(model, loader, criterion, device):
    """Runs evaluation on validation set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += targets.size(0)
            correct += preds.eq(targets).sum().item()
            
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_model(
    dataset_name: str = 'mnist',
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    seed: int = RANDOM_SEED
) -> Tuple[nn.Module, dict]:
    """
    Trains CNN model for selected dataset.
    
    Args:
        dataset_name: 'mnist' or 'emnist'
        epochs: Maximum epochs
        batch_size: Batch size
        lr: Initial learning rate
        seed: Random seed
        
    Returns:
        (trained_model, history_dict)
    """
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n==========================================", flush=True)
    print(f" Starting Training for Dataset: {dataset_name.upper()}", flush=True)
    print(f" Device: {device}", flush=True)
    print(f"==========================================", flush=True)
    
    # 1. Load Data
    if dataset_name.lower() == 'mnist':
        num_classes = MNIST_NUM_CLASSES
        train_loader, val_loader, test_loader = load_mnist_data(
            batch_size=batch_size, seed=seed
        )
        model_save_path = MODELS_DIR / "mnist_cnn.pt"
    elif dataset_name.lower() == 'emnist':
        num_classes = EMNIST_NUM_CLASSES
        train_loader, val_loader, test_loader = load_emnist_data(
            split='balanced', batch_size=batch_size, seed=seed
        )
        model_save_path = MODELS_DIR / "emnist_cnn.pt"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    # 2. Build Model
    model = build_model(num_classes=num_classes).to(device)
    print(f"Model parameters: {count_parameters(model):,}", flush=True)
    
    # 3. Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    early_stopping = EarlyStopping(patience=PATIENCE)
    
    # History recording
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device)
        
        # Step LR scheduler
        scheduler.step(val_loss)
        
        # Log history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%", flush=True)
        
        # Save best model checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'num_classes': num_classes,
                'val_acc': val_acc,
                'epoch': epoch
            }, model_save_path)
            print(f" --> Saved best checkpoint to {model_save_path.name} (Val Acc: {val_acc*100:.2f}%)", flush=True)
            
        # Early Stopping
        if early_stopping(val_loss):
            print(f"Early stopping triggered at epoch {epoch}")
            break
            
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes. Best Val Acc: {best_val_acc*100:.2f}%")
    
    # Save training history to JSON file
    history_file = REPORTS_DIR / f"history_{dataset_name}.json"
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=4)
        
    # Load best model for returning
    checkpoint = torch.load(model_save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train CNN for Handwritten Character Recognition")
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'emnist'], help="Dataset to train")
    parser.add_argument('--epochs', type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help="Learning rate")
    args = parser.parse_args()
    
    train_model(dataset_name=args.dataset, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

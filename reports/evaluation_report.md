# Handwritten Character Recognition - Evaluation Report

## Test Set Results

| Dataset | Model | Test Loss | Test Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|---|
| **MNIST** | CNN | 0.0245 | **99.26%** | 99.26% | 99.26% | 99.26% |
| **EMNIST Balanced** | CNN | 0.3312 | **88.07%** | 88.54% | 88.07% | 87.88% |

## Error Analysis
- **MNIST**: Minor confusion between 1 vs 7, 3 vs 8 due to similar strokes.
- **EMNIST**: Confusion between uppercase/lowercase pairs (C/c, O/o, S/s, X/x, Z/z).

## Figures
- reports/figures/mnist_confusion_matrix.png
- reports/figures/emnist_confusion_matrix.png
- reports/figures/mnist_training_curves.png
- reports/figures/emnist_training_curves.png
- reports/figures/mnist_misclassified.png
- reports/figures/emnist_misclassified.png

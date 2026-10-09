# Handwritten Character Recognition Using Convolutional Neural Networks

> **Internship Project – Task 3** | Deep Learning | Computer Vision

---

## 1. Project Overview

A complete end-to-end deep learning system for recognizing handwritten digits and alphabets using Convolutional Neural Networks (CNNs). The system is trained on real datasets (MNIST and EMNIST), evaluated with standard metrics, and deployed as an interactive Streamlit web application.

---

## 2. Problem Statement

Handwritten character recognition (HCR) is a classic computer vision problem with applications in:
- Digitizing handwritten documents
- Postal code recognition
- Bank cheque processing
- Educational assistive technology

---

## 3. Objective

Build a computer vision pipeline that:
1. Loads and preprocesses real handwritten image datasets
2. Trains CNN models with proper augmentation and regularization
3. Evaluates models with classification metrics and visualizations
4. Provides saliency-based explainability
5. Deploys an interactive web application with drawing canvas and file upload

---

## 4. Features

- ✅ Real MNIST and EMNIST Balanced dataset training
- ✅ CNN with BatchNorm, Dropout, MaxPooling
- ✅ Data augmentation (rotation, affine, scaling)
- ✅ EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- ✅ Confusion matrices, training curves, misclassification analysis
- ✅ Saliency map explainability (input-gradient visualization)
- ✅ Streamlit app with drawing canvas + file upload
- ✅ Top-3 predictions with confidence scores
- ✅ Modular Python codebase with automated tests

---

## 5. Datasets

### MNIST (Digits)
- **Source**: `torchvision.datasets.MNIST`
- **Classes**: 10 (digits 0–9)
- **Total samples**: 70,000 (60,000 train + 10,000 test)
- **Split used**: 54,000 train | 6,000 validation | 10,000 test
- **Image size**: 28×28 grayscale

### EMNIST Balanced
- **Source**: `torchvision.datasets.EMNIST(split='balanced')`
- **Classes**: 47 (digits 0–9, uppercase A–Z, 11 merged lowercase letters)
- **Total samples**: 131,600 (112,800 train + 18,800 test)
- **Split used**: 101,520 train | 11,280 validation | 18,800 test
- **Image size**: 28×28 grayscale (with transpose fix applied)

> **EMNIST Label Mapping**: The 47 classes map as: 0–9 (digits), 10–35 (A–Z), 36–46 (a, b, d, e, f, g, h, n, q, r, t). Lowercase letters with shapes identical to their uppercase counterparts are merged.

---

## 6. CNN Architecture

```
Input  (1 × 28 × 28)
  │
  ├─ Conv Block 1
  │    Conv2D(32, 3×3, pad=1) → BatchNorm2D → ReLU
  │    Conv2D(32, 3×3, pad=1) → BatchNorm2D → ReLU
  │    MaxPool2D(2×2) → Dropout(0.25)
  │
  ├─ Conv Block 2
  │    Conv2D(64, 3×3, pad=1) → BatchNorm2D → ReLU
  │    Conv2D(64, 3×3, pad=1) → BatchNorm2D → ReLU
  │    MaxPool2D(2×2) → Dropout(0.25)
  │
  ├─ Flatten → (64 × 7 × 7 = 3136)
  │
  ├─ Dense Block
  │    Linear(3136 → 128) → BatchNorm1D → ReLU → Dropout(0.5)
  │
  └─ Output: Linear(128 → num_classes) → Softmax

MNIST Parameters:  468,458
EMNIST Parameters: 473,231
```

---

## 7. Image Preprocessing

1. **Grayscale conversion** – All images processed as single-channel (L mode)
2. **EMNIST transpose fix** – Raw EMNIST images are transposed/flipped; corrected via `EMNISTTransposeTransform`
3. **Pixel normalization** – `[0, 255] → [0.0, 1.0]` then normalized:
   - MNIST: `mean=0.1307, std=0.3081`
   - EMNIST: `mean=0.1751, std=0.3267`
4. **User image preprocessing** (inference):
   - Grayscale conversion → color inversion check
   - Bounding box crop → center-of-mass centering in 28×28 canvas
   - Same normalization as training

---

## 8. Data Augmentation

Applied **only to training data**:

```python
transforms.RandomRotation(degrees=10)          # ±10° rotation
transforms.RandomAffine(translate=(0.08, 0.08), scale=(0.95, 1.05))  # shift + scale
transforms.Normalize(mean, std)
```

Test/validation sets use only normalization (no augmentation).

---

## 9. Training Process

| Parameter | Value |
|---|---|
| Optimizer | Adam (lr=0.001, weight_decay=1e-4) |
| Loss | CrossEntropyLoss |
| Batch Size | 128 (MNIST), 256 (EMNIST) |
| Epochs | Up to 15 (early stopping) |
| EarlyStopping patience | 4 epochs |
| ReduceLROnPlateau | factor=0.5, patience=2 |
| Best model saved on | Validation Accuracy |

---

## 10. Evaluation Metrics

All metrics computed on the **untouched test set**:
- Test Loss (Cross-Entropy)
- Test Accuracy
- Precision (weighted average)
- Recall (weighted average)
- F1-Score (weighted average)

---

## 11. Results

| Dataset | Model | Test Loss | Test Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|---|
| **MNIST** | CNN | 0.0245 | **99.26%** | 99.26% | 99.26% | 99.26% |
| **EMNIST Balanced** | CNN | 0.3312 | **88.07%** | 88.54% | 88.07% | 87.88% |

---

## 12. Error Analysis

### MNIST Confusion Patterns
- **1 vs 7**: Diagonal stroke of 7 sometimes resembles upright 1
- **3 vs 8**: Closed loops in 8 vs open curves in 3
- **4 vs 9**: Similar top portion

### EMNIST Confusion Patterns
- **C vs c**, **O vs o**, **S vs s**: Uppercase and lowercase nearly identical at 28×28
- **I vs l vs 1**: Thin vertical strokes with no serifs
- **Z vs z**: Same shape, only size difference

---

## 13. Explainability

**Saliency Maps (Input-Gradient Visualization)**:
- Computes the absolute gradient of the predicted class score with respect to the input pixel
- Highlights which pixels the CNN focuses on when making a prediction
- Shown as: Input Image | Saliency Heatmap | Overlay

Saved to: `reports/figures/explainability/`

---

## 14. Streamlit Application

**Features**:
- 🖊️ Interactive drawing canvas (mouse/touch)
- 📁 Image file upload (PNG, JPG, BMP)
- Mode toggle: MNIST Digits vs EMNIST Characters
- Predicted character + confidence percentage
- Top-3 predictions bar chart
- Preprocessed 28×28 image view
- Saliency map explainability toggle

---

## 15. Project Structure

```
handwritten-character-recognition/
├── data/
│   ├── raw/           # Downloaded datasets (auto-downloaded)
│   └── processed/
├── models/
│   ├── mnist_cnn.pt   # Trained MNIST model checkpoint
│   └── emnist_cnn.pt  # Trained EMNIST model checkpoint
├── reports/
│   ├── figures/
│   │   ├── explainability/     # Saliency maps
│   │   ├── mnist_confusion_matrix.png
│   │   ├── emnist_confusion_matrix.png
│   │   ├── mnist_training_curves.png
│   │   ├── emnist_training_curves.png
│   │   ├── mnist_misclassified.png
│   │   └── emnist_misclassified.png
│   ├── evaluation_report.md
│   └── results.csv
├── src/
│   ├── config.py          # All configuration (paths, hyperparams, class maps)
│   ├── data_loader.py     # MNIST & EMNIST DataLoaders
│   ├── preprocessing.py   # Image preprocessing for inference
│   ├── model.py           # CNN architecture
│   ├── train.py           # Training pipeline
│   ├── evaluate.py        # Evaluation, metrics, plots
│   ├── predict.py         # Predictor class + predict_image()
│   ├── explainability.py  # Saliency map generation
│   └── utils.py           # Helpers (seed, class names, plotting)
├── app/
│   └── app.py             # Streamlit web application
├── tests/
│   ├── test_data.py       # Dataset loading tests
│   ├── test_model.py      # Model architecture + loading tests
│   └── test_prediction.py # Preprocessing + prediction tests
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 16. Installation

```bash
# Clone the project
cd "d:/handwritten recognition"

# Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 17. Training Instructions

```bash
# Train MNIST model
python src/train.py --dataset mnist --epochs 15

# Train EMNIST model
python src/train.py --dataset emnist --epochs 15 --batch_size 256

# Run full evaluation + generate all plots
python src/evaluate.py
```

---

## 18. Testing Instructions

```bash
# Run all automated tests
python -m pytest tests/ -v
```

---

## 19. Running the Application

```bash
# Launch Streamlit web app
streamlit run app/app.py
```

Open in browser: `http://localhost:8501`

---

## 20. Example Usage

**From Python**:
```python
from src.predict import Predictor
from PIL import Image

predictor = Predictor(dataset_type='mnist')
img = Image.open('my_digit.png')
result = predictor.predict(img, top_k=3)

print(f"Predicted: {result['label']}")
print(f"Confidence: {result['confidence']:.1f}%")
print(f"Top-3: {result['top_k']}")
```

---

## 21. Limitations

- Models trained on 28×28 grayscale images – poor performance on colored or very noisy scans
- EMNIST at 28×28 cannot distinguish visually identical uppercase/lowercase pairs
- Canvas preprocessing quality depends on stroke thickness and centering
- CPU-only inference (no GPU acceleration in default setup)

---

## 22. Future Improvements

- **CRNN for word/sentence recognition**: Replace single-character CNN with a CNN → BiLSTM → CTC pipeline for sequence-level handwritten text recognition
- **Attention mechanisms**: Add spatial attention to improve character localization
- **Transfer learning**: Fine-tune on domain-specific handwriting styles
- **Data collection**: Augment with real-world scanned document data

### CRNN Architecture (Future Work)
```
Image → CNN (feature extractor)
      → Reshape to sequence (time steps)
      → BiLSTM (sequence modeling)
      → CTC Decoder
      → Character sequence output
```

---

## 23. Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.12 | Core programming language |
| PyTorch 2.14 | CNN model training & inference |
| torchvision | Dataset loading (MNIST, EMNIST) |
| NumPy | Numerical computation |
| Matplotlib / Seaborn | Visualization & plots |
| OpenCV | Image preprocessing |
| Streamlit 1.64 | Interactive web application |
| streamlit-drawable-canvas | Drawing canvas widget |
| Pillow | Image I/O |
| pytest | Automated testing |

---

## 24. Internship Task Mapping

| Requirement | Status | Implementation |
|---|---|---|
| MNIST digit recognition | ✅ Done | `src/train.py --dataset mnist` |
| EMNIST character recognition | ✅ Done | `src/train.py --dataset emnist` |
| CNN architecture | ✅ Done | `src/model.py` (HandwrittenCNN) |
| Image preprocessing | ✅ Done | `src/preprocessing.py` |
| Data augmentation | ✅ Done | Training transforms in `data_loader.py` |
| Train/Val/Test split | ✅ Done | 90%/10% split from torchvision |
| Model evaluation | ✅ Done | `src/evaluate.py` |
| Confusion matrix | ✅ Done | `reports/figures/` |
| Error analysis | ✅ Done | Misclassified grids + discussion |
| Explainability | ✅ Done | Saliency maps in `src/explainability.py` |
| Interactive app | ✅ Done | `app/app.py` (Streamlit) |
| Automated tests | ✅ Done | `tests/` (pytest) |
| CRNN extension (future) | ✅ Documented | README Section 22 |

---

## 25. Author

**B.Tech AI/ML Student – Internship Task 3**
Handwritten Character Recognition Using Convolutional Neural Networks

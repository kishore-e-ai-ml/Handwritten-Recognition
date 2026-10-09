"""
Explainability module: Saliency/Input-gradient visualization for CNN predictions.
Highlights which pixels of the input image most influence the prediction.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Union

from src.config import EXPLAINABILITY_DIR
from src.preprocessing import preprocess_image_for_model


def compute_saliency_map(
    model: torch.nn.Module,
    tensor_input: torch.Tensor,
    target_class: int = None
) -> np.ndarray:
    """
    Computes input-gradient (saliency) map showing pixel importance.

    Args:
        model: Loaded PyTorch model.
        tensor_input: Preprocessed image tensor shape (1, 1, 28, 28).
        target_class: Target class index. If None, uses predicted class.

    Returns:
        saliency: numpy array (28, 28) with absolute gradient values normalized [0,1].
    """
    model.eval()
    inp = tensor_input.clone().detach().requires_grad_(True)

    output = model(inp)
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    model.zero_grad()
    score = output[0, target_class]
    score.backward()

    saliency = inp.grad.data.abs().squeeze().cpu().numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency


def save_saliency_visualization(
    original_np: np.ndarray,
    saliency: np.ndarray,
    predicted_label: str,
    confidence: float,
    save_path: Path,
    title: str = "Saliency Map"
):
    """
    Saves a side-by-side visualization: original image | saliency heatmap | overlay.

    Args:
        original_np: (28,28) float numpy array [0,1].
        saliency: (28,28) float numpy array [0,1].
        predicted_label: Predicted character string.
        confidence: Prediction confidence percent.
        save_path: File path to save PNG.
        title: Plot title.
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    axes[0].imshow(original_np, cmap='gray')
    axes[0].set_title('Input Image', fontsize=11)
    axes[0].axis('off')

    heatmap = axes[1].imshow(saliency, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Saliency Map\n(Pixel Importance)', fontsize=11)
    axes[1].axis('off')
    plt.colorbar(heatmap, ax=axes[1], fraction=0.046)

    # Overlay: blend original grayscale with saliency heatmap
    orig_rgb = np.stack([original_np] * 3, axis=-1)
    hot_cmap = plt.colormaps['hot']
    sal_rgb = hot_cmap(saliency)[:, :, :3]
    overlay = 0.5 * orig_rgb + 0.5 * sal_rgb
    overlay = np.clip(overlay, 0, 1)
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay', fontsize=11)
    axes[2].axis('off')

    fig.suptitle(
        f"{title}\nPredicted: '{predicted_label}' | Confidence: {confidence:.1f}%",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_explainability_for_samples(
    model: torch.nn.Module,
    test_loader,
    class_names: list,
    dataset_name: str,
    num_samples: int = 4,
    device: torch.device = None
):
    """
    Generates and saves saliency visualizations for a few test samples.

    Args:
        model: Loaded PyTorch model.
        test_loader: DataLoader for test set.
        class_names: List of class names.
        dataset_name: 'mnist' or 'emnist'.
        num_samples: Number of sample explanations to generate.
        device: Torch device.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.eval()

    count = 0
    for images, targets in test_loader:
        for i in range(images.size(0)):
            if count >= num_samples:
                break
            inp = images[i:i+1].to(device)
            with torch.no_grad():
                probs = F.softmax(model(inp), dim=1)
            pred_idx = probs.argmax(dim=1).item()
            confidence = probs[0, pred_idx].item() * 100

            saliency = compute_saliency_map(model, inp, target_class=pred_idx)
            orig_np = images[i].squeeze().cpu().numpy()
            # Denormalize for display
            orig_display = np.clip(orig_np * 0.3081 + 0.1307, 0, 1) if dataset_name == 'mnist' else np.clip(orig_np * 0.3267 + 0.1751, 0, 1)

            save_path = EXPLAINABILITY_DIR / f"{dataset_name}_saliency_sample_{count+1}.png"
            save_saliency_visualization(
                orig_display, saliency,
                predicted_label=class_names[pred_idx],
                confidence=confidence,
                save_path=save_path,
                title=f"{dataset_name.upper()} Saliency Map"
            )
            count += 1
        if count >= num_samples:
            break

    print(f"Saved {count} saliency visualizations to {EXPLAINABILITY_DIR}")

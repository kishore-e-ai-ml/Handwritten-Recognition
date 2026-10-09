"""
Preprocessing module for user-drawn or uploaded images.
Includes grayscale conversion, color inversion, bounding box cropping, center of mass centering,
and PyTorch tensor normalization matching dataset training distribution.
"""

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import cv2
from typing import Tuple, Union


def crop_and_center_character(image: Image.Image, target_size: int = 28) -> Image.Image:
    """
    Crops the character bounding box and centers it in a target_size x target_size canvas
    matching MNIST format (20x20 bounding box centered in 28x28 image).
    
    Args:
        image: PIL Image in grayscale (white text on black background).
        target_size: Target image dimension (28).
        
    Returns:
        Processed PIL Image (28x28).
    """
    np_img = np.array(image)
    
    # Threshold image to locate character pixels
    _, thresh = cv2.threshold(np_img, 30, 255, cv2.THRESH_BINARY)
    
    coords = cv2.findNonZero(thresh)
    if coords is None:
        # Blank drawing fallback
        return image.resize((target_size, target_size), Image.Resampling.BILINEAR)
    
    x, y, w, h = cv2.boundingRect(coords)
    
    # Extract bounding box
    cropped = np_img[y:y+h, x:x+w]
    
    # Scale bounding box so max dimension is 20 pixels (preserving aspect ratio)
    box_size = 20
    if h > w:
        new_h = box_size
        new_w = max(1, int(w * (box_size / h)))
    else:
        new_w = box_size
        new_h = max(1, int(h * (box_size / w)))
        
    cropped_pil = Image.fromarray(cropped)
    resized_cropped = cropped_pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    # Create blank 28x28 black image and paste character in center
    centered = Image.new('L', (target_size, target_size), color=0)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    centered.paste(resized_cropped, (pad_x, pad_y))
    
    return centered


def preprocess_image_for_model(
    image_input: Union[Image.Image, np.ndarray, str],
    dataset_type: str = 'mnist'
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Preprocesses user image for model inference.
    
    Args:
        image_input: PIL Image, numpy array (canvas/upload), or image file path.
        dataset_type: 'mnist' or 'emnist'.
        
    Returns:
        (processed_tensor, normalized_numpy_array)
    """
    # 1. Load image to PIL
    if isinstance(image_input, str):
        pil_img = Image.open(image_input).convert('L')
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 3 and image_input.shape[2] == 4:
            # Composite RGBA over black; the alpha channel may be opaque across
            # the whole canvas and therefore cannot be used as the drawing.
            rgba = image_input.astype(np.float32)
            rgb = rgba[:, :, :3] * (rgba[:, :, 3:4] / 255.0)
            gray = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            pil_img = Image.fromarray(gray)
        elif len(image_input.shape) == 3:
            gray = cv2.cvtColor(image_input, cv2.COLOR_RGB2GRAY)
            pil_img = Image.fromarray(gray)
        else:
            pil_img = Image.fromarray(image_input)
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert('L')
    else:
        raise ValueError(f"Unsupported image type: {type(image_input)}")
        
    # 2. Color Inversion check: MNIST expects white character (high intensity) on black background (low intensity)
    np_raw = np.array(pil_img)
    if np.mean(np_raw) > 127:
        pil_img = ImageOps.invert(pil_img)
        
    # 3. Crop and Center Character
    centered_img = crop_and_center_character(pil_img, target_size=28)
    
    # 4. Convert to NumPy and Normalize [0, 1]
    np_centered = np.array(centered_img, dtype=np.float32) / 255.0
    
    # 5. Apply normalization transform matching dataset
    if dataset_type.lower() == 'emnist':
        mean, std = 0.1751, 0.3267
    else:
        mean, std = 0.1307, 0.3081
        
    norm_np = (np_centered - mean) / std
    
    # Shape: (1, 1, 28, 28) PyTorch BCHW Tensor
    tensor_input = torch.tensor(norm_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    return tensor_input, np_centered

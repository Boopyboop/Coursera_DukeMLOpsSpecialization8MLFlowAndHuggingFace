"""Utilities for OCR project: image preprocessing and feature extraction."""

from PIL import Image
import numpy as np


def image_to_feature_vector(image_path: str, size=(28, 28)) -> np.ndarray:
    """
    Convert an image to a flattened grayscale feature vector.
    This is intentionally simple for demo/test purposes.
    """
    img = Image.open(image_path).convert("L")
    img = img.resize(size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.flatten()

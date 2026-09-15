"""Shared package for the handwritten digit recognition project.

Created: 2025-09-15
Both the desktop version and the web version import the model definition and
the image preprocessing helpers from this package, so the two front ends always
use exactly the same inference pipeline.
"""

from .model import DigitCNN, MODEL_PATH, load_model
from .preprocess import canvas_to_tensor, has_ink, predict_digit, preprocess_image

__all__ = [
    "DigitCNN",
    "MODEL_PATH",
    "load_model",
    "canvas_to_tensor",
    "has_ink",
    "preprocess_image",
    "predict_digit",
]

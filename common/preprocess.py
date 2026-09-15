"""Turn a user drawing into an MNIST-style 28x28 tensor and run inference.

Created: 2025-09-15
MNIST digits are not raw 28x28 crops: every digit was size-normalized to fit a
20x20 box and then centered inside a 28x28 frame by its center of mass. A
drawing that skips those two steps looks nothing like the training data, so the
same normalization is reproduced here for both front ends.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps

from .model import MNIST_MEAN, MNIST_STD

# Size of the box the digit strokes are scaled into, and the final frame size.
DIGIT_BOX = 20
CANVAS_SIZE = 28

# Pixels below this value (0-255 scale) are treated as background.
INK_THRESHOLD = 20


def _to_white_on_black(image: Image.Image) -> Image.Image:
    """Return a grayscale image whose strokes are bright on a dark background.

    Users draw black ink on a white canvas, while MNIST stores white ink on a
    black background, so the image is inverted when the border pixels are
    bright.
    """
    gray = image.convert("L")
    pixels = np.asarray(gray, dtype=np.float32)

    # Sample the four borders: a white canvas has a bright frame.
    border = np.concatenate(
        [pixels[0, :], pixels[-1, :], pixels[:, 0], pixels[:, -1]]
    )
    if border.mean() > 127:
        gray = ImageOps.invert(gray)

    return gray


def _center_by_mass(digit: Image.Image) -> Image.Image:
    """Paste the digit into a 28x28 frame, aligning its center of mass."""
    frame = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)

    pixels = np.asarray(digit, dtype=np.float32)
    total = pixels.sum()
    if total == 0:
        # Nothing was drawn; fall back to a plain geometric centering.
        offset_x = (CANVAS_SIZE - digit.width) // 2
        offset_y = (CANVAS_SIZE - digit.height) // 2
        frame.paste(digit, (offset_x, offset_y))
        return frame

    rows = np.arange(pixels.shape[0], dtype=np.float32)
    cols = np.arange(pixels.shape[1], dtype=np.float32)
    center_y = float((pixels.sum(axis=1) * rows).sum() / total)
    center_x = float((pixels.sum(axis=0) * cols).sum() / total)

    # Shift so that the center of mass lands on the middle of the frame.
    offset_x = int(round(CANVAS_SIZE / 2.0 - center_x))
    offset_y = int(round(CANVAS_SIZE / 2.0 - center_y))

    # Keep the digit fully inside the frame.
    offset_x = max(0, min(offset_x, CANVAS_SIZE - digit.width))
    offset_y = max(0, min(offset_y, CANVAS_SIZE - digit.height))

    frame.paste(digit, (offset_x, offset_y))
    return frame


def has_ink(image: Image.Image) -> bool:
    """Return True when the image actually contains strokes.

    An empty canvas still produces a confident-looking prediction, so callers
    check this first and report "draw something" instead of a wrong digit.
    """
    gray = _to_white_on_black(image)
    cleaned = gray.point(lambda value: value if value > INK_THRESHOLD else 0)
    return cleaned.getbbox() is not None


def preprocess_image(image: Image.Image) -> Image.Image:
    """Convert an arbitrary drawing into a 28x28 MNIST-style image.

    Steps: invert to white-on-black, crop to the ink bounding box, scale the
    longer side to 20 pixels, then center the result by center of mass.
    """
    gray = _to_white_on_black(image)

    # Drop faint anti-aliasing noise before measuring the bounding box.
    cleaned = gray.point(lambda value: value if value > INK_THRESHOLD else 0)

    bbox = cleaned.getbbox()
    if bbox is None:
        # Empty canvas: return a blank frame so the caller can report low
        # confidence instead of crashing.
        return Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)

    digit = cleaned.crop(bbox)

    # Scale the longer side to 20 pixels and keep the aspect ratio.
    width, height = digit.size
    scale = DIGIT_BOX / float(max(width, height))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    digit = digit.resize((new_width, new_height), Image.LANCZOS)

    return _center_by_mass(digit)


def canvas_to_tensor(image: Image.Image) -> Tuple[torch.Tensor, Image.Image]:
    """Return a normalized 1x1x28x28 tensor plus the 28x28 preview image."""
    processed = preprocess_image(image)

    pixels = np.asarray(processed, dtype=np.float32) / 255.0
    normalized = (pixels - MNIST_MEAN) / MNIST_STD

    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    return tensor, processed


def predict_digit(
    model: torch.nn.Module, image: Image.Image
) -> Tuple[int, float, List[float], Image.Image]:
    """Predict the digit drawn in `image`.

    Returns:
        A tuple of (predicted digit, confidence in percent, per-class
        probabilities in percent, the 28x28 preview image).
    """
    tensor, processed = canvas_to_tensor(image)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    digit = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[digit].item()) * 100.0
    all_scores = [float(value) * 100.0 for value in probabilities]

    return digit, confidence, all_scores, processed

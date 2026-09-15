"""CNN model definition and checkpoint loading.

Created: 2025-09-15
The network is a small two-block convolutional neural network that reaches
about 99% accuracy on the MNIST test set after a few epochs on a CPU.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Project root is the parent directory of this package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Trained weights are shared by the desktop version and the web version.
MODEL_PATH = PROJECT_ROOT / "model" / "mnist_cnn.pt"

# MNIST normalization constants (dataset mean and standard deviation).
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class DigitCNN(nn.Module):
    """Two convolution blocks followed by a small classifier head."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        # Block 1: 28x28 -> 14x14, 32 feature maps.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # Block 2: 14x14 -> 7x7, 64 feature maps.
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.dropout_conv = nn.Dropout(0.25)
        self.dropout_fc = nn.Dropout(0.5)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits for a batch of 1x28x28 images."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout_conv(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout_conv(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        return self.fc2(x)


def load_model(model_path: Path | str = MODEL_PATH, device: str = "cpu") -> DigitCNN:
    """Load the trained checkpoint and return the model in evaluation mode.

    Raises:
        FileNotFoundError: If the checkpoint does not exist yet. The caller is
            expected to tell the user to run `python train_model.py` first.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'.\n"
            "Run 'python train_model.py' first to train and save the model."
        )

    model = DigitCNN()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

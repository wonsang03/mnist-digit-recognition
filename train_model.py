"""Train the MNIST digit classifier used by both front ends.

Created: 2025-09-15
Usage:
    python train_model.py                # train with the default settings
    python train_model.py --epochs 5     # train longer for a better score
    python train_model.py --force        # retrain even if a checkpoint exists

The trained weights are written to model/mnist_cnn.pt, which the desktop and
web versions load at start-up.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from common.model import MNIST_MEAN, MNIST_STD, MODEL_PATH, DigitCNN

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"


def build_dataloaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    """Download MNIST if needed and return the training and test loaders.

    Light random affine jitter is applied to the training split only. Digits
    drawn with a mouse are rarely as tidy as MNIST samples, and the jitter makes
    the model noticeably more forgiving of that.
    """
    train_transform = transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5
            ),
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )

    train_set = datasets.MNIST(
        root=str(DATA_DIR), train=True, download=True, transform=train_transform
    )
    test_set = datasets.MNIST(
        root=str(DATA_DIR), train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> float:
    """Run a single training epoch and return the average loss."""
    model.train()
    running_loss = 0.0
    batches = len(loader)

    for index, (images, labels) in enumerate(loader, start=1):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if index % 100 == 0 or index == batches:
            print(
                f"  epoch {epoch} | batch {index:>4}/{batches} | "
                f"loss {running_loss / index:.4f}",
                flush=True,
            )

    return running_loss / batches


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Return the accuracy on the test split as a percentage."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += labels.size(0)

    return 100.0 * correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MNIST digit classifier.")
    parser.add_argument("--epochs", type=int, default=3, help="number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--force", action="store_true", help="retrain even if a checkpoint exists"
    )
    args = parser.parse_args()

    # The launchers call this script every time, so skip the work when the
    # model is already on disk.
    if MODEL_PATH.exists() and not args.force:
        print(f"Model already exists at '{MODEL_PATH}'. Use --force to retrain.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Loading the MNIST dataset (it is downloaded on the first run)...")

    train_loader, test_loader = build_dataloaders(args.batch_size)
    print(
        f"Train samples: {len(train_loader.dataset):,} | "
        f"Test samples: {len(test_loader.dataset):,}"
    )

    model = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

    started = time.time()
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        average_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        accuracy = evaluate(model, test_loader, device)
        scheduler.step()
        print(f"  -> average loss {average_loss:.4f} | test accuracy {accuracy:.2f}%")

    elapsed = time.time() - started
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)

    print(f"\nTraining finished in {elapsed:.1f}s.")
    print(f"Saved the trained model to '{MODEL_PATH}'.")


if __name__ == "__main__":
    main()

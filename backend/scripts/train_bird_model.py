#!/usr/bin/env python3
"""Fine-tune ResNet-50 on a 30-class CUB-200-2011 subset.

Usage
-----
    cd backend
    python -m scripts.train_bird_model [--epochs 20] [--lr 0.001] [--batch-size 32]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from models.bird_model import CUB_SELECTED, CUB_ID_TO_LOCAL, NUM_CLASSES

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

# Paths
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
_CUB_DIR = os.path.join(_PROJECT_ROOT, "datasets", "CUB_200_2011")
_SAVE_DIR = os.path.join(_BACKEND_DIR, "data", "models")


class CUBSubsetDataset(Dataset):
    """Load CUB-200-2011 images for the selected 30-class subset."""

    def __init__(self, cub_dir: str, train: bool = True, transform=None):
        self.cub_dir = cub_dir
        self.transform = transform
        self.samples: list[tuple[str, int]] = []  # (image_path, local_label)

        # Read CUB metadata
        images = {}  # id → relative path
        with open(os.path.join(cub_dir, "images.txt")) as f:
            for line in f:
                img_id, path = line.strip().split()
                images[int(img_id)] = path

        labels = {}  # id → class_id
        with open(os.path.join(cub_dir, "image_class_labels.txt")) as f:
            for line in f:
                img_id, class_id = line.strip().split()
                labels[int(img_id)] = int(class_id)

        splits = {}  # id → is_train (1 or 0)
        with open(os.path.join(cub_dir, "train_test_split.txt")) as f:
            for line in f:
                img_id, is_train = line.strip().split()
                splits[int(img_id)] = int(is_train)

        # Filter to selected classes and split
        for img_id, rel_path in images.items():
            class_id = labels.get(img_id)
            if class_id not in CUB_ID_TO_LOCAL:
                continue
            is_train = splits.get(img_id, 1)
            if (train and is_train == 1) or (not train and is_train == 0):
                full_path = os.path.join(cub_dir, "images", rel_path)
                self.samples.append((full_path, CUB_ID_TO_LOCAL[class_id]))

        logger.info("CUB %s: %d images across %d classes",
                     "train" if train else "test", len(self.samples), NUM_CLASSES)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Datasets
    train_ds = CUBSubsetDataset(_CUB_DIR, train=True, transform=train_transform)
    test_ds = CUBSubsetDataset(_CUB_DIR, train=False, transform=test_transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    # Model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(2048, NUM_CLASSES)
    model.to(device)

    # Freeze early layers initially (conv1 + bn1 + layer1 + layer2)
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in ["conv1", "bn1", "layer1", "layer2"]):
            param.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    os.makedirs(_SAVE_DIR, exist_ok=True)
    save_path = os.path.join(_SAVE_DIR, "bird_resnet50.pth")

    for epoch in range(1, args.epochs + 1):
        # Unfreeze all at epoch 6
        if epoch == 6:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1,
                                         weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - 5)
            logger.info("Unfreezing all layers at epoch %d", epoch)

        # Train
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)
        scheduler.step()
        train_acc = correct / total

        # Evaluate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                correct += (logits.argmax(1) == labels).sum().item()
                total += imgs.size(0)
        test_acc = correct / total

        logger.info("Epoch %02d/%02d — loss: %.4f  train_acc: %.3f  test_acc: %.3f",
                     epoch, args.epochs, total_loss / len(train_ds), train_acc, test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            logger.info("  → saved best model (test_acc=%.3f)", best_acc)

    logger.info("Training complete. Best test accuracy: %.3f", best_acc)
    logger.info("Model saved to: %s", save_path)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune ResNet-50 on CUB-200 subset")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

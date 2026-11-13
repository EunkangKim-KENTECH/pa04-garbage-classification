"""Shared utilities: class names, seeding, transforms, dataset, and model."""
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms(train: bool):
    """Light augmentation for training; deterministic resize/crop for testing."""
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ImageListDataset(Dataset):
    """Dataset built from a DataFrame with columns [path, label].

    `path` is relative to `data_root`, e.g. "glass/glass12.jpg".
    """

    def __init__(self, df: pd.DataFrame, data_root: str, train: bool):
        self.paths = df["path"].tolist()
        self.labels = [CLASS_TO_IDX[c] for c in df["label"]]
        self.data_root = data_root
        self.transform = get_transforms(train)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = os.path.join(self.data_root, self.paths[idx])
        img = Image.open(path).convert("RGB")
        return self.transform(img), self.labels[idx]


def build_model(num_classes: int = len(CLASSES), pretrained: bool = True) -> nn.Module:
    """ResNet-18 (ImageNet-pretrained) with a new classification head."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)

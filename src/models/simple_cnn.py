"""Simple CNN model architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """Simple CNN adapted from PyTorch tutorial.

    Suitable for CIFAR-10/CIFAR-100 sized images (32x32).
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SimpleCNNLarge(nn.Module):
    """Larger Simple CNN for more complex tasks.

    Better suited for Tiny-ImageNet or when more capacity is needed.
    """

    def __init__(self, num_classes: int = 200, in_channels: int = 3, image_size: int = 64):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # calculate flattened size based on image size
        self.flat_size = 256 * (image_size // 16) * (image_size // 16)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.flat_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

"""CNN model architecture from the Ditto paper.

Paper: Ditto: Fair and Robust Federated Learning Through Personalization
       Li et al., 2021
       https://arxiv.org/abs/2012.04221

Architecture matches the Fashion MNIST / FEMNIST CNN from the official
implementation at github.com/litian96/ditto:
  Conv2d(in->32, 5x5, same padding) -> ReLU -> MaxPool(2x2)
  Conv2d(32->64, 5x5, same padding) -> ReLU -> MaxPool(2x2)
  Flatten -> Dense(3136->1024) -> ReLU
  Dense(1024->num_classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DittoCNN(nn.Module):
    """CNN matching the Ditto paper for Fashion MNIST / FEMNIST.

    Input: [B, in_channels, 28, 28]
    Output: [B, num_classes]
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        # after two 2x2 max-pools on 28x28 input: 7x7 spatial size
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))  # [B, 32, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 64, 7, 7]
        x = x.view(x.size(0), -1)             # [B, 3136]
        x = F.relu(self.fc1(x))               # [B, 1024]
        return self.fc2(x)                     # [B, num_classes]

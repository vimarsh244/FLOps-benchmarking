"""Model architectures for federated learning."""

from src.models.registry import MODEL_REGISTRY, get_model
from src.models.resnet import ResNet18, ResNet34, ResNet50, ResNetCIFAR
from src.models.simple_cnn import SimpleCNN, SimpleCNNLarge
from src.models.vit import ViT

__all__ = [
    "SimpleCNN",
    "SimpleCNNLarge",
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNetCIFAR",
    "ViT",
    "get_model",
    "MODEL_REGISTRY",
]

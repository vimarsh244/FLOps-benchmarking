"""Model architectures for federated learning."""

from src.models.simple_cnn import SimpleCNN
from src.models.resnet import ResNet18, ResNet34, ResNet50
from src.models.vit import ViT
from src.models.registry import get_model, MODEL_REGISTRY

__all__ = [
    "SimpleCNN",
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ViT",
    "get_model",
    "MODEL_REGISTRY",
]


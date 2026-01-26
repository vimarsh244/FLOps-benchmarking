"""Model registry for easy model creation."""

from typing import Any, Dict, Optional, Type
import torch.nn as nn
from omegaconf import DictConfig

from src.models.simple_cnn import SimpleCNN, SimpleCNNLarge
from src.models.resnet import ResNet18, ResNet34, ResNet50, ResNetCIFAR
from src.models.vit import ViT, ViTSmall

# registry mapping model names to classes
MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "simplecnn": SimpleCNN,
    "simplecnn_large": SimpleCNNLarge,
    "resnet18": ResNet18,
    "resnet18_gn": ResNet18,  # with GroupNorm (via use_groupnorm=True in config)
    "resnet34": ResNet34,
    "resnet34_gn": ResNet34,  # with GroupNorm
    "resnet50": ResNet50,
    "resnet50_gn": ResNet50,  # with GroupNorm
    "resnet_cifar": ResNetCIFAR,
    "vit": ViT,
    "vit_small": ViTSmall,
}


def get_model(
    model_name: str,
    num_classes: int,
    in_channels: int = 3,
    image_size: int = 32,
    pretrained: bool = False,
    **kwargs: Any,
) -> nn.Module:
    """Create a model by name.

    Args:
        model_name: name of the model (must be in MODEL_REGISTRY)
        num_classes: number of output classes
        in_channels: number of input channels
        image_size: input image size
        pretrained: whether to use pretrained weights
        **kwargs: additional model-specific arguments

    Returns:
        Initialized model

    Raises:
        ValueError: if model_name is not in registry
    """
    model_name = model_name.lower()

    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")

    model_class = MODEL_REGISTRY[model_name]

    # build kwargs based on model type
    model_kwargs = {"num_classes": num_classes}

    if model_name in ["simplecnn"]:
        model_kwargs["in_channels"] = in_channels
    elif model_name in ["simplecnn_large"]:
        model_kwargs["in_channels"] = in_channels
        model_kwargs["image_size"] = image_size
    elif model_name.startswith("resnet"):
        model_kwargs["pretrained"] = pretrained
        model_kwargs["in_channels"] = in_channels
    elif model_name == "resnet_cifar":
        model_kwargs["in_channels"] = in_channels
        if "depth" in kwargs:
            model_kwargs["depth"] = kwargs["depth"]
    elif model_name == "vit":
        model_kwargs["pretrained"] = pretrained
        model_kwargs["image_size"] = image_size
        model_kwargs["in_channels"] = in_channels
        if "model_name" in kwargs:
            model_kwargs["model_name"] = kwargs["model_name"]
    elif model_name == "vit_small":
        model_kwargs["image_size"] = image_size
        model_kwargs["in_channels"] = in_channels
        for key in ["patch_size", "embed_dim", "depth", "num_heads"]:
            if key in kwargs:
                model_kwargs[key] = kwargs[key]

    # add any remaining kwargs
    for key, value in kwargs.items():
        if key not in model_kwargs:
            model_kwargs[key] = value

    return model_class(**model_kwargs)


def get_model_from_config(
    model_cfg: DictConfig,
    dataset_cfg: DictConfig,
) -> nn.Module:
    """Create a model from Hydra config.

    Args:
        model_cfg: model configuration from Hydra
        dataset_cfg: dataset configuration (for num_classes, image_size)

    Returns:
        Initialized model
    """
    return get_model(
        model_name=model_cfg.name,
        num_classes=dataset_cfg.num_classes,
        in_channels=dataset_cfg.channels,
        image_size=dataset_cfg.image_size,
        pretrained=model_cfg.get("pretrained", False),
        **{k: v for k, v in model_cfg.items() if k not in ["name", "_target_", "pretrained"]},
    )

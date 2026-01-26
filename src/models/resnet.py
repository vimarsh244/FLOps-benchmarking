"""ResNet model architectures."""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


def _replace_bn_with_gn(module: nn.Module, num_groups: int = 32) -> nn.Module:
    """Replace all BatchNorm layers with GroupNorm.

    This is needed for FedOpt (FedAdam, FedYogi) compatibility because
    BatchNorm's running statistics don't work well with adaptive server optimizers.

    Args:
        module: PyTorch module to modify
        num_groups: Number of groups for GroupNorm (default 32)

    Returns:
        Modified module with GroupNorm instead of BatchNorm
    """
    for name, child in module.named_children():
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            num_channels = child.num_features
            # ensure num_groups divides num_channels
            groups = min(num_groups, num_channels)
            while num_channels % groups != 0:
                groups -= 1
            setattr(module, name, nn.GroupNorm(groups, num_channels))
        else:
            _replace_bn_with_gn(child, num_groups)
    return module


def _create_resnet(
    model_fn,
    num_classes: int,
    pretrained: bool = False,
    in_channels: int = 3,
) -> nn.Module:
    """Create a ResNet model with custom number of classes.

    Args:
        model_fn: torchvision model constructor
        num_classes: number of output classes
        pretrained: whether to use pretrained weights
        in_channels: number of input channels

    Returns:
        ResNet model
    """
    if pretrained:
        weights = "IMAGENET1K_V1"
    else:
        weights = None

    model = model_fn(weights=weights)

    # modify first conv layer if input channels != 3
    if in_channels != 3:
        old_conv = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        # initialize new conv layer
        if pretrained:
            # average the pretrained weights across channels
            with torch.no_grad():
                model.conv1.weight[:] = old_conv.weight.mean(dim=1, keepdim=True).repeat(
                    1, in_channels, 1, 1
                )

    # modify final fc layer for custom number of classes
    if num_classes != 1000:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    return model


class ResNet18(nn.Module):
    """ResNet-18 wrapper for federated learning."""

    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = False,
        in_channels: int = 3,
        use_groupnorm: bool = False,
    ):
        """
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            in_channels: Number of input channels
            use_groupnorm: If True, replace BatchNorm with GroupNorm (for FedOpt compatibility)
        """
        super().__init__()
        self.model = _create_resnet(
            models.resnet18,
            num_classes=num_classes,
            pretrained=pretrained,
            in_channels=in_channels,
        )
        if use_groupnorm:
            _replace_bn_with_gn(self.model)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ResNet34(nn.Module):
    """ResNet-34 wrapper for federated learning."""

    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = False,
        in_channels: int = 3,
        use_groupnorm: bool = False,
    ):
        super().__init__()
        self.model = _create_resnet(
            models.resnet34,
            num_classes=num_classes,
            pretrained=pretrained,
            in_channels=in_channels,
        )
        if use_groupnorm:
            _replace_bn_with_gn(self.model)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ResNet50(nn.Module):
    """ResNet-50 wrapper for federated learning."""

    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = False,
        in_channels: int = 3,
        use_groupnorm: bool = False,
    ):
        super().__init__()
        self.model = _create_resnet(
            models.resnet50,
            num_classes=num_classes,
            pretrained=pretrained,
            in_channels=in_channels,
        )
        if use_groupnorm:
            _replace_bn_with_gn(self.model)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# small resnet variants for CIFAR (fewer parameters)
class ResNetCIFAR(nn.Module):
    """ResNet variant optimized for CIFAR-sized images (32x32).

    Uses smaller initial conv and no initial maxpool.
    """

    def __init__(
        self,
        num_classes: int = 10,
        depth: int = 18,
        in_channels: int = 3,
    ):
        super().__init__()

        if depth == 18:
            base_model = models.resnet18(weights=None)
        elif depth == 34:
            base_model = models.resnet34(weights=None)
        else:
            raise ValueError(f"Unsupported depth: {depth}")

        # modify for CIFAR: smaller initial conv, no maxpool
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        # skip maxpool for small images

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

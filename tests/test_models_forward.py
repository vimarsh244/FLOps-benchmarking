import torch

from src.models.resnet import ResNet18, ResNetCIFAR
from src.models.simple_cnn import SimpleCNN, SimpleCNNLarge
from src.models.vit import ViTSmall


def test_simple_cnn_forward_shapes():
    model = SimpleCNN(num_classes=10, in_channels=3)
    output = model(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, 10)


def test_simple_cnn_large_forward_shapes():
    model = SimpleCNNLarge(num_classes=5, in_channels=3, image_size=64)
    output = model(torch.randn(4, 3, 64, 64))
    assert output.shape == (4, 5)


def test_resnet18_forward_shapes():
    model = ResNet18(num_classes=7, pretrained=False, in_channels=3)
    output = model(torch.randn(2, 3, 224, 224))
    assert output.shape == (2, 7)


def test_resnet_cifar_forward_shapes():
    model = ResNetCIFAR(num_classes=10, depth=18, in_channels=3)
    output = model(torch.randn(3, 3, 32, 32))
    assert output.shape == (3, 10)


def test_vitsmall_forward_shapes():
    model = ViTSmall(num_classes=11, image_size=32, patch_size=4, in_channels=3)
    output = model(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, 11)

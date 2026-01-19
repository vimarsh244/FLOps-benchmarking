"""Dataset loading utilities for federated learning."""

from typing import Dict, Optional, Tuple, Any, Callable
from functools import lru_cache

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose, 
    Normalize, 
    ToTensor, 
    RandomCrop, 
    RandomHorizontalFlip,
    Resize,
    CenterCrop,
    Lambda,
)
from PIL import Image
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import Partitioner, IidPartitioner, DirichletPartitioner
from omegaconf import DictConfig


# global cache for FederatedDataset instances
_fds_cache: Dict[str, FederatedDataset] = {}


def ensure_rgb(img):
    """Convert image to RGB if it's grayscale or has alpha channel."""
    if isinstance(img, Image.Image):
        if img.mode != "RGB":
            img = img.convert("RGB")
    return img


def get_transforms(
    dataset_cfg: DictConfig,
    is_train: bool = True,
) -> Compose:
    """Create transforms based on dataset configuration.
    
    Args:
        dataset_cfg: dataset configuration from Hydra
        is_train: whether these are training transforms
    
    Returns:
        Composed transforms
    """
    transforms_list = []
    
    # first, ensure image is RGB (handle grayscale images)
    transforms_list.append(Lambda(ensure_rgb))
    
    # resize if needed (for ViT or Tiny-ImageNet)
    aug_cfg = dataset_cfg.augmentation.train if is_train else dataset_cfg.augmentation.test
    
    if "resize" in aug_cfg and aug_cfg.resize:
        transforms_list.append(Resize(aug_cfg.resize))
    
    # data augmentation for training
    if is_train and aug_cfg.get("random_crop", False):
        pad = 4 if dataset_cfg.image_size == 32 else 8
        transforms_list.append(RandomCrop(dataset_cfg.image_size, padding=pad))
    
    if is_train and aug_cfg.get("random_horizontal_flip", False):
        transforms_list.append(RandomHorizontalFlip())
    
    # convert to tensor
    transforms_list.append(ToTensor())
    
    # normalize
    transforms_list.append(
        Normalize(
            mean=list(dataset_cfg.normalize.mean),
            std=list(dataset_cfg.normalize.std),
        )
    )
    
    return Compose(transforms_list)


def get_partitioner(
    partitioner_cfg: DictConfig,
    num_partitions: int,
    label_key: str = "label",
) -> Partitioner:
    """Create a partitioner based on configuration.
    
    Args:
        partitioner_cfg: partitioner configuration from Hydra
        num_partitions: number of partitions to create
        label_key: the label key from dataset config (used for partition_by)
    
    Returns:
        Configured partitioner
    """
    name = partitioner_cfg.name.lower()
    
    if name == "iid":
        return IidPartitioner(num_partitions=num_partitions)
    elif name == "dirichlet":
        # use the dataset's label_key for partition_by
        return DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by=label_key,
            alpha=partitioner_cfg.get("alpha", 0.5),
            seed=partitioner_cfg.get("seed", 42),
        )
    else:
        raise ValueError(f"Unknown partitioner: {name}")


def get_federated_dataset(
    dataset_cfg: DictConfig,
    partitioner_cfg: DictConfig,
    num_partitions: int,
) -> FederatedDataset:
    """Get or create a FederatedDataset instance.
    
    Uses caching to avoid re-downloading data.
    
    Args:
        dataset_cfg: dataset configuration from Hydra
        partitioner_cfg: partitioner configuration from Hydra
        num_partitions: number of partitions
    
    Returns:
        FederatedDataset instance
    """
    cache_key = f"{dataset_cfg.dataset_name}_{partitioner_cfg.name}_{num_partitions}"
    
    if cache_key not in _fds_cache:
        # get label_key from dataset config (with fallback for backwards compatibility)
        label_key = dataset_cfg.get("label_key", "label")
        partitioner = get_partitioner(partitioner_cfg, num_partitions, label_key)
        _fds_cache[cache_key] = FederatedDataset(
            dataset=dataset_cfg.dataset_name,
            partitioners={"train": partitioner},
        )
    
    return _fds_cache[cache_key]


def load_data(
    partition_id: int,
    num_partitions: int,
    dataset_cfg: DictConfig,
    partitioner_cfg: DictConfig,
    batch_size: int = 32,
    test_fraction: float = 0.2,
) -> Tuple[DataLoader, DataLoader]:
    """Load partitioned data for a client.
    
    Args:
        partition_id: ID of the partition to load
        num_partitions: total number of partitions
        dataset_cfg: dataset configuration
        partitioner_cfg: partitioner configuration
        batch_size: batch size for data loaders
        test_fraction: fraction of data to use for testing
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    fds = get_federated_dataset(dataset_cfg, partitioner_cfg, num_partitions)
    
    # load partition
    partition = fds.load_partition(partition_id)
    
    # split into train/test
    partition_train_test = partition.train_test_split(
        test_size=test_fraction, 
        seed=42,
    )
    
    # get transforms
    train_transforms = get_transforms(dataset_cfg, is_train=True)
    test_transforms = get_transforms(dataset_cfg, is_train=False)
    
    # get image and label keys from config (with fallbacks for backwards compatibility)
    image_key = dataset_cfg.get("image_key", "img")
    label_key = dataset_cfg.get("label_key", "label")
    
    def apply_train_transforms(batch):
        batch[image_key] = [train_transforms(img) for img in batch[image_key]]
        return batch
    
    def apply_test_transforms(batch):
        batch[image_key] = [test_transforms(img) for img in batch[image_key]]
        return batch
    
    # apply transforms
    train_data = partition_train_test["train"].with_transform(apply_train_transforms)
    test_data = partition_train_test["test"].with_transform(apply_test_transforms)
    
    # create custom collate function
    def collate_fn(batch):
        images = torch.stack([item[image_key] for item in batch])
        labels = torch.tensor([item[label_key] for item in batch])
        return {"img": images, "label": labels}
    
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # avoid issues with multiprocessing in FL
        drop_last=True, # becasue resnet has batchnorm and if for one client last batch is not divisible by batch size, it will cause issues - so we drop it
    )
    
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    return train_loader, test_loader


def get_centralized_testset(
    dataset_cfg: DictConfig,
    batch_size: int = 32,
) -> DataLoader:
    """Get the centralized test set for server-side evaluation.
    
    Args:
        dataset_cfg: dataset configuration
        batch_size: batch size
    
    Returns:
        DataLoader for test set
    """
    from datasets import load_dataset
    
    # load the test split directly
    dataset = load_dataset(dataset_cfg.dataset_name, split="test")
    
    test_transforms = get_transforms(dataset_cfg, is_train=False)
    
    # get image and label keys from config (with fallbacks for backwards compatibility)
    image_key = dataset_cfg.get("image_key", "img")
    label_key = dataset_cfg.get("label_key", "label")
    
    def apply_transforms(batch):
        batch[image_key] = [test_transforms(img) for img in batch[image_key]]
        return batch
    
    dataset = dataset.with_transform(apply_transforms)
    
    def collate_fn(batch):
        images = torch.stack([item[image_key] for item in batch])
        labels = torch.tensor([item[label_key] for item in batch])
        return {"img": images, "label": labels}
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )


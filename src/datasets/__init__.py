"""Dataset loading and partitioning utilities."""

from src.datasets.loader import load_data, get_federated_dataset
from src.datasets.partitioners import get_partitioner

__all__ = [
    "load_data",
    "get_federated_dataset",
    "get_partitioner",
]


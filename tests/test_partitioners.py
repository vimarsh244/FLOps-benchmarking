import numpy as np
import pytest
from datasets import Dataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from omegaconf import OmegaConf

from src.datasets.partitioners import (
    PathologicalPartitioner,
    QuantitySkewPartitioner,
    get_partitioner,
)


def _partitioner_cfg(name: str, **kwargs):
    cfg = {"name": name}
    cfg.update(kwargs)
    return OmegaConf.create(cfg)


def _label_dataset(num_classes: int, samples_per_class: int) -> Dataset:
    labels = [label for label in range(num_classes) for _ in range(samples_per_class)]
    return Dataset.from_dict({"label": labels, "id": list(range(len(labels)))})


def test_get_partitioner_iid_and_dirichlet():
    iid_cfg = _partitioner_cfg("iid")
    partitioner = get_partitioner(iid_cfg, num_partitions=3)
    assert isinstance(partitioner, IidPartitioner)
    assert partitioner.num_partitions == 3

    dirichlet_cfg = _partitioner_cfg("dirichlet", alpha=0.5, seed=7)
    partitioner = get_partitioner(dirichlet_cfg, num_partitions=4)
    assert isinstance(partitioner, DirichletPartitioner)
    partitioner.dataset = _label_dataset(num_classes=4, samples_per_class=20)
    partitioner._min_partition_size = 1
    assert partitioner.num_partitions == 4


def test_get_partitioner_invalid_name():
    with pytest.raises(ValueError):
        get_partitioner(_partitioner_cfg("unknown"), num_partitions=2)


def test_pathological_partitioner_creates_class_shards():
    # PathologicalPartitioner is instantiated directly (not via get_partitioner).
    dataset = _label_dataset(num_classes=4, samples_per_class=5)
    partitioner = PathologicalPartitioner(num_partitions=2, num_classes_per_partition=2, seed=1)
    partitioner.dataset = dataset

    partitions = [partitioner.load_partition(idx) for idx in range(2)]
    sizes = [len(partition) for partition in partitions]

    assert sum(sizes) == len(dataset)
    for partition in partitions:
        assert len(set(partition["label"])) == 2
    assert all(size == 10 for size in sizes)

    merged_ids = sorted([item for partition in partitions for item in partition["id"]])
    assert merged_ids == list(range(len(dataset)))


def test_quantity_skew_partitioner_distributes_samples():
    dataset = _label_dataset(num_classes=5, samples_per_class=10)
    partitioner = QuantitySkewPartitioner(num_partitions=5, min_samples_per_partition=5, seed=5)
    partitioner.dataset = dataset

    partitions = [partitioner.load_partition(idx) for idx in range(5)]
    sizes = [len(partition) for partition in partitions]

    assert sum(sizes) == len(dataset)
    assert len(set(sizes)) > 1

    merged_ids = sorted([item for partition in partitions for item in partition["id"]])
    assert merged_ids == list(range(len(dataset)))

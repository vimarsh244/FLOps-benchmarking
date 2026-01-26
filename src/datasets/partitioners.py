"""Custom partitioners for federated learning."""

from typing import Dict, List, Optional
import numpy as np
from flwr_datasets.partitioner import Partitioner, IidPartitioner, DirichletPartitioner
from omegaconf import DictConfig


def get_partitioner(
    partitioner_cfg: DictConfig,
    num_partitions: int,
) -> Partitioner:
    """Create a partitioner based on configuration.

    Args:
        partitioner_cfg: partitioner configuration from Hydra
        num_partitions: number of partitions to create

    Returns:
        Configured partitioner
    """
    name = partitioner_cfg.name.lower()

    if name == "iid":
        return IidPartitioner(num_partitions=num_partitions)
    elif name == "dirichlet":
        return DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by=partitioner_cfg.get("partition_by", "label"),
            alpha=partitioner_cfg.get("alpha", 0.5),
            seed=partitioner_cfg.get("seed", 42),
        )
    else:
        raise ValueError(f"Unknown partitioner: {name}. Available: iid, dirichlet")


class PathologicalPartitioner(Partitioner):
    """Pathological non-IID partitioner.

    Each client receives data from only a subset of classes.
    Based on the original FedAvg paper setup.
    """

    def __init__(
        self,
        num_partitions: int,
        num_classes_per_partition: int = 2,
        seed: int = 42,
    ):
        super().__init__()
        self.num_partitions = num_partitions
        self.num_classes_per_partition = num_classes_per_partition
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._partition_id_to_indices: Optional[Dict[int, List[int]]] = None

    def load_partition(self, partition_id: int):
        """Load a single partition."""
        if self._partition_id_to_indices is None:
            self._create_partitions()

        indices = self._partition_id_to_indices[partition_id]
        return self._dataset.select(indices)

    def _create_partitions(self):
        """Create the partition mapping."""
        labels = np.array(self._dataset["label"])
        num_classes = len(np.unique(labels))

        # sort indices by label
        sorted_indices = np.argsort(labels)

        # split into shards (each shard is a contiguous block of one class)
        num_shards = self.num_partitions * self.num_classes_per_partition
        shard_size = len(labels) // num_shards

        shards = [sorted_indices[i * shard_size : (i + 1) * shard_size] for i in range(num_shards)]

        # randomly assign shards to partitions
        shard_ids = list(range(num_shards))
        self._rng.shuffle(shard_ids)

        self._partition_id_to_indices = {}
        for partition_id in range(self.num_partitions):
            partition_shards = shard_ids[
                partition_id
                * self.num_classes_per_partition : (partition_id + 1)
                * self.num_classes_per_partition
            ]
            indices = np.concatenate([shards[s] for s in partition_shards])
            self._partition_id_to_indices[partition_id] = indices.tolist()

    @property
    def num_partitions(self) -> int:
        return self._num_partitions

    @num_partitions.setter
    def num_partitions(self, value: int):
        self._num_partitions = value


class QuantitySkewPartitioner(Partitioner):
    """Quantity skew partitioner.

    Each client receives different amounts of data.
    Follows a power law distribution.
    """

    def __init__(
        self,
        num_partitions: int,
        min_samples_per_partition: int = 10,
        power: float = 1.5,
        seed: int = 42,
    ):
        super().__init__()
        self.num_partitions = num_partitions
        self.min_samples = min_samples_per_partition
        self.power = power
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._partition_id_to_indices: Optional[Dict[int, List[int]]] = None

    def load_partition(self, partition_id: int):
        """Load a single partition."""
        if self._partition_id_to_indices is None:
            self._create_partitions()

        indices = self._partition_id_to_indices[partition_id]
        return self._dataset.select(indices)

    def _create_partitions(self):
        """Create the partition mapping with quantity skew."""
        n_samples = len(self._dataset)

        # generate partition sizes following power law
        sizes = self._rng.power(self.power, self.num_partitions)
        sizes = sizes / sizes.sum()  # normalize

        # ensure minimum samples per partition
        sizes = np.maximum(sizes, self.min_samples / n_samples)
        sizes = sizes / sizes.sum()  # re-normalize

        # convert to actual sizes
        partition_sizes = (sizes * n_samples).astype(int)
        partition_sizes[-1] = n_samples - partition_sizes[:-1].sum()  # fix rounding

        # shuffle all indices
        all_indices = list(range(n_samples))
        self._rng.shuffle(all_indices)

        # assign to partitions
        self._partition_id_to_indices = {}
        start = 0
        for partition_id in range(self.num_partitions):
            end = start + partition_sizes[partition_id]
            self._partition_id_to_indices[partition_id] = all_indices[start:end]
            start = end

    @property
    def num_partitions(self) -> int:
        return self._num_partitions

    @num_partitions.setter
    def num_partitions(self, value: int):
        self._num_partitions = value

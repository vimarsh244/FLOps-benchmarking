import torch
from datasets import Dataset
from omegaconf import OmegaConf
from PIL import Image

from src.datasets import loader


def _dataset_cfg(dataset_name: str = "dummy", image_size: int = 32):
    return OmegaConf.create(
        {
            "dataset_name": dataset_name,
            "image_size": image_size,
            "normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
            "augmentation": {
                "train": {"random_crop": True, "random_horizontal_flip": True},
                "test": {},
            },
            "image_key": "img",
            "label_key": "label",
        }
    )


def _partitioner_cfg(name: str = "iid"):
    return OmegaConf.create({"name": name})


def _dummy_dataset(num_samples: int = 6):
    images = [Image.new("RGB", (32, 32), color=(i, i, i)) for i in range(num_samples)]
    labels = list(range(num_samples))
    return Dataset.from_dict({"img": images, "label": labels})


def test_get_transforms_outputs_tensor_and_normalizes():
    dataset_cfg = _dataset_cfg()
    transforms = loader.get_transforms(dataset_cfg, is_train=False)
    sample = transforms(Image.new("L", (32, 32)))
    assert isinstance(sample, torch.Tensor)
    assert sample.shape == (3, 32, 32)


def test_load_data_uses_federated_dataset(monkeypatch):
    dataset = _dummy_dataset()

    class DummyFederatedDataset:
        def load_partition(self, partition_id):
            assert partition_id == 0
            return dataset

    def fake_get_federated_dataset(*_args, **_kwargs):
        return DummyFederatedDataset()

    monkeypatch.setattr(loader, "get_federated_dataset", fake_get_federated_dataset)

    train_loader, test_loader = loader.load_data(
        partition_id=0,
        num_partitions=1,
        dataset_cfg=_dataset_cfg(),
        partitioner_cfg=_partitioner_cfg(),
        batch_size=2,
        test_fraction=0.5,
    )

    train_batch = next(iter(train_loader))
    test_batch = next(iter(test_loader))

    assert train_batch["img"].shape == (2, 3, 32, 32)
    assert test_batch["label"].shape[0] == 2

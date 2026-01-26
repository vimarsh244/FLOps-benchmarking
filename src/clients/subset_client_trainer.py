"""Subset training utilities for DIWS."""

from __future__ import annotations

import time
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict


class DictStyleDataset(Dataset):
    """Simple dataset wrapper for img/label tensors."""

    def __init__(self, images: List[torch.Tensor], labels: List[torch.Tensor]):
        self.images = images
        self.labels = labels

    def __getitem__(self, idx):
        return {"img": self.images[idx], "label": self.labels[idx]}

    def __len__(self):
        return len(self.labels)


class SubsetClientTrainer:
    """Train a model on a subset of local data."""

    def __init__(
        self,
        model: nn.Module,
        trainloader: DataLoader,
        device: torch.device,
        learning_rate: float,
        epochs: int = 1,
    ):
        self.model = model
        self.trainloader = trainloader
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model.to(self.device)

    def set_parameters(self, params):
        params_dict = zip(self.model.state_dict().keys(), params)
        state_dict = OrderedDict(
            {
                k: torch.tensor(v) if v.shape != torch.Size([]) else torch.tensor([0])
                for k, v in params_dict
            }
        )
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters):
        start_time = time.time()
        self.set_parameters(parameters)

        if len(self.trainloader.dataset) == 0:
            return self.get_parameters(), 0, {"train_loss": 0.0, "subset_empty": True}

        self.model.train()
        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        running_loss = 0.0
        num_batches = 0
        for _ in range(self.epochs):
            for batch in self.trainloader:
                images = batch["img"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                num_batches += 1

        train_loss = running_loss / max(num_batches, 1)
        runtime = time.time() - start_time
        metrics = {"train_loss": train_loss, "runtime": runtime}
        return self.get_parameters(), len(self.trainloader.dataset), metrics


def load_subset_data(
    subset_distribution: Dict[int, int],
    trainloader: DataLoader,
) -> DataLoader:
    """Build a subset dataloader matching the target label distribution."""
    if not subset_distribution:
        empty_dataset = DictStyleDataset([], [])
        return DataLoader(empty_dataset, batch_size=trainloader.batch_size or 1, shuffle=False)

    # collect samples until target distribution is met
    current_counts = {label: 0 for label in subset_distribution}
    collected_inputs: List[torch.Tensor] = []
    collected_labels: List[torch.Tensor] = []

    for batch in trainloader:
        images = batch["img"]
        labels = batch["label"]

        for img, label in zip(images, labels):
            label_int = int(label.item())
            if label_int in subset_distribution and current_counts[label_int] < subset_distribution[label_int]:
                collected_inputs.append(img)
                collected_labels.append(label)
                current_counts[label_int] += 1

            if all(current_counts[l] >= subset_distribution[l] for l in subset_distribution):
                break
        if all(current_counts[l] >= subset_distribution[l] for l in subset_distribution):
            break

    if not collected_inputs:
        empty_dataset = DictStyleDataset([], [])
        return DataLoader(empty_dataset, batch_size=trainloader.batch_size or 1, shuffle=False)

    # build the subset dataloader
    inputs_tensor = torch.stack(collected_inputs)
    labels_tensor = torch.stack(collected_labels)
    target_dataset = DictStyleDataset(inputs_tensor, labels_tensor)
    return DataLoader(
        target_dataset,
        batch_size=trainloader.batch_size or 1,
        shuffle=False,
    )


def get_subset_client_trainer(
    model: nn.Module,
    subset_distribution: Dict[int, int],
    trainloader: DataLoader,
    device: torch.device,
    learning_rate: float,
    epochs: int = 1,
) -> SubsetClientTrainer:
    subset_trainloader = load_subset_data(subset_distribution, trainloader)
    return SubsetClientTrainer(
        model=model,
        trainloader=subset_trainloader,
        device=device,
        learning_rate=learning_rate,
        epochs=epochs,
    )

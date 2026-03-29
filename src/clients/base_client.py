"""Flower client implementation for federated learning."""

import time
import copy
import os
from typing import Dict, List, Tuple, Any, Optional
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar

from omegaconf import DictConfig


_RUNTIME_CONFIGURED = False


class FlowerClient(NumPyClient):
    """Flower client for federated learning.

    Supports:
    - Standard FedAvg/FedProx training
    - Node drop scenarios (returns garbage when disconnected)
    - Straggler simulation with delays
    - FedProx proximal term
    - SCAFFOLD control variates
    """

    def __init__(
        self,
        model: nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        partition_id: int,
        config: DictConfig,
        scenario_handler: Optional[Any] = None,
    ):
        """Initialize the Flower client.

        Args:
            model: PyTorch model
            trainloader: Training data loader
            valloader: Validation data loader
            partition_id: Client partition ID
            config: Hydra configuration
            scenario_handler: Optional scenario handler for special behaviors
        """
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.partition_id = partition_id
        self.config = config
        self.scenario = scenario_handler

        # runtime tuning (cpu threads, cudnn)
        self._configure_runtime()

        # device setup
        self.device = self._get_device()
        self.model.to(self.device)
        print(f"[Client {self.partition_id}] Using device: {self.device}")

        # training config
        self.local_epochs = config.client.get("local_epochs", 1)
        self.learning_rate = config.client.get("learning_rate", 0.01)
        self.batch_size = config.client.get("batch_size", 32)

        # global parameters cache (for FedProx)
        self._global_params: Optional[List[torch.Tensor]] = None

    def _get_device(self) -> torch.device:
        """Get the device to use for training."""
        device_cfg = self.config.training.get("device", "auto")

        if device_cfg == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device_cfg)

    def _resolve_auto_int(self, value: Any, default: int) -> int:
        if value is None:
            return default
        if isinstance(value, str) and value.lower() == "auto":
            return default
        return max(int(value), 1)

    def _configure_runtime(self) -> None:
        global _RUNTIME_CONFIGURED
        if _RUNTIME_CONFIGURED:
            return

        cpu_count = os.cpu_count() or 1
        training_cfg = self.config.get("training", {})

        cpu_threads_cfg = training_cfg.get("cpu_threads", "auto")
        interop_threads_cfg = training_cfg.get("interop_threads", "auto")

        if cpu_threads_cfg is not None:
            cpu_threads = self._resolve_auto_int(cpu_threads_cfg, cpu_count)
            os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
            os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))
            torch.set_num_threads(cpu_threads)

        if interop_threads_cfg is not None:
            interop_threads = self._resolve_auto_int(interop_threads_cfg, cpu_count)
            try:
                torch.set_num_interop_threads(interop_threads)
            except RuntimeError as exc:
                print(f"[Runtime] interop threads unchanged: {exc}")

        cudnn_cfg = training_cfg.get("cudnn_benchmark", "auto")
        if cudnn_cfg == "auto":
            torch.backends.cudnn.benchmark = torch.cuda.is_available()
        elif cudnn_cfg is not None:
            torch.backends.cudnn.benchmark = bool(cudnn_cfg)

        _RUNTIME_CONFIGURED = True

    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model weights from numpy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {
                k: torch.tensor(v) if v.shape != torch.Size([]) else torch.tensor([0])
                for k, v in params_dict
            }
        )
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Get model weights as numpy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def _should_participate(self, current_round: int) -> bool:
        """Check if client should participate in this round."""
        if self.scenario is None:
            return True
        return self.scenario.should_client_participate(self.partition_id, current_round)

    def _apply_straggler_delay(self, current_round: int) -> None:
        """Apply straggler delay if applicable."""
        if self.scenario is None:
            return

        scenario_config = self.scenario.get_client_config(self.partition_id, current_round)

        if scenario_config.get("is_straggler", False):
            # simulate delay by sleeping
            delay_factor = scenario_config.get("delay_multiplier", 1.0)
            # base delay proportional to data size
            base_delay = len(self.trainloader.dataset) * 0.001  # 1ms per sample
            actual_delay = base_delay * (delay_factor - 1)  # additional delay
            if actual_delay > 0:
                time.sleep(min(actual_delay, 10.0))  # cap at 10 seconds

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train the model on local data.

        Args:
            parameters: Model parameters from server
            config: Configuration from server

        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        current_round = config.get("current_round", 0)

        # check if client should participate (node drop scenario)
        if not self._should_participate(current_round):
            print(f"[Client {self.partition_id}] DROPPED for round {current_round}")
            # return garbage to signal disconnection
            return [], 0, {"disconnected": True}

        start_time = time.time()

        # set parameters
        self.set_parameters(parameters)

        # cache global parameters for FedProx
        if "proximal_mu" in config:
            self._global_params = [p.clone().detach() for p in self.model.parameters()]

        # apply straggler delay
        self._apply_straggler_delay(current_round)

        # log cluster assignment if using clustered FL
        if "cluster_id" in config:
            print(f"[Client {self.partition_id}] Assigned cluster {config['cluster_id']}")

        # train
        train_loss = self._train(
            epochs=self.local_epochs,
            proximal_mu=config.get("proximal_mu", 0.0),
        )

        end_time = time.time()
        runtime = end_time - start_time
        print(f"[Client {self.partition_id}] Training took {runtime:.2f}s")

        return (
            self.get_parameters({}),
            len(self.trainloader.dataset),
            {"train_loss": train_loss, "runtime": runtime},
        )

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the model on local test data.

        Note: Evaluation always runs on ALL clients regardless of node drop status.
        This ensures we measure true model performance on the full dataset.
        Node drop only affects training, not evaluation.

        Args:
            parameters: Model parameters from server
            config: Configuration from server

        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        start_time = time.time()

        self.set_parameters(parameters)

        # log cluster if applicable
        if "cluster_id" in config:
            print(f"[Client {self.partition_id}] Eval with cluster {config['cluster_id']}")

        loss, accuracy = self._test()

        end_time = time.time()
        runtime = end_time - start_time

        return loss, len(self.valloader.dataset), {"accuracy": accuracy, "runtime": runtime}

    def _train(self, epochs: int, proximal_mu: float = 0.0) -> float:
        """Train the model for specified epochs.

        Args:
            epochs: Number of local epochs
            proximal_mu: FedProx proximal term coefficient

        Returns:
            Average training loss
        """
        self.model.to(self.device)
        self.model.train()

        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        running_loss = 0.0
        num_batches = 0

        for _ in range(epochs):
            for batch in self.trainloader:
                images = batch["img"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                # add proximal term for FedProx: (mu/2) * ||w - w_global||^2
                if proximal_mu > 0 and self._global_params is not None:
                    proximal_term = 0.0
                    for local_param, global_param in zip(
                        self.model.parameters(), self._global_params
                    ):
                        # use squared L2 norm as per paper
                        proximal_term += (local_param - global_param.to(self.device)).norm(2).pow(2)
                    loss = loss + (proximal_mu / 2) * proximal_term

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                num_batches += 1

        return running_loss / max(num_batches, 1)

    def _test(self) -> Tuple[float, float]:
        """Evaluate the model on validation data.

        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.to(self.device)
        self.model.eval()

        criterion = nn.CrossEntropyLoss()
        correct = 0
        total_loss = 0.0
        num_samples = 0

        with torch.no_grad():
            for batch in self.valloader:
                images = batch["img"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(images)
                total_loss += criterion(outputs, labels).item() * len(labels)

                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                num_samples += len(labels)

        accuracy = correct / max(num_samples, 1)
        avg_loss = total_loss / max(num_samples, 1)

        return avg_loss, accuracy


def create_client_fn(config: DictConfig):
    """Create a client function for Flower simulation.

    Args:
        config: Hydra configuration

    Returns:
        Client function for Flower ClientApp
    """
    from flwr.common import Context
    from src.models.registry import get_model_from_config
    from src.datasets.loader import load_data
    from src.scenarios.registry import get_scenario

    # create scenario handler
    scenario = get_scenario(config.scenario)

    def client_fn(context: Context):
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]

        # load data
        trainloader, valloader = load_data(
            partition_id=partition_id,
            num_partitions=num_partitions,
            dataset_cfg=config.dataset,
            partitioner_cfg=config.partitioner,
            batch_size=config.client.batch_size,
            test_fraction=config.evaluation.test_fraction,
            dataloader_cfg=config.training,
        )

        # create model
        model = get_model_from_config(config.model, config.dataset)

        # create client
        client = FlowerClient(
            model=model,
            trainloader=trainloader,
            valloader=valloader,
            partition_id=partition_id,
            config=config,
            scenario_handler=scenario,
        )

        return client.to_client()

    return client_fn

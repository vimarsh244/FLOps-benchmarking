"""Flower client implementation for federated learning."""

import copy
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from flwr.client import NumPyClient
from flwr.common import Context, NDArrays, Scalar
from omegaconf import DictConfig
from torch.utils.data import DataLoader


class FlowerClient(NumPyClient):
    """Flower client for federated learning.

    Supports:
    - Standard FedAvg/FedProx training
    - Node drop scenarios (returns garbage when disconnected)
    - Straggler simulation with delays
    - FedProx proximal term
    - SCAFFOLD control variates
    - State persistence for personalized strategies
    """

    def __init__(
        self,
        model: nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        partition_id: int,
        config: DictConfig,
        scenario_handler: Optional[Any] = None,
        context: Optional[Context] = None,
    ):
        """Initialize the Flower client.

        Args:
            model: PyTorch model
            trainloader: Training data loader
            valloader: Validation data loader
            partition_id: Client partition ID
            config: Hydra configuration
            scenario_handler: Optional scenario handler for special behaviors
            context: Optional Flower context for state persistence
        """
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.partition_id = partition_id
        self.config = config
        self.scenario = scenario_handler
        self.context = context

        # device setup
        self.device = self._get_device()
        self.model.to(self.device)

        # training config
        self.local_epochs = config.client.get("local_epochs", 1)
        self.local_iters = config.client.get("local_iters", 0)  # 0 = use epochs; >0 = SGD steps
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

    def _get_attack_config(self, current_round: int) -> Dict[str, Any]:
        """Fetch scenario-provided attack metadata for this client/round."""
        if self.scenario is None:
            return {}
        return self.scenario.get_client_config(self.partition_id, current_round)

    def _apply_model_replacement_if_needed(
        self,
        current_round: int,
        initial_parameters: NDArrays,
        updated_parameters: NDArrays,
    ) -> Tuple[NDArrays, bool]:
        """Apply A3 model replacement scaling to this client's update when configured.

        The outgoing model becomes: w_out = w_in + s * (w_local - w_in).
        """
        attack_cfg = self._get_attack_config(current_round)
        is_adversary = bool(attack_cfg.get("is_adversary", False))
        attack_type = str(attack_cfg.get("attack_type", ""))

        if not is_adversary or attack_type != "model_replacement":
            return updated_parameters, False

        scale = float(attack_cfg.get("replacement_scale", 1.0))
        if abs(scale - 1.0) < 1e-12:
            return updated_parameters, True

        attacked: NDArrays = []
        for w_in, w_out in zip(initial_parameters, updated_parameters):
            w_in_arr = np.asarray(w_in, dtype=np.float32)
            w_out_arr = np.asarray(w_out, dtype=np.float32)
            attacked.append(w_in_arr + np.float32(scale) * (w_out_arr - w_in_arr))
        return attacked, True

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

        # train (use step-based training when local_iters > 0)
        proximal_mu = config.get("proximal_mu", 0.0)
        if self.local_iters > 0:
            train_loss = self._train_steps(
                steps=self.local_iters,
                proximal_mu=proximal_mu,
            )
        else:
            train_loss = self._train(
                epochs=self.local_epochs,
                proximal_mu=proximal_mu,
            )

        end_time = time.time()
        runtime = end_time - start_time
        print(f"[Client {self.partition_id}] Training took {runtime:.2f}s")

        local_params = self.get_parameters({})
        outgoing_params, is_adversary = self._apply_model_replacement_if_needed(
            current_round=current_round,
            initial_parameters=parameters,
            updated_parameters=local_params,
        )

        return (
            outgoing_params,
            len(self.trainloader.dataset),
            {
                "train_loss": train_loss,
                "runtime": runtime,
                "is_adversary": float(is_adversary),
            },
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

        attack_cfg = self._get_attack_config(config.get("current_round", 0))
        is_adversary = bool(attack_cfg.get("is_adversary", False))

        return (
            loss,
            len(self.valloader.dataset),
            {
                "accuracy": accuracy,
                "runtime": runtime,
                "is_adversary": float(is_adversary),
            },
        )

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

        # optimizer based on config
        optimizer_name = self.config.client.get("optimizer", "sgd").lower()
        weight_decay = self.config.client.get("weight_decay", 0.0)

        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=weight_decay,
            )
        elif optimizer_name == "sgd":
            momentum = self.config.client.get("momentum", 0.0)
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}. Use 'sgd' or 'adam'.")

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

    def _train_steps(self, steps: int, proximal_mu: float = 0.0) -> float:
        """Train the model for a fixed number of SGD steps.

        Unlike _train which iterates over full epochs, this method runs
        exactly steps mini-batch gradient updates, cycling over the
        dataloader as needed.
        
        Args:
            steps: Number of SGD steps to perform
            proximal_mu: Proximal term coefficient

        Returns:
            Average training loss
        """
        self.model.to(self.device)
        self.model.train()

        criterion = nn.CrossEntropyLoss().to(self.device)

        # optimizer
        optimizer_name = self.config.client.get("optimizer", "sgd").lower()
        weight_decay = self.config.client.get("weight_decay", 0.0)

        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=weight_decay,
            )
        elif optimizer_name == "sgd":
            momentum = self.config.client.get("momentum", 0.0)
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}. Use 'sgd' or 'adam'.")

        running_loss = 0.0
        step_count = 0
        data_iter = iter(self.trainloader)

        while step_count < steps:
            # cycle over dataloader
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.trainloader)
                batch = next(data_iter)

            images = batch["img"].to(self.device)
            labels = batch["label"].to(self.device)

            optimizer.zero_grad()

            outputs = self.model(images)
            loss = criterion(outputs, labels)

            # proximal term: (mu/2) * ||w - w_global||^2
            if proximal_mu > 0 and self._global_params is not None:
                proximal_term = 0.0
                for local_param, global_param in zip(
                    self.model.parameters(), self._global_params
                ):
                    proximal_term += (local_param - global_param.to(self.device)).norm(2).pow(2)
                loss = loss + (proximal_mu / 2) * proximal_term

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            step_count += 1

        return running_loss / max(step_count, 1)

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

    from src.datasets.loader import load_data
    from src.models.registry import get_model_from_config
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
            context=context,
        )

        return client.to_client()

    return client_fn

"""NLP-specific Flower client for language modeling tasks."""

import time
from typing import Dict, Tuple, Any, Optional
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar

from omegaconf import DictConfig


class NLPFlowerClient(NumPyClient):
    """Flower client for NLP/language modeling tasks.
    
    Key differences from base FlowerClient:
    - Handles input_ids/target_ids instead of img/label
    - Uses CrossEntropyLoss for language modeling
    - Computes perplexity as evaluation metric
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
        """Initialize the NLP client.
        
        Args:
            model: PyTorch model (LSTM or Transformer)
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
        
        # Device setup
        self.device = self._get_device()
        self.model.to(self.device)
        
        # Training config
        self.local_epochs = config.client.get("local_epochs", 1)
        self.learning_rate = config.client.get("learning_rate", 0.001)
        self.batch_size = config.client.get("batch_size", 32)
        
        # Global parameters cache (for FedProx)
        self._global_params: Optional[list] = None

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
        return self.scenario.should_client_participate(
            self.partition_id, current_round
        )

    def _apply_straggler_delay(self, current_round: int) -> None:
        """Apply straggler delay if applicable."""
        if self.scenario is None:
            return
        
        scenario_config = self.scenario.get_client_config(
            self.partition_id, current_round
        )
        
        if scenario_config.get("is_straggler", False):
            delay_factor = scenario_config.get("delay_multiplier", 1.0)
            base_delay = len(self.trainloader.dataset) * 0.001
            actual_delay = base_delay * (delay_factor - 1)
            if actual_delay > 0:
                time.sleep(min(actual_delay, 10.0))

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
        
        # Check if client should participate
        if not self._should_participate(current_round):
            print(f"[NLP Client {self.partition_id}] DROPPED for round {current_round}")
            return [], 0, {"disconnected": True}
        
        start_time = time.time()
        
        # Set parameters
        self.set_parameters(parameters)
        
        # Cache global parameters for FedProx
        if "proximal_mu" in config:
            self._global_params = [
                p.clone().detach() for p in self.model.parameters()
            ]
        
        # Apply straggler delay
        self._apply_straggler_delay(current_round)
        
        # Train
        train_loss, perplexity = self._train(
            epochs=self.local_epochs,
            proximal_mu=config.get("proximal_mu", 0.0),
        )
        
        end_time = time.time()
        runtime = end_time - start_time
        print(f"[NLP Client {self.partition_id}] Training took {runtime:.2f}s, loss: {train_loss:.4f}, ppl: {perplexity:.2f}")
        
        return (
            self.get_parameters({}),
            len(self.trainloader.dataset),
            {"train_loss": train_loss, "perplexity": perplexity, "runtime": runtime},
        )

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the model on local test data.
        
        Args:
            parameters: Model parameters from server
            config: Configuration from server
        
        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        start_time = time.time()
        
        self.set_parameters(parameters)
        
        loss, perplexity, accuracy = self._test()
        
        end_time = time.time()
        runtime = end_time - start_time
        
        return loss, len(self.valloader.dataset), {"perplexity": perplexity, "accuracy": accuracy, "runtime": runtime}

    def _train(self, epochs: int, proximal_mu: float = 0.0) -> Tuple[float, float]:
        """Train the model for specified epochs.
        
        Args:
            epochs: Number of local epochs
            proximal_mu: FedProx proximal term coefficient
        
        Returns:
            Tuple of (average_loss, perplexity)
        """
        self.model.to(self.device)
        self.model.train()
        
        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
        )
        
        running_loss = 0.0
        num_batches = 0
        
        for _ in range(epochs):
            for batch in self.trainloader:
                input_ids = batch["input_ids"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(input_ids)
                
                # Reshape for loss: (batch * seq, vocab) vs (batch * seq)
                batch_size, seq_len, vocab_size = outputs.shape
                outputs_flat = outputs.view(-1, vocab_size)
                targets_flat = target_ids.view(-1)
                
                loss = criterion(outputs_flat, targets_flat)
                
                # Add proximal term for FedProx
                if proximal_mu > 0 and self._global_params is not None:
                    proximal_term = 0.0
                    for local_param, global_param in zip(
                        self.model.parameters(), self._global_params
                    ):
                        proximal_term += (local_param - global_param.to(self.device)).norm(2).pow(2)
                    loss = loss + (proximal_mu / 2) * proximal_term
                
                loss.backward()
                
                # Gradient clipping (important for LM)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                
                running_loss += loss.item()
                num_batches += 1
        
        avg_loss = running_loss / max(num_batches, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return avg_loss, perplexity

    def _test(self) -> Tuple[float, float, float]:
        """Evaluate the model on validation data.
        
        Returns:
            Tuple of (loss, perplexity, accuracy)
        """
        self.model.to(self.device)
        self.model.eval()
        
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        num_tokens = 0
        
        with torch.no_grad():
            for batch in self.valloader:
                input_ids = batch["input_ids"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)
                
                outputs = self.model(input_ids)
                
                batch_size, seq_len, vocab_size = outputs.shape
                outputs_flat = outputs.view(-1, vocab_size)
                targets_flat = target_ids.view(-1)
                
                loss = criterion(outputs_flat, targets_flat)
                total_loss += loss.item() * (batch_size * seq_len)
                
                # Compute accuracy
                predictions = outputs_flat.argmax(dim=1)
                correct += (predictions == targets_flat).sum().item()
                num_tokens += batch_size * seq_len
        
        avg_loss = total_loss / max(num_tokens, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        accuracy = correct / max(num_tokens, 1)
        
        return avg_loss, perplexity, accuracy

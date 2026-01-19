"""SCAFFOLD client implementation with control variate training.

Paper: SCAFFOLD: Stochastic Controlled Averaging for Federated Learning
       Karimireddy et al., 2020
       https://arxiv.org/abs/1910.06378

Important: Control variates are maintained ONLY for trainable parameters,
not for buffers (like BatchNorm running_mean/running_var). This ensures
proper alignment between server and client control variate application.
"""

import time
from typing import Dict, List, Tuple, Any, Optional
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar

from omegaconf import DictConfig


class ScaffoldClient(NumPyClient):
    """SCAFFOLD client with control variate correction during training.
    
    The SCAFFOLD algorithm corrects for client drift by maintaining control
    variates that adjust the gradient updates. This client implements:
    
    1. Unpacks [full_model, server_control, client_control] from server
       where control variates are for trainable params only
    2. Trains with correction: grad - c_i + c (variance reduction)
    3. Computes new control variate after training
    4. Returns [full_model, new_c_i, delta_c] to server
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
        """Initialize the SCAFFOLD client.
        
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
        
        # device setup
        self.device = self._get_device()
        self.model.to(self.device)
        
        # training config
        self.local_epochs = config.client.get("local_epochs", 1)
        self.learning_rate = config.client.get("learning_rate", 0.01)
        self.batch_size = config.client.get("batch_size", 32)
        
        # cache counts
        self._num_trainable = len(list(self.model.parameters()))
        self._num_full_state = len(list(self.model.state_dict().keys()))

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
        """Set model weights from numpy arrays (full state_dict)."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {
                k: torch.tensor(v) if v.shape != torch.Size([]) else torch.tensor([0])
                for k, v in params_dict
            }
        )
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Get model weights as numpy arrays (full state_dict)."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def get_trainable_parameters(self) -> NDArrays:
        """Get only trainable parameters as numpy arrays."""
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

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

    def _unpack_scaffold_params(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, NDArrays, NDArrays]:
        """Unpack SCAFFOLD parameters from server.
        
        Server sends: 
        - First round: [full_model...] only (num_trainable=0)
        - Later rounds: [full_model..., server_control..., client_control...]
        where control variates are for trainable params only.
        
        Returns:
            Tuple of (full_model_params, server_control, client_control)
        """
        # get counts from config or compute from model
        num_full_state = int(config.get("num_full_state", self._num_full_state))
        num_trainable = int(config.get("num_trainable", self._num_trainable))
        is_first_round = int(config.get("first_round", 0)) == 1
        
        full_model = parameters[:num_full_state]
        
        if is_first_round or num_trainable == 0:
            # first round: no control variates sent, initialize to zeros
            # use actual number of trainable params from model
            server_control = [np.zeros_like(p.detach().cpu().numpy(), dtype=np.float32) 
                            for p in self.model.parameters()]
            client_control = [np.zeros_like(p.detach().cpu().numpy(), dtype=np.float32) 
                            for p in self.model.parameters()]
        else:
            # later rounds: unpack control variates from parameters
            server_control = parameters[num_full_state:num_full_state + num_trainable]
            client_control = parameters[num_full_state + num_trainable:num_full_state + 2*num_trainable]
        
        return full_model, server_control, client_control

    def _pack_scaffold_results(
        self,
        full_model: NDArrays,
        new_client_control: NDArrays,
        delta_control: NDArrays,
    ) -> NDArrays:
        """Pack SCAFFOLD results to send to server.
        
        Returns: [full_model..., new_client_control..., delta_control...]
        """
        return list(full_model) + list(new_client_control) + list(delta_control)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train with SCAFFOLD control variate correction.
        
        Args:
            parameters: Packed [full_model, server_control, client_control] from server
            config: Configuration including client_lr and current_round
        
        Returns:
            Tuple of (packed_results, num_examples, metrics)
        """
        current_round = config.get("current_round", 0)
        
        # check if client should participate (node drop scenario)
        if not self._should_participate(current_round):
            print(f"[ScaffoldClient {self.partition_id}] DROPPED for round {current_round}")
            return [], 0, {"disconnected": True}
        
        start_time = time.time()
        
        # unpack scaffold parameters
        full_model, server_control, client_control = self._unpack_scaffold_params(parameters, config)
        
        # set full model parameters
        self.set_parameters(full_model)
        
        # convert control variates to torch tensors
        # these are for TRAINABLE params only, matching model.parameters() order
        server_control_tensors = [
            torch.tensor(c, dtype=torch.float32, device=self.device) 
            for c in server_control
        ]
        client_control_tensors = [
            torch.tensor(c, dtype=torch.float32, device=self.device) 
            for c in client_control
        ]
        
        # cache initial trainable params for control variate computation
        initial_trainable = [p.clone().detach() for p in self.model.parameters()]
        
        # apply straggler delay
        self._apply_straggler_delay(current_round)
        
        # get learning rate from config (server may override)
        client_lr = config.get("client_lr", self.learning_rate)
        
        # train with scaffold correction
        train_loss, total_steps = self._train_scaffold(
            epochs=self.local_epochs,
            client_lr=client_lr,
            server_control=server_control_tensors,
            client_control=client_control_tensors,
        )
        
        # compute new client control variate using option 2 from paper:
        # c_i^+ = c_i - c + (1/(K*eta)) * (x - y)
        # where x is initial trainable params, y is trained trainable params
        new_model_full = self.get_parameters({})
        
        # compute new control variates for trainable params only
        new_client_control = []
        delta_control = []
        
        for i, (init_p, new_p, c_i, c) in enumerate(zip(
            initial_trainable, 
            self.model.parameters(),
            client_control_tensors,
            server_control_tensors
        )):
            # (x - y) / (K * eta)
            update_term = (init_p - new_p.detach()) / (total_steps * client_lr)
            
            # c_i^+ = c_i - c + update_term
            new_c_i = c_i - c + update_term
            new_client_control.append(new_c_i.cpu().numpy())
            
            # delta_c = c_i^+ - c_i
            delta_c = new_c_i - c_i
            delta_control.append(delta_c.cpu().numpy())
        
        end_time = time.time()
        runtime = end_time - start_time
        print(f"[ScaffoldClient {self.partition_id}] Training took {runtime:.2f}s")
        
        # pack results: [full_model, new_c_i (trainable), delta_c (trainable)]
        packed_results = self._pack_scaffold_results(
            new_model_full, new_client_control, delta_control
        )
        
        return (
            packed_results,
            len(self.trainloader.dataset),
            {
                "train_loss": train_loss, 
                "runtime": runtime,
                "num_trainable": self._num_trainable,
                "num_full_state": self._num_full_state,
            },
        )

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the model on local test data.
        
        Note: For evaluation, we only use the model parameters (not control variates).
        Evaluation runs on ALL clients regardless of node drop status.
        """
        start_time = time.time()
        
        # for evaluation, parameters are just full model params (not packed)
        self.set_parameters(parameters)
        
        loss, accuracy = self._test()
        
        end_time = time.time()
        runtime = end_time - start_time
        
        return loss, len(self.valloader.dataset), {"accuracy": accuracy, "runtime": runtime}

    def _train_scaffold(
        self,
        epochs: int,
        client_lr: float,
        server_control: List[torch.Tensor],
        client_control: List[torch.Tensor],
    ) -> Tuple[float, int]:
        """Train with SCAFFOLD control variate correction.
        
        The gradient correction is: grad - c_i + c
        where c_i is client control, c is server control.
        
        Control variates are applied to TRAINABLE parameters only,
        in the same order as model.parameters().
        
        Args:
            epochs: Number of local epochs
            client_lr: Learning rate for local training
            server_control: Server control variate tensors (trainable params only)
            client_control: Client control variate tensors (trainable params only)
        
        Returns:
            Tuple of (average_loss, total_steps)
        """
        self.model.to(self.device)
        self.model.train()
        
        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=client_lr,
        )
        
        running_loss = 0.0
        total_steps = 0
        
        for _ in range(epochs):
            for batch in self.trainloader:
                images = batch["img"].to(self.device)
                labels = batch["label"].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # apply scaffold correction to gradients: grad - c_i + c
                # iterate over trainable params and their corresponding controls
                with torch.no_grad():
                    for param, c_i, c in zip(
                        self.model.parameters(),
                        client_control,
                        server_control
                    ):
                        if param.grad is not None:
                            # correction: subtract client control, add server control
                            param.grad.add_(c - c_i)
                
                optimizer.step()
                
                running_loss += loss.item()
                total_steps += 1
        
        return running_loss / max(total_steps, 1), total_steps

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

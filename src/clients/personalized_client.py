"""Personalized client implementation."""

import time
from typing import Dict, List, Optional, Tuple

import torch
from flwr.common import ArrayRecord, NDArrays, Scalar

from src.clients.base_client import FlowerClient


class PersonalizedClient(FlowerClient):
    """Client supporting personalization.

    This client properly persists personal parameters across rounds,
    which is needed for FedPer to work correctly as personal layers
    must be kept local across rounds.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._param_names: List[str] = list(self.model.state_dict().keys())
        self._num_params: int = len(self._param_names)

        # local
        self._personal_params: Optional[List[torch.Tensor]] = None

        # load any persisted state from previous rounds
        self._load_personal_state()

    def _save_personal_state(self, key: str, params: List[torch.Tensor]) -> None:
        """Save personal parameters to context.state for persistence across rounds.

        Args:
            key: The key to use for storage
            params: List of parameter tensors to save
        """
        if self.context is None:
            # no context available
            return

        if not hasattr(self.context, "state") or self.context.state is None:
            return

        numpy_params = [p.cpu().numpy() for p in params]

        # store
        self.context.state[key] = ArrayRecord(numpy_ndarrays=numpy_params)

    def _load_personal_state(self) -> None:
        """Load personal parameters from context.state if available."""
        if self.context is None:
            return

        if not hasattr(self.context, "state") or self.context.state is None:
            return

        # load personal params
        if "fedper_key" in self.context.state:
            try:
                record = self.context.state["fedper_key"]
                numpy_params = record.to_numpy_ndarrays()
                self._personal_params = [torch.tensor(p) for p in numpy_params]
            except Exception as e:
                print(f"[Client {self.partition_id}] Warning: Failed to load FedPer state: {e}")

    def _set_parameters_partial(self, shared_params: NDArrays, num_personal: int) -> None:
        """Set model parameters using shared params and persisted personal params."""
        num_personal = max(int(num_personal), 0)
        num_shared = self._num_params - num_personal

        if len(shared_params) == self._num_params:
            # full parameters provided
            self.set_parameters(shared_params)
            return

        if len(shared_params) != num_shared:
            raise ValueError(f"Expected {num_shared} shared params, got {len(shared_params)}")

        state_dict = self.model.state_dict()
        # set shared parameters from server
        for i in range(len(shared_params)):
            name = self._param_names[i]
            arr = shared_params[i]
            state_dict[name] = torch.tensor(arr)

        if self._personal_params is None:
            # first round: initialize personal params from current model state
            self._personal_params = [
                state_dict[name].detach().clone() for name in self._param_names[num_shared:]
            ]
            print(f"[Client {self.partition_id}] Initialized personal params (first round)")

        # set personal parameters
        for i in range(len(self._personal_params)):
            name = self._param_names[num_shared + i]
            tensor = self._personal_params[i]
            state_dict[name] = tensor.detach().clone().to(state_dict[name].device)

        self.model.load_state_dict(state_dict, strict=True)

    def _get_shared_params(self, num_personal: int) -> NDArrays:
        """Get shared parameters (as numpy arrays)."""
        num_personal = max(int(num_personal), 0)
        num_shared = self._num_params - num_personal
        state = self.model.state_dict()
        return [state[name].cpu().numpy() for name in self._param_names[:num_shared]]

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train model with FedPer logic."""
        strategy = str(config.get("strategy", "")).lower()
        current_round = config.get("current_round", 0)

        if not self._should_participate(current_round):
            print(f"[Client {self.partition_id}] DROPPED for round {current_round}")
            return [], 0, {"disconnected": True}

        start_time = time.time()

        if strategy == "fedper":
            num_personal = int(config.get("personal_layer_count", 0))

            # set shared params from server + personal params from persistent state
            self._set_parameters_partial(parameters, num_personal)

            self._apply_straggler_delay(current_round)

            train_loss = self._train(epochs=self.local_epochs, proximal_mu=0.0)

            # update personal params state from trained model
            state = self.model.state_dict()
            if num_personal > 0:
                self._personal_params = [
                    state[name].detach().clone()
                    for name in self._param_names[self._num_params - num_personal :]
                ]
                # persist personal params to context.state for next round
                self._save_personal_state("fedper_key", self._personal_params)
            else:
                self._personal_params = []

            end_time = time.time()
            runtime = end_time - start_time

            return (
                self._get_shared_params(num_personal),
                len(self.trainloader.dataset),
                {"train_loss": train_loss, "runtime": runtime},
            )

        return super().fit(parameters, config)

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate model with FedPer personalization."""
        strategy = str(config.get("strategy", "")).lower()

        if strategy == "fedper":
            start_time = time.time()
            num_personal = int(config.get("personal_layer_count", 0))
            # use persisted personal params from context.state
            self._set_parameters_partial(parameters, num_personal)
            loss, accuracy = self._test()
            end_time = time.time()
            eval_runtime = end_time - start_time
            return (
                loss,
                len(self.valloader.dataset),
                {"accuracy": accuracy, "eval_runtime": eval_runtime},
            )

        return super().evaluate(parameters, config)

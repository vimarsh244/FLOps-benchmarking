"""Personalized client implementations for FedPer and Ditto."""

import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from flwr.common import NDArrays, Scalar, ArrayRecord

from src.clients.base_client import FlowerClient


class PersonalizedClient(FlowerClient):
    """Client supporting FedPer and Ditto personalization.

    This client properly persists personal parameters across rounds using
    Flower's context.state mechanism, which is needed for FedPer and Ditto
    to work correctly (personal layers must be kept local across rounds).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._param_names: List[str] = list(self.model.state_dict().keys())
        self._num_params: int = len(self._param_names)
        # local cache
        self._personal_params: Optional[List[torch.Tensor]] = None
        self._personal_model_params: Optional[List[torch.Tensor]] = None

        # load any persisted state from previous rounds
        self._load_personal_state()

    def _save_personal_state(self, key: str, params: List[torch.Tensor]) -> None:
        """Save personal parameters to context.state for persistence across rounds.

        Args:
            key: The key to use for storage ("fedper_key" or "ditto_key")
            params: List of parameter tensors to save
        """
        if self.context is None:
            # no context available, can't persist
            return

        # convert tensors to numpy arrays
        numpy_params = [p.cpu().numpy() for p in params]

        # access state and store
        if not hasattr(self.context, "state") or self.context.state is None:
            return

        # store
        self.context.state[key] = ArrayRecord(numpy_ndarrays=numpy_params)

    def _is_full_state_compatible(self, params: Optional[List[torch.Tensor]]) -> bool:
        """Check whether cached params match the model full state_dict layout."""
        if params is None or len(params) != self._num_params:
            return False

        state_dict = self.model.state_dict()
        for name, tensor in zip(self._param_names, params):
            if tuple(tensor.shape) != tuple(state_dict[name].shape):
                return False
        return True

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

        if "ditto_key" in self.context.state:
            try:
                record = self.context.state["ditto_key"]
                numpy_params = record.to_numpy_ndarrays()
                candidate = [torch.tensor(p) for p in numpy_params]
                if self._is_full_state_compatible(candidate):
                    self._personal_model_params = candidate
                else:
                    self._personal_model_params = None
                    print(
                        f"[Client {self.partition_id}] Warning: Ignoring incompatible Ditto state"
                    )
            except Exception as e:
                print(f"[Client {self.partition_id}] Warning: Failed to load Ditto state: {e}")

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
        for name, arr in zip(self._param_names[:num_shared], shared_params):
            state_dict[name] = torch.tensor(arr)

        # use persisted personal params if available, otherwise initialize from model
        if self._personal_params is None:
            # first round: initialize personal params from current model state
            # (which is random for new models, or could be from initial_parameters)
            self._personal_params = [
                state_dict[name].detach().clone() for name in self._param_names[num_shared:]
            ]
            print(f"[Client {self.partition_id}] Initialized personal params (first round)")

        # set personal parameters from persisted cache
        for name, tensor in zip(self._param_names[num_shared:], self._personal_params):
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
        """Train model with FedPer or Ditto logic."""
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

            # update personal params cache from trained model
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

            local_shared = self._get_shared_params(num_personal)
            outgoing_shared, is_adversary = self._apply_model_replacement_if_needed(
                current_round=current_round,
                initial_parameters=parameters,
                updated_parameters=local_shared,
            )

            return (
                outgoing_shared,
                len(self.trainloader.dataset),
                {
                    "train_loss": train_loss,
                    "runtime": runtime,
                    "is_adversary": float(is_adversary),
                },
            )

        if strategy == "ditto":
            # global model update (standard FedAvg local SGD)
            self.set_parameters(parameters)
            # anchor w^t: the global model at the start of this round (before local updates)
            global_anchor = [p.detach().clone() for p in self.model.parameters()]
            self._apply_straggler_delay(current_round)

            # use step-based training when local_iters > 0
            if self.local_iters > 0:
                global_train_loss = self._train_steps(steps=self.local_iters, proximal_mu=0.0)
            else:
                global_train_loss = self._train(epochs=self.local_epochs, proximal_mu=0.0)
            global_params = self.get_parameters({})

            # personalized model update
            # v_k = v_k - eta * (grad_F_k(v_k) + lambda_ * (v_k - w_t))
            ditto_mu = float(config.get("ditto_mu", 0.1))
            personal_epochs = int(config.get("personal_epochs", 0))
            if personal_epochs <= 0:
                personal_epochs = self.local_epochs

            # initialize personalized model from w^t (server model) on first round
            if self._personal_model_params is None:
                state = self.model.state_dict()
                self._personal_model_params = [
                    state[name].detach().cpu().clone() for name in self._param_names
                ]

            if not self._is_full_state_compatible(self._personal_model_params):
                state = self.model.state_dict()
                self._personal_model_params = [
                    state[name].detach().cpu().clone() for name in self._param_names
                ]

            # load personalized model and set w^t as the proximal anchor
            self.set_parameters([p.cpu().numpy() for p in self._personal_model_params])
            self._global_params = global_anchor

            # use step-based training when local_iters > 0
            if self.local_iters > 0:
                personal_train_loss = self._train_steps(
                    steps=self.local_iters, proximal_mu=ditto_mu
                )
            else:
                personal_train_loss = self._train(
                    epochs=personal_epochs, proximal_mu=ditto_mu
                )

            # store personalized params
            state = self.model.state_dict()
            self._personal_model_params = [
                state[name].detach().cpu().clone() for name in self._param_names
            ]
            # persist personal params to context.state for next round
            self._save_personal_state("ditto_key", self._personal_model_params)

            end_time = time.time()
            runtime = end_time - start_time

            outgoing_global, is_adversary = self._apply_model_replacement_if_needed(
                current_round=current_round,
                initial_parameters=parameters,
                updated_parameters=global_params,
            )

            return (
                outgoing_global,
                len(self.trainloader.dataset),
                {
                    "train_loss": global_train_loss,
                    "personal_loss": personal_train_loss,
                    "runtime": runtime,
                    "is_adversary": float(is_adversary),
                },
            )

        # fallback to base behavior
        return super().fit(parameters, config)

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate model with FedPer or Ditto personalization."""
        strategy = str(config.get("strategy", "")).lower()

        if strategy == "fedper":
            start_time = time.time()
            num_personal = int(config.get("personal_layer_count", 0))
            # this will use persisted personal params from context.state
            self._set_parameters_partial(parameters, num_personal)
            loss, accuracy = self._test()
            end_time = time.time()
            eval_runtime = end_time - start_time
            attack_cfg = self._get_attack_config(config.get("current_round", 0))
            return (
                loss,
                len(self.valloader.dataset),
                {
                    "accuracy": accuracy,
                    "eval_runtime": eval_runtime,
                    "is_adversary": float(bool(attack_cfg.get("is_adversary", False))),
                },
            )

        if strategy == "ditto":
            start_time = time.time()
            use_personal = bool(config.get("evaluate_personalized", True))
            if use_personal and self._is_full_state_compatible(self._personal_model_params):
                self.set_parameters([p.cpu().numpy() for p in self._personal_model_params])
            else:
                self.set_parameters(parameters)
            loss, accuracy = self._test()
            end_time = time.time()
            eval_runtime = end_time - start_time
            attack_cfg = self._get_attack_config(config.get("current_round", 0))
            return (
                loss,
                len(self.valloader.dataset),
                {
                    "accuracy": accuracy,
                    "eval_runtime": eval_runtime,
                    "is_adversary": float(bool(attack_cfg.get("is_adversary", False))),
                },
            )

        return super().evaluate(parameters, config)

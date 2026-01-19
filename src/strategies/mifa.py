"""MIFA (Fast Federated Learning under Device Unavailability) strategy.

Paper: Fast Federated Learning in the Presence of Arbitrary Device Unavailability
       https://arxiv.org/abs/2106.04159

MIFA maintains a cached per-client update table and uses it to handle
client unavailability gracefully.
"""

from logging import WARNING
from typing import Callable, Optional, Union, List, Tuple, Dict, Set

import numpy as np

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import weighted_loss_avg
from flwr.server.strategy.strategy import Strategy

from src.strategies.base import (
    weighted_average,
    zeros_like,
    add_inplace,
    copy_weights,
)
from src.utils.wandb_logger import log_round_metrics


WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server.
"""


class CustomMIFA(Strategy):
    """MIFA strategy for handling device unavailability.

    Paper: Fast Federated Learning in the Presence of Arbitrary Device Unavailability
           https://arxiv.org/abs/2106.04159

    Maintains a cached per-client update table U_i. On each round t, for each
    successful client i we set:

        U_i <- (w_i^t - w^t) / (K * eta_local)

    where K is the number of local steps and eta_local is the client learning rate.
    This normalizes updates to be comparable across clients.

    Then the global is updated using the average cached update across ALL clients:

        w^{t+1} = w^t + eta_server * (1/N) * sum_i U_i

    Parameters
    ----------
    base_server_lr : float
        Base server learning rate for aggregation.
    client_lr : float
        Client-side learning rate (used for update normalization).
    local_steps : int
        Number of local SGD steps per client (used for normalization).
    wait_for_all_clients_init : bool
        If True, wait for all clients to participate before starting
        the MIFA update rule.
    """

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        base_server_lr: float = 1.0,
        client_lr: float = 0.01,
        local_steps: int = 1,
        local_epochs: int = 1,
        batch_size: int = 32,
        wait_for_all_clients_init: bool = True,
    ) -> None:
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn or weighted_average
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn or weighted_average

        # mifa state
        self._server_lr = float(base_server_lr)
        self._client_lr = float(client_lr)
        self._local_steps = int(local_steps)
        self._local_epochs = int(local_epochs)
        self._batch_size = int(batch_size)
        self._true_round = 0  # counts only updates after table init
        self._table_initialized = False
        self._wait_for_all = wait_for_all_clients_init
        self._all_client_ids: Set[str] = set()
        self._update_table: Dict[str, NDArrays] = {}
        self._latest_global: Optional[NDArrays] = None

    def __repr__(self) -> str:
        return f"CustomMIFA(server_lr={self._server_lr}, client_lr={self._client_lr})"

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None
        if initial_parameters is not None:
            self._latest_global = parameters_to_ndarrays(initial_parameters)
        return initial_parameters

    def _normalization_factor(self, num_samples: int = 0) -> float:
        """Get the normalization factor for client updates: K * eta_local.
        
        Args:
            num_samples: Number of samples the client trained on.
                        Used to estimate actual gradient steps.
        
        The number of gradient steps K = local_epochs * ceil(num_samples / batch_size).
        If local_steps is explicitly set > 1, use that instead.
        """
        if self._local_steps > 1:
            # use configured local_steps if explicitly set
            actual_steps = self._local_steps
        elif num_samples > 0 and self._batch_size > 0:
            # estimate from training config
            batches_per_epoch = max(1, (num_samples + self._batch_size - 1) // self._batch_size)
            actual_steps = self._local_epochs * batches_per_epoch
        else:
            # fallback to 1 epoch worth of steps
            actual_steps = max(1, self._local_steps)
        
        return float(actual_steps) * self._client_lr

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        if self._latest_global is None and parameters is not None:
            self._latest_global = parameters_to_ndarrays(parameters)

        # capture all client ids at round 1
        if server_round == 1:
            try:
                all_clients_map = client_manager.all()
                self._all_client_ids = set(all_clients_map.keys())
            except Exception:
                self._all_client_ids = set()

        config: Dict[str, Scalar] = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        config["current_round"] = server_round

        # sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        fit_ins = FitIns(ndarrays_to_parameters(self._latest_global), config)
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []
        if self._latest_global is None and parameters is not None:
            self._latest_global = parameters_to_ndarrays(parameters)

        config: Dict[str, Scalar] = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)
        config["current_round"] = server_round

        evaluate_ins = EvaluateIns(ndarrays_to_parameters(self._latest_global), config)

        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        return [(client, evaluate_ins) for client in clients]

    def _ensure_table_entries(self) -> None:
        """Initialize zeros for clients not yet seen."""
        for cid in self._all_client_ids:
            if cid not in self._update_table and self._latest_global is not None:
                self._update_table[cid] = zeros_like(self._latest_global)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using MIFA update rule."""
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        # filter out disconnected clients (those with num_examples=0 or empty parameters)
        valid_results = [
            (client, res) for client, res in results
            if res.num_examples > 0 and len(parameters_to_ndarrays(res.parameters)) > 0
        ]
        
        if not valid_results:
            # return current global if no valid results
            if self._latest_global is not None:
                return ndarrays_to_parameters(self._latest_global), {}
            return None, {}

        if self._latest_global is None:
            self._latest_global = parameters_to_ndarrays(valid_results[0][1].parameters)

        # expand known clients set (use all results to track client IDs)
        for client, _ in results:
            cid = getattr(client, "cid", None) or str(client)
            self._all_client_ids.add(cid)

        self._ensure_table_entries()

        # compute normalized updates (only for valid results)
        # normalize by K * eta_local so updates are comparable across clients
        old_global = copy_weights(self._latest_global)

        for client, fit_res in valid_results:
            cid = getattr(client, "cid", None) or str(client)
            w_i = parameters_to_ndarrays(fit_res.parameters)
            
            # compute normalization factor based on this client's training
            num_samples = fit_res.num_examples
            norm_factor = self._normalization_factor(num_samples)
            
            # U_i = (w_i - w_t) / (K * eta_local)
            upd = [
                (np.asarray(w_i[k], dtype=np.float32) - np.asarray(old_global[k], dtype=np.float32)) / np.float32(norm_factor)
                for k in range(len(w_i))
            ]
            self._update_table[cid] = upd

        # check if table is initialized
        if self._wait_for_all and not self._table_initialized:
            self._ensure_table_entries()
            have_all = (
                len(self._update_table) > 0
                and all(cid in self._update_table for cid in self._all_client_ids)
                and len(self._all_client_ids) > 0
            )
            if not have_all:
                params = ndarrays_to_parameters(self._latest_global)
                metrics_aggregated = {}
                if self.fit_metrics_aggregation_fn:
                    fit_metrics = [(res.num_examples, res.metrics) for _, res in valid_results]
                    metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
                return params, metrics_aggregated
            self._table_initialized = True

        # compute mean of cached updates across ALL clients (including stale ones)
        self._ensure_table_entries()
        N = max(1, len(self._all_client_ids))
        mean_upd = zeros_like(self._latest_global)
        for cid in self._all_client_ids:
            add_inplace(mean_upd, self._update_table[cid], alpha=1.0 / float(N))

        # apply server update: w_{t+1} = w_t + eta_server * mean(U)
        new_global = copy_weights(old_global)
        add_inplace(new_global, mean_upd, alpha=self._server_lr)
        self._latest_global = new_global
        self._true_round += 1

        params = ndarrays_to_parameters(self._latest_global)

        # aggregate metrics (use valid_results for metrics)
        metrics_aggregated: Dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in valid_results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        # log fit metrics to wandb
        if metrics_aggregated:
            log_round_metrics(server_round, fit_metrics=metrics_aggregated)

        return params, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters."""
        if self.evaluate_fn is None:
            return None
        params_nd = self._latest_global
        if params_nd is None and parameters is not None:
            params_nd = parameters_to_ndarrays(parameters)
        if params_nd is None:
            return None
        result = self.evaluate_fn(server_round, params_nd, {})
        if result is None:
            return None
        loss, metrics = result
        return loss, metrics or {}

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses."""
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [(evaluate_res.num_examples, evaluate_res.loss) for _, evaluate_res in results]
        )
        
        metrics_aggregated: Dict[str, Scalar] = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        
        # log eval metrics to wandb
        log_round_metrics(server_round, evaluate_metrics=metrics_aggregated, loss=loss_aggregated)

        return loss_aggregated, metrics_aggregated


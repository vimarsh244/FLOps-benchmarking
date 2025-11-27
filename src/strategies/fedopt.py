"""FedOpt strategy implementations (FedAdam, FedYogi, FedAdagrad).

Paper: Adaptive Federated Optimization
       Reddi et al., 2020
       https://arxiv.org/abs/2003.00295

FedOpt applies adaptive optimizers on the server side for better convergence.
"""

from logging import WARNING
from typing import Callable, Optional, Union, List, Tuple, Dict

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
    copy_weights,
    compute_update,
    aggregate_parameters,
    get_parameters_from_results,
)
from src.utils.wandb_logger import log_round_metrics


WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server.
"""


class FedOptBase(Strategy):
    """Base class for FedOpt strategies.

    Implements common functionality for FedAdam, FedYogi, and FedAdagrad.

    Parameters
    ----------
    server_lr : float
        Server-side learning rate (eta in the paper).
    beta_1 : float
        First moment decay parameter.
    beta_2 : float
        Second moment decay parameter.
    tau : float
        Adaptivity/stability parameter.
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
        server_lr: float = 0.01,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        tau: float = 1e-3,
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

        # fedopt state
        self.server_lr = server_lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.tau = tau
        self._global_model: Optional[NDArrays] = None
        self._m: Optional[NDArrays] = None  # first moment
        self._v: Optional[NDArrays] = None  # second moment

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
            self._global_model = parameters_to_ndarrays(initial_parameters)
            self._m = zeros_like(self._global_model)
            self._v = [np.full_like(w, self.tau**2, dtype=np.float32) for w in self._global_model]
        return initial_parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        if self._global_model is None:
            self._global_model = parameters_to_ndarrays(parameters)
            self._m = zeros_like(self._global_model)
            self._v = [np.full_like(w, self.tau**2, dtype=np.float32) for w in self._global_model]

        config: Dict[str, Scalar] = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        config["current_round"] = server_round

        fit_ins = FitIns(ndarrays_to_parameters(self._global_model), config)

        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []

        if self._global_model is None:
            self._global_model = parameters_to_ndarrays(parameters)

        config: Dict[str, Scalar] = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)
        config["current_round"] = server_round

        evaluate_ins = EvaluateIns(
            ndarrays_to_parameters(self._global_model), config
        )

        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        return [(client, evaluate_ins) for client in clients]

    def _compute_pseudo_gradient(
        self, results: List[Tuple[ClientProxy, FitRes]]
    ) -> NDArrays:
        """Compute pseudo-gradient (negative of aggregated update)."""
        weights_results = get_parameters_from_results(results)
        aggregated = aggregate_parameters(weights_results, inplace=False)
        
        # pseudo-gradient = old_model - new_model (negative update)
        delta = compute_update(self._global_model, aggregated)
        return delta

    def _update_second_moment(self, delta: NDArrays) -> None:
        """Update second moment (to be overridden by subclasses)."""
        raise NotImplementedError

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results with adaptive server optimizer."""
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        # compute pseudo-gradient
        delta = self._compute_pseudo_gradient(results)

        # update first moment: m = beta_1 * m + (1 - beta_1) * delta
        for i in range(len(self._m)):
            self._m[i] = (
                self.beta_1 * self._m[i] 
                + (1 - self.beta_1) * np.asarray(delta[i], dtype=np.float32)
            )

        # update second moment (strategy-specific)
        self._update_second_moment(delta)

        # update global model: x = x + eta * m / (sqrt(v) + tau)
        for i in range(len(self._global_model)):
            self._global_model[i] = (
                self._global_model[i]
                + self.server_lr * self._m[i] / (np.sqrt(self._v[i]) + self.tau)
            )

        params = ndarrays_to_parameters(self._global_model)

        # aggregate metrics
        metrics_aggregated: Dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
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
        if self._global_model is None:
            return None
        result = self.evaluate_fn(server_round, self._global_model, {})
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


class FedAdam(FedOptBase):
    """FedAdam: Federated Adam optimizer.

    Uses Adam's second moment update rule:
        v = beta_2 * v + (1 - beta_2) * delta^2
    """

    def __repr__(self) -> str:
        return f"FedAdam(server_lr={self.server_lr}, beta_1={self.beta_1}, beta_2={self.beta_2})"

    def _update_second_moment(self, delta: NDArrays) -> None:
        """Update second moment using Adam rule."""
        for i in range(len(self._v)):
            delta_sq = np.asarray(delta[i], dtype=np.float32) ** 2
            self._v[i] = self.beta_2 * self._v[i] + (1 - self.beta_2) * delta_sq


class FedYogi(FedOptBase):
    """FedYogi: Federated Yogi optimizer.

    Uses Yogi's second moment update rule (more conservative than Adam):
        v = v - (1 - beta_2) * delta^2 * sign(v - delta^2)

    This prevents the learning rate from increasing too much in any direction.
    """

    def __repr__(self) -> str:
        return f"FedYogi(server_lr={self.server_lr}, beta_1={self.beta_1}, beta_2={self.beta_2})"

    def _update_second_moment(self, delta: NDArrays) -> None:
        """Update second moment using Yogi rule."""
        for i in range(len(self._v)):
            delta_sq = np.asarray(delta[i], dtype=np.float32) ** 2
            sign = np.sign(self._v[i] - delta_sq)
            self._v[i] = self._v[i] - (1 - self.beta_2) * delta_sq * sign


class FedAdagrad(FedOptBase):
    """FedAdagrad: Federated Adagrad optimizer.

    Uses Adagrad's second moment update rule (accumulates gradients):
        v = v + delta^2

    Note: Does not use beta_2 parameter.
    """

    def __repr__(self) -> str:
        return f"FedAdagrad(server_lr={self.server_lr}, beta_1={self.beta_1})"

    def _update_second_moment(self, delta: NDArrays) -> None:
        """Update second moment using Adagrad rule."""
        for i in range(len(self._v)):
            delta_sq = np.asarray(delta[i], dtype=np.float32) ** 2
            self._v[i] = self._v[i] + delta_sq


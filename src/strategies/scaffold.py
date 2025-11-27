"""SCAFFOLD strategy implementation.

Paper: SCAFFOLD: Stochastic Controlled Averaging for Federated Learning
       Karimireddy et al., 2020
       https://arxiv.org/abs/1910.06378

SCAFFOLD uses control variates to reduce client drift in heterogeneous settings.
"""

from logging import WARNING, INFO
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
    add_inplace,
    copy_weights,
    compute_update,
)
from src.utils.wandb_logger import log_round_metrics


WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server.
"""


class SCAFFOLD(Strategy):
    """SCAFFOLD strategy for handling client drift.

    SCAFFOLD maintains server and client control variates to correct for
    client drift caused by local updates on heterogeneous data.

    The algorithm works as follows:
    1. Server sends global model and server control variate to clients
    2. Clients perform local training with control variate correction
    3. Clients compute new control variates and send updates to server
    4. Server aggregates updates and updates global model and control variate

    Parameters
    ----------
    server_lr : float
        Server-side learning rate for aggregation.
    client_lr : float
        Client-side learning rate (used for control variate computation).
    warm_start : bool
        Whether to use warm start for control variates.
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
        server_lr: float = 1.0,
        client_lr: float = 0.01,
        warm_start: bool = False,
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

        # scaffold state
        self.server_lr = server_lr
        self.client_lr = client_lr
        self.warm_start = warm_start
        self._global_model: Optional[NDArrays] = None
        self._server_control: Optional[NDArrays] = None
        self._client_controls: Dict[str, NDArrays] = {}

    def __repr__(self) -> str:
        return f"SCAFFOLD(server_lr={self.server_lr}, client_lr={self.client_lr})"

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
            self._server_control = zeros_like(self._global_model)
        return initial_parameters

    def _get_client_control(self, cid: str) -> NDArrays:
        """Get control variate for a client, initializing if needed."""
        if cid not in self._client_controls and self._global_model is not None:
            self._client_controls[cid] = zeros_like(self._global_model)
        return self._client_controls.get(cid, [])

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure training with control variates."""
        if self._global_model is None:
            self._global_model = parameters_to_ndarrays(parameters)
            self._server_control = zeros_like(self._global_model)

        config: Dict[str, Scalar] = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        config["current_round"] = server_round
        config["client_lr"] = self.client_lr

        # sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # create fit instructions with control variates
        fit_instructions = []
        for client in clients:
            cid = getattr(client, "cid", None) or str(client)
            
            # pack model and control variates into parameters
            # format: [model_params..., server_control..., client_control...]
            client_control = self._get_client_control(cid)
            packed = self._global_model + self._server_control + client_control
            
            fit_ins = FitIns(ndarrays_to_parameters(packed), config)
            fit_instructions.append((client, fit_ins))

        return fit_instructions

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure evaluation."""
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

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results with SCAFFOLD update rule."""
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        # determine number of parameters in model
        num_model_params = len(self._global_model)

        # collect model updates and control variate updates
        model_updates: List[Tuple[NDArrays, int]] = []
        control_updates: List[NDArrays] = []

        for client, fit_res in results:
            cid = getattr(client, "cid", None) or str(client)
            all_params = parameters_to_ndarrays(fit_res.parameters)
            
            # unpack: [new_model..., new_client_control..., delta_control...]
            new_model = all_params[:num_model_params]
            new_client_control = all_params[num_model_params:2*num_model_params]
            delta_control = all_params[2*num_model_params:]
            
            # compute model update (delta_y)
            delta_y = compute_update(new_model, self._global_model)
            model_updates.append((delta_y, fit_res.num_examples))
            control_updates.append(delta_control)
            
            # update client control
            self._client_controls[cid] = new_client_control

        # aggregate model updates
        total_examples = sum(n for _, n in model_updates)
        if total_examples == 0:
            total_examples = len(model_updates)

        agg_delta_y = zeros_like(self._global_model)
        for delta_y, n in model_updates:
            weight = n / total_examples if total_examples > 0 else 1.0 / len(model_updates)
            add_inplace(agg_delta_y, delta_y, alpha=weight)

        # aggregate control updates
        num_clients = len(results)
        agg_delta_c = zeros_like(self._global_model)
        for delta_c in control_updates:
            add_inplace(agg_delta_c, delta_c, alpha=1.0 / num_clients)

        # update global model: x = x + server_lr * delta_y
        add_inplace(self._global_model, agg_delta_y, alpha=self.server_lr)

        # update server control: c = c + (|S|/N) * delta_c
        # approximation: use fraction of participating clients
        participation_fraction = num_clients / max(1, len(self._client_controls))
        add_inplace(self._server_control, agg_delta_c, alpha=participation_fraction)

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


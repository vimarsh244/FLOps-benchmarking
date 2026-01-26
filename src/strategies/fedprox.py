"""FedProx (Federated Optimization) strategy implementation.

Paper: Federated Optimization in Heterogeneous Networks
       Li et al., 2018
       https://arxiv.org/abs/1812.06127

Note: The proximal term is applied on the client side during training.
      This strategy mainly ensures the proximal_mu is sent to clients.
"""

from typing import Callable, Optional, List, Tuple, Dict

from flwr.common import (
    EvaluateIns,
    FitIns,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from src.strategies.fedavg import CustomFedAvg


class CustomFedProx(CustomFedAvg):
    """Federated Optimization strategy (FedProx).

    Implementation based on https://arxiv.org/abs/1812.06127

    The strategy itself is similar to FedAvg, but it sends an additional
    `proximal_mu` parameter to clients. The client needs to add a proximal
    term to its loss function:

        loss = original_loss + (mu/2) * ||w - w_global||^2

    where w are the local weights and w_global are the global weights.

    Parameters
    ----------
    proximal_mu : float
        The weight of the proximal term. 0.0 = FedAvg, higher = more regularization.
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
        proximal_mu: float = 0.1,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.proximal_mu = proximal_mu

    def __repr__(self) -> str:
        return (
            f"CustomFedProx(accept_failures={self.accept_failures}, proximal_mu={self.proximal_mu})"
        )

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training with proximal_mu."""
        # get base config from parent
        client_config_pairs = super().configure_fit(server_round, parameters, client_manager)

        # add proximal_mu to config
        return [
            (
                client,
                FitIns(
                    fit_ins.parameters,
                    {
                        **fit_ins.config,
                        "proximal_mu": self.proximal_mu,
                    },
                ),
            )
            for client, fit_ins in client_config_pairs
        ]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        client_config_pairs = super().configure_evaluate(server_round, parameters, client_manager)

        return [
            (
                client,
                EvaluateIns(
                    evaluate_ins.parameters,
                    {
                        **evaluate_ins.config,
                        "current_round": server_round,
                    },
                ),
            )
            for client, evaluate_ins in client_config_pairs
        ]

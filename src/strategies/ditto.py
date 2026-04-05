"""Ditto strategy implementation.

Paper: Ditto: Fair and Robust Federated Learning Through Personalization
           Li et al., 2021
           https://arxiv.org/abs/2012.04221

Ditto trains a global model via FedAvg while each client maintains a
personalized model trained with a proximal regularization term to the
global model. The personalized model is kept on the client and is not
aggregated by the server.
"""

from typing import Callable, Dict, List, Optional, Tuple

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


class Ditto(CustomFedAvg):
    """Ditto strategy (server-side).

    Ditto behaves like FedAvg on the server, but sends extra configuration
    so clients can update their personalized models using a proximal term.

    Parameters
    ----------
    ditto_mu : float
            Proximal regularization strength for personalized training.
    personal_epochs : Optional[int]
            Number of epochs for personalized training on client. If None,
            clients will use their local_epochs.
    evaluate_personalized : bool
            If True, clients evaluate using personalized model if available.
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
        inplace: bool = True,
        ditto_mu: float = 0.1,
        personal_epochs: Optional[int] = None,
        evaluate_personalized: bool = True,
        local_iters: Optional[int] = None,
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
            inplace=inplace,
        )
        self.ditto_mu = ditto_mu
        self.personal_epochs = personal_epochs
        self.evaluate_personalized = evaluate_personalized
        self.local_iters = local_iters

    def __repr__(self) -> str:
        return "Ditto(accept_failures=" f"{self.accept_failures}, ditto_mu={self.ditto_mu})"

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure training with Ditto personalization parameters."""
        client_config_pairs = super().configure_fit(server_round, parameters, client_manager)

        return [
            (
                client,
                FitIns(
                    fit_ins.parameters,
                    {**fit_ins.config,
                        "strategy": "ditto",
                        "ditto_mu": self.ditto_mu,
                        "personal_epochs": (
                            self.personal_epochs if self.personal_epochs is not None else 0
                        ),
                        **({
                            "local_iters": self.local_iters
                        } if self.local_iters is not None else {}),
                    },
                ),
            )
            for client, fit_ins in client_config_pairs
        ]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure evaluation with Ditto personalization flags."""
        client_config_pairs = super().configure_evaluate(server_round, parameters, client_manager)

        return [
            (
                client,
                EvaluateIns(
                    evaluate_ins.parameters,
                    {
                        **evaluate_ins.config,
                        "strategy": "ditto",
                        "evaluate_personalized": self.evaluate_personalized,
                    },
                ),
            )
            for client, evaluate_ins in client_config_pairs
        ]

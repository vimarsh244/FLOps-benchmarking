"""Distribution Informed Weight Substitution (DIWS) strategy implementation."""

from __future__ import annotations

import pickle
from concurrent.futures import ThreadPoolExecutor
from math import floor
from typing import Dict, List, Optional, Tuple, Union

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate_inplace
from flwr.server.strategy.strategy import Strategy

from src.strategies.fedavg import CustomFedAvg


class DIWS(Strategy):
    """Distribution Informed Weight Substitution (DIWS) strategy.

    Wrapper for any Flower Strategy that substitutes weights for dropped clients.
    """

    def __init__(
        self,
        *,
        aggregator_strategy: Optional[Strategy] = None,
        substitution_timeout: float = 600.0,
    ) -> None:
        super().__init__()
        self.aggregator_strategy = aggregator_strategy or CustomFedAvg()
        self.substitution_timeout = substitution_timeout
        self.global_parameters: Optional[Parameters] = None
        self.label_distribution: Dict[str, Dict[int, int]] = {}

    def __repr__(self) -> str:
        return repr(self.aggregator_strategy)

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        self.global_parameters = self.aggregator_strategy.initialize_parameters(
            client_manager
        )
        return self.global_parameters

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return self.aggregator_strategy.evaluate(server_round, parameters)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        self.global_parameters = parameters
        return self.aggregator_strategy.configure_fit(
            server_round, parameters, client_manager
        )

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        return self.aggregator_strategy.configure_evaluate(
            server_round, parameters, client_manager
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results, substituting dropped clients if needed."""
        if server_round == 1:
            for client_proxy, fitres in results:
                if not fitres.metrics or "label_distribution" not in fitres.metrics:
                    continue
                client_label_distribution = pickle.loads(
                    fitres.metrics.get("label_distribution")
                )
                self.label_distribution[client_proxy.cid] = client_label_distribution

        print(f"Number of results before substitution: {len(results)}")
        self.substitute_dropped_clients(server_round, results, failures)
        print(f"Number of results after substitution: {len(results)}")

        return self.aggregator_strategy.aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        aggregated = self.aggregator_strategy.aggregate_evaluate(
            server_round, results, failures
        )
        print(f"Results of aggregate evaluate: {aggregated}")
        return aggregated

    def substitute_dropped_clients(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> None:
        """Substitute dropped clients with subset training, if required."""
        if len(failures) == 0:
            print(f"No dropped clients to substitute in round {server_round}.")
            return

        active_client_ids = [client_proxy.cid for client_proxy, _ in results]
        client_subset_distributions = self.get_subset_distribution_for_active_clients(
            active_client_ids
        )
        if not client_subset_distributions:
            print(
                f"No subset distributions available for round {server_round}; skipping."
            )
            return

        with ThreadPoolExecutor() as executor:
            futures = []
            for client_proxy, _ in results:
                subset_distribution_bytes = pickle.dumps(
                    client_subset_distributions[client_proxy.cid]
                )
                config = {
                    "subset_distribution": subset_distribution_bytes,
                    "custom_rpc": "handle_missing_clients",
                }
                fit_ins = FitIns(parameters=self.global_parameters, config=config)
                futures.append(
                    executor.submit(
                        client_proxy.fit,
                        fit_ins,
                        self.substitution_timeout,
                        server_round,
                    )
                )
            outputs = [future.result() for future in futures]

        substituted_parameters_fitres = self.aggregate_substitution_parameters(outputs)
        results.append((None, substituted_parameters_fitres))

    def consolidate_label_distributions(
        self, active_clients_ids: List[str]
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        dropped_clients_ids = set(self.label_distribution.keys()) - set(
            active_clients_ids
        )
        print(f"Dropped clients IDs: {dropped_clients_ids}")
        print(f"Active clients IDs: {active_clients_ids}")

        dropped_clients_distribution: Dict[int, int] = {}
        for cid in dropped_clients_ids:
            client_dist = self.label_distribution.get(cid, {})
            for label, count in client_dist.items():
                dropped_clients_distribution[label] = (
                    dropped_clients_distribution.get(label, 0) + count
                )

        active_clients_distribution: Dict[int, int] = {}
        for cid in active_clients_ids:
            client_dist = self.label_distribution.get(cid, {})
            for label, count in client_dist.items():
                active_clients_distribution[label] = (
                    active_clients_distribution.get(label, 0) + count
                )

        print(f"Dropped clients distribution: {dropped_clients_distribution}")
        print(f"Active clients distribution: {active_clients_distribution}")

        return dropped_clients_distribution, active_clients_distribution

    def get_consolidated_representative_distribution(
        self,
        dropped_clients_distribution: Dict[int, int],
        active_clients_distribution: Dict[int, int],
    ) -> Dict[int, int]:
        representative_subset_distribution: Dict[int, int] = {}
        total_dropped = sum(dropped_clients_distribution.values())
        if total_dropped == 0:
            return representative_subset_distribution

        target_percentages = {
            label: count / total_dropped
            for label, count in dropped_clients_distribution.items()
        }

        anchor_label = max(
            dropped_clients_distribution,
            key=lambda label: dropped_clients_distribution[label],
        )

        representative_subset_distribution[anchor_label] = min(
            active_clients_distribution.get(anchor_label, 0),
            dropped_clients_distribution[anchor_label],
        )

        anchor_label_total = floor(
            representative_subset_distribution[anchor_label]
            / target_percentages[anchor_label]
        )

        for label, count in active_clients_distribution.items():
            if label == anchor_label:
                continue
            target_count = floor(target_percentages[label] * anchor_label_total)
            representative_subset_distribution[label] = min(target_count, count)

        print(
            f"Subset distribution for active clients: {representative_subset_distribution}"
        )
        return representative_subset_distribution

    def get_subset_distribution_for_active_clients(
        self, active_clients_ids: List[str]
    ) -> Dict[str, Dict[int, int]]:
        """Get representative subset distribution for active clients."""
        dropped_clients_distribution, active_clients_distribution = (
            self.consolidate_label_distributions(active_clients_ids)
        )

        representative_subset_distribution = (
            self.get_consolidated_representative_distribution(
                dropped_clients_distribution, active_clients_distribution
            )
        )
        if not representative_subset_distribution:
            return {cid: {} for cid in active_clients_ids}

        subset_distribution_per_client: Dict[str, Dict[int, int]] = {
            cid: {} for cid in active_clients_ids
        }
        for label, total_needed in representative_subset_distribution.items():
            client_counts = [
                (cid, self.label_distribution.get(cid, {}).get(label, 0))
                for cid in active_clients_ids
            ]
            idx = 0
            while total_needed > 0 and any(count > 0 for _, count in client_counts):
                cid, available = client_counts[idx % len(client_counts)]
                if available > 0:
                    subset_distribution_per_client[cid][label] = (
                        subset_distribution_per_client[cid].get(label, 0) + 1
                    )
                    client_counts[idx % len(client_counts)] = (cid, available - 1)
                    total_needed -= 1
                idx += 1

        print(f"Subset distribution per client: {subset_distribution_per_client}")
        return subset_distribution_per_client

    def aggregate_substitution_parameters(self, results: List[FitRes]) -> FitRes:
        results_with_proxies = [(None, fit_res) for fit_res in results]
        aggregated_parameters = aggregate_inplace(results_with_proxies)
        total_samples = sum(fit_res.num_examples for _, fit_res in results_with_proxies)

        aggregated_results = FitRes(
            parameters=ndarrays_to_parameters(aggregated_parameters),
            num_examples=total_samples,
            metrics=None,
            status=None,
        )
        return aggregated_results

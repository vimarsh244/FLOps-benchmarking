"""Clustered Federated Learning strategy implementation.

Based on: Clustered Federated Learning: Model-Agnostic Distributed Multi-Task 
          Optimization under Privacy Constraints
          Sattler et al., 2019
          https://arxiv.org/abs/1910.01991

Maintains per-cluster models and dynamically splits clusters based on
client update similarity.
"""

from logging import WARNING, INFO
from typing import Callable, Optional, Union, List, Tuple, Dict, Set
from collections import defaultdict

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

from src.strategies.base import weighted_average
from src.utils.wandb_logger import log_round_metrics


WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server.
"""


class CustomClusteredFL(Strategy):
    """Clustered Federated Learning strategy.

    Based on: Clustered Federated Learning: Model-Agnostic Distributed Multi-Task
              Optimization under Privacy Constraints (Sattler et al., 2019)

    Maintains per-cluster models and sends each client the model of its cluster.
    Supports dynamic client disconnect/rejoin and on-the-fly cluster splits.

    Split detection uses cosine similarity between client updates (as per paper):
    - If mean pairwise cosine similarity is high (clients agree) -> no split
    - If min pairwise cosine similarity is low (clients disagree) -> trigger split

    Parameters
    ----------
    cosine_similarity_threshold : float
        Minimum mean cosine similarity to keep cluster together.
        Below this triggers split consideration. Default 0.7.
    min_cosine_similarity : float
        If min pairwise cosine similarity falls below this, consider splitting.
        Default 0.3.
    split_warmup_rounds : int
        Number of rounds to wait before allowing splits.
    split_cooldown_rounds : int
        Number of rounds to wait between splits.
    min_clients_for_split : int
        Minimum number of clients in a cluster to consider splitting.
    min_cluster_size : int
        Minimum clients per resulting cluster after split.
    max_kmeans_iters : int
        Maximum iterations for spherical k-means.
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
        # cfl specific knobs (cosine similarity based, per paper)
        cosine_similarity_threshold: float = 0.7,
        min_cosine_similarity: float = 0.3,
        split_warmup_rounds: int = 5,
        split_cooldown_rounds: int = 3,
        min_clients_for_split: int = 3,
        min_cluster_size: int = 2,
        max_kmeans_iters: int = 10,
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
        self.inplace = inplace

        # cfl state
        self._cluster_models: Dict[int, NDArrays] = {}
        self._client_to_cluster: Dict[str, int] = {}
        self._cluster_clients: Dict[int, Set[str]] = defaultdict(set)
        self._round_assignments: Dict[int, Dict[str, int]] = {}
        self._last_split_round: int = 0
        self._next_cluster_id: int = 0

        # cfl params (cosine similarity based)
        self.cosine_sim_threshold = cosine_similarity_threshold
        self.min_cosine_sim = min_cosine_similarity
        self.split_warmup_rounds = split_warmup_rounds
        self.split_cooldown_rounds = split_cooldown_rounds
        self.min_clients_for_split = min_clients_for_split
        self.min_cluster_size = min_cluster_size
        self.max_kmeans_iters = max_kmeans_iters

    def __repr__(self) -> str:
        return f"CustomClusteredFL(accept_failures={self.accept_failures}, clusters={len(self._cluster_models)})"

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize the first cluster with the provided initial parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None
        if initial_parameters is not None:
            self._cluster_models = {0: parameters_to_ndarrays(initial_parameters)}
            self._next_cluster_id = 1
        return initial_parameters

    def _select_initial_cluster_for(self, cid: str) -> int:
        """Select cluster for a client (reuse if seen before)."""
        if cid in self._client_to_cluster:
            return self._client_to_cluster[cid]
        if not self._cluster_models:
            return 0
        # choose the cluster with most members
        if self._cluster_clients:
            sizes = {k: len(v) for k, v in self._cluster_clients.items()}
            return max(sizes, key=sizes.get)
        return 0

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure training with per-client cluster models."""
        config: Dict[str, Scalar] = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        config["current_round"] = server_round

        # seed default cluster if uninitialized
        if not self._cluster_models and parameters is not None:
            self._cluster_models[0] = parameters_to_ndarrays(parameters)
            self._next_cluster_id = 1

        # sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # assign per-client cluster and send that cluster's model
        round_assignments: Dict[str, int] = {}
        fit_instructions: List[Tuple[ClientProxy, FitIns]] = []
        
        for client in clients:
            cid = getattr(client, "cid", None) or str(client)
            cluster_id = self._select_initial_cluster_for(cid)
            round_assignments[cid] = cluster_id
            self._client_to_cluster[cid] = cluster_id
            self._cluster_clients[cluster_id].add(cid)
            cluster_params = ndarrays_to_parameters(self._cluster_models[cluster_id])
            cfg = {**config, "cluster_id": cluster_id}
            fit_instructions.append((client, FitIns(cluster_params, cfg)))

        self._round_assignments[server_round] = round_assignments
        return fit_instructions

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure evaluation with per-client cluster models."""
        if self.fraction_evaluate == 0.0:
            return []

        config: Dict[str, Scalar] = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)
        config["current_round"] = server_round

        if not self._cluster_models and parameters is not None:
            self._cluster_models[0] = parameters_to_ndarrays(parameters)
            self._next_cluster_id = 1

        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        eval_instructions: List[Tuple[ClientProxy, EvaluateIns]] = []
        for client in clients:
            cid = getattr(client, "cid", None) or str(client)
            cluster_id = self._client_to_cluster.get(
                cid, self._select_initial_cluster_for(cid)
            )
            params = ndarrays_to_parameters(self._cluster_models[cluster_id])
            cfg = {**config, "cluster_id": cluster_id}
            eval_instructions.append((client, EvaluateIns(params, cfg)))

        return eval_instructions

    @staticmethod
    def _weighted_aggregate(
        results: List[Tuple[NDArrays, int]], inplace: bool = True
    ) -> NDArrays:
        """Compute weighted average of model parameters."""
        if not results:
            raise ValueError("No results to aggregate")

        total_examples = sum(n for _, n in results)
        if total_examples == 0:
            return results[0][0]

        aggregated = []
        for i in range(len(results[0][0])):
            layer_sum = np.zeros_like(results[0][0][i], dtype=np.float32)
            for weights, n in results:
                layer_sum += (n / total_examples) * np.asarray(weights[i], dtype=np.float32)
            aggregated.append(layer_sum)

        return aggregated

    @staticmethod
    def _flatten_params_difference(new: NDArrays, old: NDArrays) -> np.ndarray:
        """Flatten the difference between new and old parameters."""
        vecs = [
            (np.asarray(n, dtype=np.float32) - np.asarray(o, dtype=np.float32)).ravel()
            for n, o in zip(new, old)
        ]
        if not vecs:
            return np.array([], dtype=np.float32)
        return np.concatenate(vecs)

    @staticmethod
    def _compute_cosine_similarities(
        update_vectors: List[np.ndarray],
    ) -> Tuple[float, float]:
        """Compute pairwise cosine similarities between update vectors.
        
        As per Sattler et al. paper, we compute cosine similarity between
        client gradient updates to determine if they should be in the same cluster.
        
        Args:
            update_vectors: List of flattened update vectors
        
        Returns:
            Tuple of (mean_cosine_similarity, min_cosine_similarity)
        """
        if len(update_vectors) < 2:
            return 1.0, 1.0
        
        eps = 1e-12
        
        # normalize vectors
        norms = [np.linalg.norm(v) + eps for v in update_vectors]
        normalized = [v / n for v, n in zip(update_vectors, norms)]
        
        # compute pairwise cosine similarities
        similarities = []
        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                sim = float(np.dot(normalized[i], normalized[j]))
                similarities.append(sim)
        
        if not similarities:
            return 1.0, 1.0
        
        return float(np.mean(similarities)), float(np.min(similarities))

    def _binary_spherical_kmeans(self, X: np.ndarray) -> np.ndarray:
        """k=2 spherical k-means for cluster splitting."""
        m = X.shape[0]
        if m < 2:
            return np.zeros(m, dtype=int)

        eps = 1e-12
        norms = np.linalg.norm(X, axis=1, keepdims=True) + eps
        Xn = X / norms

        # init centers by picking two farthest points
        S = Xn @ Xn.T
        i = 0
        j = int(np.argmin(S[0]))
        for _ in range(2):
            i = int(np.argmin(S[j]))
            j = int(np.argmin(S[i]))

        c0, c1 = Xn[i], Xn[j]
        labels = np.zeros(m, dtype=int)

        for _ in range(self.max_kmeans_iters):
            sim0 = Xn @ c0
            sim1 = Xn @ c1
            new_labels = (sim1 > sim0).astype(int)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels

            if np.any(labels == 0):
                c0 = Xn[labels == 0].mean(axis=0)
                c0 = c0 / (np.linalg.norm(c0) + eps)
            if np.any(labels == 1):
                c1 = Xn[labels == 1].mean(axis=0)
                c1 = c1 / (np.linalg.norm(c1) + eps)

        return labels

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results with cluster-aware aggregation and splitting."""
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
            return None, {}

        # group results by cluster
        round_assign = self._round_assignments.get(server_round, {})
        grouped: Dict[int, List[Tuple[str, NDArrays, int]]] = defaultdict(list)

        for client, fit_res in valid_results:
            cid = getattr(client, "cid", None) or str(client)
            params_nd = parameters_to_ndarrays(fit_res.parameters)
            n = fit_res.num_examples
            cluster_id = round_assign.get(cid, self._client_to_cluster.get(cid, 0))
            grouped[cluster_id].append((cid, params_nd, n))

        # aggregate per cluster and check for splits
        cluster_new_models: Dict[int, NDArrays] = {}
        split_candidates: List[int] = []

        for cid_cluster, items in grouped.items():
            old_model = self._cluster_models[cid_cluster]
            agg = self._weighted_aggregate([(p, n) for _, p, n in items], self.inplace)
            cluster_new_models[cid_cluster] = agg

            # compute update vectors for split detection
            upd_vecs = []
            for _, p, _ in items:
                dv = self._flatten_params_difference(p, old_model)
                upd_vecs.append(dv)

            if len(upd_vecs) >= self.min_clients_for_split:
                # compute pairwise cosine similarities (as per Sattler et al. paper)
                mean_cosine, min_cosine = self._compute_cosine_similarities(upd_vecs)
                
                # split if cosine similarity indicates divergence among clients
                should_split = (
                    server_round >= self.split_warmup_rounds
                    and server_round - self._last_split_round >= self.split_cooldown_rounds
                    and (mean_cosine < self.cosine_sim_threshold or min_cosine < self.min_cosine_sim)
                )
                
                if should_split:
                    split_candidates.append(cid_cluster)

                log(
                    INFO,
                    f"[CFL] Round {server_round} cluster {cid_cluster}: "
                    f"mean_cosine={mean_cosine:.4f}, min_cosine={min_cosine:.4f}"
                )

        # apply at most one split per round
        for cid_cluster in split_candidates[:1]:
            items = grouped[cid_cluster]
            old_model = self._cluster_models[cid_cluster]
            X = np.stack(
                [self._flatten_params_difference(p, old_model) for _, p, _ in items],
                axis=0,
            )
            labels = self._binary_spherical_kmeans(X)
            g0 = [it for it, lbl in zip(items, labels) if lbl == 0]
            g1 = [it for it, lbl in zip(items, labels) if lbl == 1]

            if len(g0) >= self.min_cluster_size and len(g1) >= self.min_cluster_size:
                # swap so g0 is larger
                if len(g1) > len(g0):
                    g0, g1 = g1, g0
                    labels = 1 - labels

                new0 = self._weighted_aggregate([(p, n) for _, p, n in g0], self.inplace)
                new1 = self._weighted_aggregate([(p, n) for _, p, n in g1], self.inplace)

                self._cluster_models[cid_cluster] = new0
                new_cluster_id = self._next_cluster_id
                self._next_cluster_id += 1
                self._cluster_models[new_cluster_id] = new1

                # reassign clients
                self._cluster_clients[cid_cluster].clear()
                self._cluster_clients[new_cluster_id].clear()
                for (cid, _, _), lbl in zip(items, labels):
                    assigned = cid_cluster if lbl == 0 else new_cluster_id
                    self._client_to_cluster[cid] = assigned
                    self._cluster_clients[assigned].add(cid)

                self._last_split_round = server_round
                log(
                    INFO,
                    f"[CFL] Split cluster {cid_cluster} -> {cid_cluster} & {new_cluster_id} "
                    f"at round {server_round}"
                )

        # update cluster models without split
        for c, new_model in cluster_new_models.items():
            self._cluster_models[c] = new_model

        # build global placeholder (weighted avg over clusters)
        total_examples = sum(n for items in grouped.values() for *_, n in items)
        if total_examples == 0:
            any_params = next(iter(self._cluster_models.values()))
            global_params = ndarrays_to_parameters(any_params)
        else:
            accum = None
            for c, items in grouped.items():
                n_c = sum(n for *_, n in items)
                w = np.float32(n_c / total_examples)
                model = self._cluster_models[c]
                if accum is None:
                    accum = [np.asarray(ww, dtype=np.float32) * w for ww in model]
                else:
                    for i in range(len(accum)):
                        accum[i] += w * np.asarray(model[i], dtype=np.float32)
            global_params = ndarrays_to_parameters(
                accum if accum is not None else next(iter(self._cluster_models.values()))
            )

        # aggregate metrics (use valid_results for metrics)
        metrics_aggregated: Dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in valid_results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        # log fit metrics to wandb
        if metrics_aggregated:
            log_round_metrics(server_round, fit_metrics=metrics_aggregated)

        return global_params, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate using largest cluster model."""
        if self.evaluate_fn is None:
            return None
        if self._cluster_clients:
            largest = max(self._cluster_clients.items(), key=lambda kv: len(kv[1]))[0]
            params_nd = self._cluster_models[largest]
        else:
            params_nd = next(iter(self._cluster_models.values()))
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


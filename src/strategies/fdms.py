"""Friend discovery and model substitution (FDMS) strategy implementation.

Based on: Combating Client Dropout in Federated Learning via Friend Model Substitution
          Wang & Xu, 2022
          https://arxiv.org/abs/2205.13222
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.strategy import Strategy

from src.strategies.base import aggregate_parameters, compute_update
from src.strategies.fedavg import CustomFedAvg


class FDMS(Strategy):
    """Federated learning with friend discovery and model substitution."""

    def __init__(
        self,
        *,
        aggregator_strategy: Optional[Strategy] = None,
        min_common_rounds: int = 1,
        fallback_mode: str = "mean",
        similarity_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.aggregator_strategy = aggregator_strategy or CustomFedAvg()
        self.min_common_rounds = max(min_common_rounds, 1)
        self.fallback_mode = fallback_mode
        self.similarity_eps = similarity_eps

        self.global_parameters: Optional[Parameters] = None
        self._round_sampled_clients: Dict[int, List[str]] = {}
        self._pair_similarity: Dict[Tuple[str, str], float] = {}
        self._pair_counts: Dict[Tuple[str, str], int] = {}
        self._last_update_vectors: Dict[str, np.ndarray] = {}
        self._last_weights: Dict[str, List[np.ndarray]] = {}
        self._last_num_examples: Dict[str, int] = {}

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
        fit_instructions = self.aggregator_strategy.configure_fit(
            server_round, parameters, client_manager
        )
        sampled_clients = [
            getattr(client, "cid", None) or str(client) for client, _ in fit_instructions
        ]
        self._round_sampled_clients[server_round] = sampled_clients
        return fit_instructions

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
        if not results:
            return self.aggregator_strategy.aggregate_fit(server_round, results, failures)

        # filter out invalid results before friend discovery
        valid_results = [
            (client, res)
            for client, res in results
            if res.num_examples > 0 and len(parameters_to_ndarrays(res.parameters)) > 0
        ]
        if not valid_results:
            return self.aggregator_strategy.aggregate_fit(server_round, results, failures)

        active_ids = [
            getattr(client, "cid", None) or str(client) for client, _ in valid_results
        ]
        sampled_ids = self._round_sampled_clients.get(server_round, active_ids)
        dropout_ids = [cid for cid in sampled_ids if cid not in active_ids]

        global_weights = (
            parameters_to_ndarrays(self.global_parameters)
            if self.global_parameters is not None
            else None
        )

        active_params: Dict[str, List[np.ndarray]] = {}
        active_vectors: Dict[str, np.ndarray] = {}
        total_active_examples = 0

        for client, res in valid_results:
            cid = getattr(client, "cid", None) or str(client)
            params = parameters_to_ndarrays(res.parameters)
            active_params[cid] = params
            self._last_weights[cid] = params
            self._last_num_examples[cid] = res.num_examples
            total_active_examples += res.num_examples

            if global_weights is not None:
                update = compute_update(params, global_weights)
                vector = self._flatten_update(update)
                active_vectors[cid] = vector
                self._last_update_vectors[cid] = vector

        if active_vectors:
            self._update_similarity_scores(active_ids, active_vectors)

        substituted_results = self._build_substituted_results(
            dropout_ids=dropout_ids,
            active_ids=active_ids,
            active_params=active_params,
            active_vectors=active_vectors,
            total_active_examples=total_active_examples,
        )

        augmented_results = valid_results + substituted_results
        return self.aggregator_strategy.aggregate_fit(
            server_round, augmented_results, failures
        )

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        return self.aggregator_strategy.aggregate_evaluate(
            server_round, results, failures
        )

    def _build_substituted_results(
        self,
        *,
        dropout_ids: List[str],
        active_ids: List[str],
        active_params: Dict[str, List[np.ndarray]],
        active_vectors: Dict[str, np.ndarray],
        total_active_examples: int,
    ) -> List[Tuple[ClientProxy, FitRes]]:
        if not dropout_ids or not active_ids:
            return []

        avg_examples = max(int(total_active_examples / len(active_ids)), 1)
        fallback_params = self._get_fallback_parameters(active_params)
        substituted: List[Tuple[ClientProxy, FitRes]] = []

        for dropout_id in dropout_ids:
            friend_id = self._select_friend(
                dropout_id=dropout_id,
                active_ids=active_ids,
                active_vectors=active_vectors,
            )
            if friend_id is not None and friend_id in active_params:
                substitute_params = active_params[friend_id]
            else:
                substitute_params = fallback_params

            num_examples = self._last_num_examples.get(dropout_id, avg_examples)
            substituted.append(
                (
                    None,
                    FitRes(
                        parameters=ndarrays_to_parameters(substitute_params),
                        num_examples=num_examples,
                        metrics={},
                        status=None,
                    ),
                )
            )

        return substituted

    def _select_friend(
        self,
        *,
        dropout_id: str,
        active_ids: List[str],
        active_vectors: Dict[str, np.ndarray],
    ) -> Optional[str]:
        best_id = None
        best_score = -1.0

        for candidate_id in active_ids:
            score = self._get_pair_similarity(dropout_id, candidate_id)
            if score is not None and score > best_score:
                best_score = score
                best_id = candidate_id

        if best_id is not None:
            return best_id

        if dropout_id not in self._last_update_vectors:
            return None

        dropout_vec = self._last_update_vectors[dropout_id]
        for candidate_id in active_ids:
            candidate_vec = active_vectors.get(candidate_id)
            if candidate_vec is None:
                continue
            score = self._cosine_similarity(dropout_vec, candidate_vec)
            if score > best_score:
                best_score = score
                best_id = candidate_id

        return best_id

    def _get_pair_similarity(self, cid_a: str, cid_b: str) -> Optional[float]:
        key = self._pair_key(cid_a, cid_b)
        count = self._pair_counts.get(key, 0)
        if count < self.min_common_rounds:
            return None
        return self._pair_similarity.get(key)

    def _update_similarity_scores(
        self, active_ids: List[str], active_vectors: Dict[str, np.ndarray]
    ) -> None:
        for i in range(len(active_ids)):
            cid_i = active_ids[i]
            vec_i = active_vectors.get(cid_i)
            if vec_i is None:
                continue
            for j in range(i + 1, len(active_ids)):
                cid_j = active_ids[j]
                vec_j = active_vectors.get(cid_j)
                if vec_j is None:
                    continue
                score = self._cosine_similarity(vec_i, vec_j)
                key = self._pair_key(cid_i, cid_j)
                prev_count = self._pair_counts.get(key, 0)
                prev_score = self._pair_similarity.get(key, score)
                new_score = (prev_score * prev_count + score) / (prev_count + 1)
                self._pair_similarity[key] = float(new_score)
                self._pair_counts[key] = prev_count + 1

    def _get_fallback_parameters(
        self, active_params: Dict[str, List[np.ndarray]]
    ) -> List[np.ndarray]:
        if not active_params:
            if self._last_weights:
                return next(iter(self._last_weights.values()))
            return []

        if self.fallback_mode == "mean":
            weights_results = [
                (params, self._last_num_examples.get(cid, 1))
                for cid, params in active_params.items()
            ]
            return aggregate_parameters(weights_results, inplace=False)

        if self.fallback_mode == "latest":
            latest = next(iter(active_params.values()))
            return latest

        return aggregate_parameters(
            [(params, 1) for params in active_params.values()], inplace=False
        )

    def _pair_key(self, cid_a: str, cid_b: str) -> Tuple[str, str]:
        return (cid_a, cid_b) if cid_a <= cid_b else (cid_b, cid_a)

    def _flatten_update(self, update: List[np.ndarray]) -> np.ndarray:
        if not update:
            return np.array([], dtype=np.float32)
        return np.concatenate(
            [np.asarray(layer, dtype=np.float32).ravel() for layer in update],
            axis=0,
        )

    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        if vec_a.size == 0 or vec_b.size == 0:
            return 0.0
        norm_a = np.linalg.norm(vec_a) + self.similarity_eps
        norm_b = np.linalg.norm(vec_b) + self.similarity_eps
        cos = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
        return 0.5 * (cos + 1.0)
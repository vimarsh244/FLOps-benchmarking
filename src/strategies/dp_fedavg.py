import numpy as np
from typing import List, Tuple, Union, Dict, Optional
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes

from src.strategies.fedavg import CustomFedAvg


def compute_epsilon(noise_multiplier: float, num_rounds: int, sampling_rate: float, delta: float) -> float:
    """Compute actual ε being achieved using RDP accountant."""
    from autodp import mechanism_zoo
    from autodp.transformer_zoo import Composition

    subsampled = mechanism_zoo.SubsampleGaussianMechanism(
        params={
            "prob": sampling_rate,
            "sigma": noise_multiplier,
            "coeff": 1,
        }
    )
    compose = Composition()
    composed = compose([subsampled], [num_rounds])
    return composed.get_approxDP(delta)


class DPFedAvg(CustomFedAvg):
    """FedAvg with server-side clipping and Gaussian DP noise."""

    def __init__(self, noise_multiplier: float, clipping_norm: float, num_rounds: int, delta: float = 1e-5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_multiplier = noise_multiplier
        self.clipping_norm = clipping_norm
        self.num_rounds = num_rounds
        self.delta = delta
        self.sampling_rate = self.fraction_fit
        #Save a copy BEFORE parent clears self.initial_parameters
        self.global_parameters = (
            parameters_to_ndarrays(self.initial_parameters)
            if self.initial_parameters is not None
            else None
        )
        #Instance-level RNG — true randomness per instance, not global seed
        # For DP we want different noise each round, not reproducible noise
        self.rng = np.random.default_rng()

    # ---------------- Clip client update ----------------
    def _clip_update(self, update: List[np.ndarray]) -> List[np.ndarray]:
        flat = np.concatenate([layer.ravel() for layer in update])
        total_norm = np.linalg.norm(flat)
        if total_norm <= self.clipping_norm:
            return update
        scale = self.clipping_norm / (total_norm + 1e-10)
        return [layer * scale for layer in update]

    # ---------------- Add DP noise ----------------
    def _add_noise(self, aggregated: List[np.ndarray], total_weight: float) -> List[np.ndarray]:
        stddev = self.noise_multiplier * self.clipping_norm / total_weight
        return [
            layer + self.rng.normal(0, stddev, layer.shape)
            for layer in aggregated
        ]

    # ---------------- DP Aggregation ----------------
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not results:
            return None, {}

       
        if self.global_parameters is None:
            print(f"[DPFedAvg] Round {server_round}: global_parameters not initialized, skipping")
            return None, {}

        global_weights = self.global_parameters

        # Filter out invalid results
        valid_results = [
            (client, res)
            for client, res in results
            if res.num_examples > 0 and len(parameters_to_ndarrays(res.parameters)) > 0
        ]

        dropped_count = len(results) - len(valid_results)
        if dropped_count > 0:
            print(
                f"[DPFedAvg] Round {server_round}: {len(results)} results, "
                f"{len(valid_results)} valid, {dropped_count} dropped"
            )

        if not valid_results:
            return None, {}

        client_deltas = []
        weights = []

        for _, fit_res in valid_results:
            client_weights = parameters_to_ndarrays(fit_res.parameters)

            # Compute client update (delta)
            delta = [cw - gw for cw, gw in zip(client_weights, global_weights)]

            # Clip update
            clipped_delta = self._clip_update(delta)
            client_deltas.append(clipped_delta)
            weights.append(fit_res.num_examples)

        # Weighted average of clipped deltas
        total_weight = sum(weights)
        avg_delta = [
            sum(delta[i] * w / total_weight for delta, w in zip(client_deltas, weights))
            for i in range(len(client_deltas[0]))
        ]

        #Add DP noise
        avg_delta = self._add_noise(avg_delta, total_weight)

        #Apply update to global model
        new_global = [gw + d for gw, d in zip(global_weights, avg_delta)]

        # Save for next round
        self.global_parameters = new_global

        parameters_aggregated = ndarrays_to_parameters(new_global)

        # Log actual epsilon being spent this round
        try:
            current_eps = compute_epsilon(
                self.noise_multiplier, server_round, self.sampling_rate, self.delta
            )
            print(f"[DPFedAvg] Round {server_round}: actual ε = {current_eps:.4f} (δ = {self.delta})")
        except Exception as e:
            print(f"[DPFedAvg] Round {server_round}: epsilon computation failed: {e}")

        #Aggregate and log metrics
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in valid_results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return parameters_aggregated, metrics_aggregated
import numpy as np
from typing import List
from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters


class DPFedAvg(FedAvg):
    """FedAvg with server-side clipping and Gaussian DP noise."""

    def __init__(self, noise_multiplier: float, clipping_norm: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_multiplier = noise_multiplier
        self.clipping_norm = clipping_norm
        self.global_parameters = None  # Server model stored here
        np.random.seed(42)

    # Flower calls this ONCE at the beginning
    def initialize_parameters(self, client_manager):
        return self.initial_parameters

    # ---------------- Clip client update ----------------
    def _clip_update(self, update: List[np.ndarray]) -> List[np.ndarray]:
        total_norm = np.sqrt(sum(np.sum(layer ** 2) for layer in update))
        if total_norm <= self.clipping_norm:
            return update
        scale = self.clipping_norm / (total_norm + 1e-10)
        return [layer * scale for layer in update]

    # ---------------- Add DP noise ----------------
    def _add_noise(self, aggregated, total_weight):
        stddev = self.noise_multiplier * self.clipping_norm / total_weight
        return [
        layer + np.random.normal(0, stddev, layer.shape)
        for layer in aggregated
    ]


    # ---------------- DP Aggregation ----------------
    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        #  First round: initialize server model
        if self.global_parameters is None:
            self.global_parameters = parameters_to_ndarrays(self.initial_parameters)

        global_weights = self.global_parameters

        client_deltas = []
        weights = []

        for _, fit_res in results:
            client_weights = parameters_to_ndarrays(fit_res.parameters)

            # Compute client update (delta)
            delta = [cw - gw for cw, gw in zip(client_weights, global_weights)]

            # Clip update
            clipped_delta = self._clip_update(delta)
            client_deltas.append(clipped_delta)
            weights.append(fit_res.num_examples)

        # Weighted average of deltas
        total_weight = sum(weights)
        avg_delta = [
            sum(delta[layer_idx] * w / total_weight for delta, w in zip(client_deltas, weights))
            for layer_idx in range(len(client_deltas[0]))
        ]

        # Add DP noise
        avg_delta = self._add_noise(avg_delta, total_weight)

        # Apply update
        new_global = [gw + d for gw, d in zip(global_weights, avg_delta)]

        #  Save model for next round
        self.global_parameters = new_global

        return ndarrays_to_parameters(new_global), {}

"""Client implementation for FHE-enabled DIWS using TenSEAL."""

from __future__ import annotations

import copy
import os
import pickle
import time
from collections import Counter
from typing import Dict, Optional, Tuple

import torch

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from flwr.common import NDArrays, Scalar

from src.clients.base_client import FlowerClient
from src.clients.subset_client_trainer import get_subset_client_trainer
from src.utils.fhe import get_tenseal, load_client_context


class FheDiwsClient(FlowerClient):
    """Flower client with FHE-enabled DIWS hooks using TenSEAL."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        fhe_cfg = self.config.strategy.get("fhe", {})
        self.client_context_path = fhe_cfg.get("client_context_path", "client_context.pkl")
        self.fhe_metrics_enabled = fhe_cfg.get("metrics_enabled", True)
        self.subset_epochs = int(fhe_cfg.get("subset_epochs", 1))
        self.feasibility_epsilon = float(fhe_cfg.get("feasibility_epsilon", 0.01))
        self.context_load_retries = int(fhe_cfg.get("context_load_retries", 20))
        self.context_load_delay_s = float(fhe_cfg.get("context_load_delay_s", 0.5))
        self.ts = get_tenseal()
        # load private context for encrypt/decrypt (TenSEAL context includes secret key)
        self.context = self._load_context_with_retry(self.client_context_path)
        
        # cache for label distribution (computed once)
        self._label_distribution_cache: Optional[Dict[int, int]] = None
        # cache for encrypted label distribution (to avoid re-encryption)
        self._encrypted_label_dist_cache: Optional[Dict[int, bytes]] = None

    def _get_process_stats(self):
        if not PSUTIL_AVAILABLE:
            return None, None
        proc = psutil.Process(os.getpid())
        cpu_times = proc.cpu_times()
        cpu_total = cpu_times.user + cpu_times.system
        mem_rss_mb = proc.memory_info().rss / (1024**2)
        return cpu_total, mem_rss_mb

    def _start_resource_timer(self):
        start_time = time.perf_counter()
        cpu_start, mem_start = self._get_process_stats()
        return start_time, cpu_start, mem_start

    def _finish_resource_timer(self, prefix: str, start_time, cpu_start, mem_start):
        metrics = {f"{prefix}_time_s": time.perf_counter() - start_time}
        if cpu_start is not None:
            cpu_end, mem_end = self._get_process_stats()
            metrics[f"{prefix}_cpu_time_s"] = max(0.0, cpu_end - cpu_start)
            if mem_end is not None and mem_start is not None:
                metrics[f"{prefix}_mem_rss_mb_delta"] = mem_end - mem_start
        return metrics

    def _load_context_with_retry(self, path: str):
        last_error = None
        for _ in range(self.context_load_retries):
            try:
                return load_client_context(path)
            except FileNotFoundError as exc:
                last_error = exc
                time.sleep(self.context_load_delay_s)
        if last_error:
            raise last_error
        return load_client_context(path)

    def get_label_distribution(self) -> Dict[int, int]:
        """Get label distribution, using cache if available."""
        if self._label_distribution_cache is not None:
            return self._label_distribution_cache

        label_counter = Counter()
        for batch in self.trainloader:
            labels = batch["label"]
            label_counter.update([int(label) for label in labels])

        self._label_distribution_cache = dict(label_counter)
        return self._label_distribution_cache

    def _handle_missing_clients(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        # decrypt target subset distribution using TenSEAL
        encrypted_target = pickle.loads(config["subset_distribution"])
        target_distribution: Dict[int, int] = {}

        dec_metrics: Dict[str, Scalar] = {}
        if self.fhe_metrics_enabled:
            start_time, cpu_start, mem_start = self._start_resource_timer()

        for label, enc_data in encrypted_target.items():
            # Decrypt: Result is a vector, take 0th element
            val = self.ts.ckks_vector_from(self.context, enc_data).decrypt()[0]
            target_distribution[int(label)] = max(0, int(round(val)))

        if self.fhe_metrics_enabled:
            dec_metrics = self._finish_resource_timer(
                "fhe_decrypt_subset", start_time, cpu_start, mem_start
            )

        subset_trainer = get_subset_client_trainer(
            model=copy.deepcopy(self.model),
            subset_distribution=target_distribution,
            trainloader=self.trainloader,
            device=self.device,
            learning_rate=self.learning_rate,
            epochs=self.subset_epochs,
        )
        params, num_examples, metrics = subset_trainer.fit(parameters)
        metrics.update(dec_metrics)
        metrics["subset_target_total"] = int(sum(target_distribution.values()))
        metrics["subset_target_labels"] = int(len(target_distribution))
        return params, num_examples, metrics

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        if config.get("custom_rpc") == "handle_missing_clients":
            return self._handle_missing_clients(parameters, config)

        params, num_examples, metrics = super().fit(parameters, config)
        if num_examples == 0 or metrics.get("disconnected"):
            return params, num_examples, metrics

        if config.get("current_round", 0) == 1:

            # encrypt label distribution on round 1 using TenSEAL
            # use cache if available to avoid re-encryption
            if self._encrypted_label_dist_cache is not None:
                encrypted_dist = self._encrypted_label_dist_cache
                enc_metrics: Dict[str, Scalar] = {"fhe_encrypt_cache_hit": 1.0}
            else:
                label_distribution = self.get_label_distribution()
                encrypted_dist = {}

                enc_metrics: Dict[str, Scalar] = {}
                if self.fhe_metrics_enabled:
                    start_time, cpu_start, mem_start = self._start_resource_timer()

                for label, count in label_distribution.items():
                    encrypted_dist[label] = self.ts.ckks_vector(
                        self.context, [float(count)]
                    ).serialize()

                if self.fhe_metrics_enabled:
                    enc_metrics = self._finish_resource_timer(
                        "fhe_encrypt_label_dist", start_time, cpu_start, mem_start
                    )

                # cache the encrypted distribution
                self._encrypted_label_dist_cache = encrypted_dist

            label_distribution = self.get_label_distribution()

            metrics.update(enc_metrics)
            metrics["label_distribution"] = pickle.dumps(encrypted_dist)
            metrics["partition_id"] = str(self.partition_id)
            metrics["label_dist_items"] = int(len(label_distribution))
            metrics["label_dist_total"] = int(sum(label_distribution.values()))
            metrics["fhe_label_dist_bytes"] = int(sum(len(v) for v in encrypted_dist.values()))

        return params, num_examples, metrics

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        if "blinded_diff" in config:

            # masked Interactive Protocol Check using TenSEAL
            # decrypt masked diffs and return only the sign
            blinded_diff_map = pickle.loads(config["blinded_diff"])
            is_capped_map = {}

            dec_metrics: Dict[str, Scalar] = {}
            if self.fhe_metrics_enabled:
                start_time, cpu_start, mem_start = self._start_resource_timer()

            for label, enc_diff in blinded_diff_map.items():
                # Decrypt: (Fair - Stock) * Mask
                # Mask is positive, so sign is preserved.
                # If > 0: Capped (Fair > Stock)
                # If < 0: Capable (Stock > Fair)
                val = self.ts.ckks_vector_from(self.context, enc_diff).decrypt()[0]
                is_capped_map[label] = val > 0

            if self.fhe_metrics_enabled:
                dec_metrics = self._finish_resource_timer(
                    "fhe_decrypt_blinded", start_time, cpu_start, mem_start
                )

            metrics = {"is_capped": pickle.dumps(is_capped_map)}
            metrics.update(dec_metrics)
            return 0.0, 0, metrics

        if "check_global_feasibility" in config:
            # distributed Target Scaling Check using TenSEAL
            # decrypt masked feasibility checks and return a boolean
            blinded_checks_map = pickle.loads(config["check_global_feasibility"])
            is_feasible = True

            dec_metrics: Dict[str, Scalar] = {}
            if self.fhe_metrics_enabled:
                start_time, cpu_start, mem_start = self._start_resource_timer()

            for _, enc_val in blinded_checks_map.items():
                # Decrypt: Active - (Dropped * k)
                # Masked with positive random value
                val = self.ts.ckks_vector_from(self.context, enc_val).decrypt()[0]
                # Use epsilon for robustness against FHE noise
                if val < -self.feasibility_epsilon:
                    is_feasible = False
                    break

            if self.fhe_metrics_enabled:
                dec_metrics = self._finish_resource_timer(
                    "fhe_decrypt_feasibility", start_time, cpu_start, mem_start
                )

            metrics = {"is_feasible": is_feasible}
            metrics.update(dec_metrics)
            return 0.0, 0, metrics

        return super().evaluate(parameters, config)

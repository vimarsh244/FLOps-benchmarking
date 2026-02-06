"""FHE-enabled Distribution Informed Weight Substitution (DIWS) strategy."""

from __future__ import annotations

import os
import pickle
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Union

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
from flwr.server.strategy.aggregate import aggregate_inplace
from flwr.server.strategy.strategy import Strategy

from src.strategies.fedavg import CustomFedAvg
from src.utils.fhe import create_and_save_context, get_tenseal, load_context
from src.utils.wandb_logger import log_metrics


class DIWSFHE(Strategy):
    """Distribution Informed Weight Substitution with FHE."""

    def __init__(
        self,
        *,
        aggregator_strategy: Optional[Strategy] = None,
        substitution_timeout: float = 600.0,
        server_context_path: str = "server_context.pkl",
        client_context_path: str = "client_context.pkl",
        poly_modulus_degree: int = 16384,
        coeff_mod_bit_sizes: Optional[List[int]] = None,
        global_scale_bits: int = 29,
        binary_search_iterations: int = 5,
        mask_range: Tuple[float, float] = (10.0, 100.0),
        max_protocol_iters: int = 3,
        feasibility_epsilon: float = 0.01,
    ) -> None:
        super().__init__()
        self.aggregator_strategy = aggregator_strategy or CustomFedAvg()
        self.substitution_timeout = substitution_timeout
        
        # Resolve context paths to absolute paths (will be resolved later with full config)
        self._server_context_path_template = server_context_path
        self._client_context_path_template = client_context_path
        self.server_context_path = server_context_path
        self.client_context_path = client_context_path
        
        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes
        self.global_scale_bits = global_scale_bits
        self.binary_search_iterations = binary_search_iterations
        self.mask_range = mask_range
        self.max_protocol_iters = max_protocol_iters
        self.feasibility_epsilon = feasibility_epsilon

        self.global_parameters: Optional[Parameters] = None
        self.label_distribution: Dict[str, Dict[int, object]] = {}
        self.context = None
        self.ts = None
        self.cid_to_partition: Dict[str, str] = {}
        self.inv_num_cache: Dict[int, object] = {}

        # computation cache for substitution shares keyed by (active_cids, dropped_pids)
        self.computation_cache: Dict[tuple, Dict[str, Dict[int, object]]] = {}
        # cache for zero encrypted vector to avoid repeated encryption
        self._zero_enc_cache: Optional[object] = None

    def __repr__(self) -> str:
        return repr(self.aggregator_strategy)

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        self.global_parameters = self.aggregator_strategy.initialize_parameters(client_manager)
        if not self.ts:
            self.ts = get_tenseal()
        
        # Resolve context paths to absolute paths if they are relative
        self.server_context_path = os.path.abspath(self._server_context_path_template)
        self.client_context_path = os.path.abspath(self._client_context_path_template)
        
        if self.context is None:
            try:
                self.context = load_context(self.server_context_path)
                print(f"[DIWS-FHE] Loaded existing server context from: {self.server_context_path}")
            except FileNotFoundError:
                # create public/server and private/client contexts
                print(f"[DIWS-FHE] Creating new FHE contexts:")
                print(f"  Server context: {self.server_context_path}")
                print(f"  Client context: {self.client_context_path}")
                create_and_save_context(
                    server_path=self.server_context_path,
                    client_path=self.client_context_path,
                    poly_modulus_degree=self.poly_modulus_degree,
                    coeff_mod_bit_sizes=self.coeff_mod_bit_sizes,
                    global_scale_bits=self.global_scale_bits,
                )
                self.context = load_context(self.server_context_path)
                print(f"[DIWS-FHE] Context files created successfully")
                print(f"[DIWS-FHE] IMPORTANT: For distributed/hardware mode, copy client_context.pkl to all client machines at the same path!")
        self.cid_to_partition = {}
        self.inv_num_cache = {}
        self.computation_cache = {}
        self._zero_enc_cache = None
        return self.global_parameters

    def _get_zero_enc(self) -> object:
        """Return cached zero encrypted vector to avoid repeated encryption."""
        if self._zero_enc_cache is None:
            self._zero_enc_cache = self.ts.ckks_vector(self.context, [0.0])
        return self._zero_enc_cache.copy()

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return self.aggregator_strategy.evaluate(server_round, parameters)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        self.global_parameters = parameters
        return self.aggregator_strategy.configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        return self.aggregator_strategy.configure_evaluate(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        dropped_cids = set()
        valid_results: List[Tuple[ClientProxy, FitRes]] = []
        for client_proxy, fit_res in results:
            if self._is_dropped_fitres(fit_res):
                dropped_cids.add(client_proxy.cid)
            else:
                valid_results.append((client_proxy, fit_res))

        for failure in failures:
            if isinstance(failure, tuple) and failure:
                client_proxy = failure[0]
                if isinstance(client_proxy, ClientProxy):
                    dropped_cids.add(client_proxy.cid)

        results = valid_results

        if server_round == 1:
            # capture encrypted label distributions on round 1
            for client_proxy, fit_res in results:
                if not fit_res.metrics or "label_distribution" not in fit_res.metrics:
                    continue
                client_dist_bytes = pickle.loads(fit_res.metrics.get("label_distribution"))
                client_dist = {
                    label: self.ts.ckks_vector_from(self.context, enc_bytes)
                    for label, enc_bytes in client_dist_bytes.items()
                }
                pid = fit_res.metrics.get("partition_id")
                if pid is not None:
                    pid_str = str(pid)
                    self.cid_to_partition[client_proxy.cid] = pid_str
                    self.label_distribution[pid_str] = client_dist
                else:
                    self.label_distribution[client_proxy.cid] = client_dist

        if dropped_cids:
            dropped_pids = [str(self.cid_to_partition.get(cid, cid)) for cid in dropped_cids]
            self.substitute_dropped_clients(
                server_round=server_round,
                results=results,
                dropped_partitions=dropped_pids,
            )

        # log per-client metrics for overhead tracking
        for client_proxy, fit_res in results:
            if fit_res.metrics:
                self._log_client_metrics(client_proxy.cid, fit_res.metrics, server_round)

        return self.aggregator_strategy.aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        return self.aggregator_strategy.aggregate_evaluate(server_round, results, failures)

    def substitute_dropped_clients(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        dropped_partitions: List[str],
    ) -> None:
        if not dropped_partitions:
            return

        active_client_proxies = [res[0] for res in results]
        if not active_client_proxies:
            return
        active_cids = [p.cid for p in active_client_proxies]
        active_pids = [str(self.cid_to_partition.get(cid, cid)) for cid in active_cids]
        actual_dropped = [pid for pid in dropped_partitions if pid not in active_pids]
        if not actual_dropped:
            return

        # generate cache key based on active and dropped partitions
        cache_key = (tuple(sorted(active_cids)), tuple(sorted(actual_dropped)))

        if cache_key in self.computation_cache:
            # cache hit - reuse previously computed shares
            final_shares = self.computation_cache[cache_key]
            log_metrics(
                {"diws_fhe/cache_hit": 1.0},
                step=server_round,
            )
        else:
            # cache miss - compute substitution shares
            final_shares = self._compute_substitution_shares(
                active_cids=active_cids,
                active_client_proxies=active_client_proxies,
                actual_dropped=actual_dropped,
                server_round=server_round,
            )
            self.computation_cache[cache_key] = final_shares
            log_metrics(
                {"diws_fhe/cache_miss": 1.0},
                step=server_round,
            )

        # trigger subset training with encrypted shares
        self._trigger_subset_training(
            active_client_proxies=active_client_proxies,
            final_shares=final_shares,
            server_round=server_round,
            results=results,
            actual_dropped=actual_dropped,
            active_cids=active_cids,
        )

    def _compute_substitution_shares(
        self,
        active_cids: List[str],
        active_client_proxies: List[ClientProxy],
        actual_dropped: List[str],
        server_round: int,
    ) -> Dict[str, Dict[int, object]]:
        """Compute substitution shares for dropped clients using FHE.

        Returns:
            Dictionary mapping client IDs to their share allocations per label.
        """
        # sum encrypted counts from dropped clients
        dropped_demand: Dict[int, object] = {}
        for pid in actual_dropped:
            dist = self.label_distribution.get(pid, {})
            for label, count_enc in dist.items():
                if label not in dropped_demand:
                    # copying to prevent in-place modulus switching degradation
                    dropped_demand[label] = count_enc.copy()
                else:
                    dropped_demand[label] += count_enc

        if not dropped_demand:
            return {cid: {} for cid in active_cids}

        # sum encrypted counts for active clients
        active_stock: Dict[int, object] = {}
        for cid in active_cids:
            cid_key = self.cid_to_partition.get(cid, cid)
            dist = self.label_distribution.get(str(cid_key), {})
            if not dist:
                dist = self.label_distribution.get(cid, {})
            for label, count_enc in dist.items():
                if label not in active_stock:
                    # copying to prevent in-place modulus switching degradation
                    active_stock[label] = count_enc.copy()
                else:
                    active_stock[label] += count_enc

        # distributed Target Scaling via Blind Binary Search
        final_k = self._binary_search_scaling_factor(
            active_client_proxies=active_client_proxies,
            active_stock=active_stock,
            dropped_demand=dropped_demand,
            server_round=server_round,
        )

        # apply scaling to dropped demand
        k_final_enc = self.ts.ckks_vector(self.context, [final_k])
        scaled_dropped_demand = {
            label: val.copy() * k_final_enc for label, val in dropped_demand.items()
        }

        # run masked interactive protocol to distribute shares
        final_shares = self._run_masked_interactive_protocol(
            active_cids=active_cids,
            active_client_proxies=active_client_proxies,
            dropped_demand=scaled_dropped_demand,
            server_round=server_round,
        )

        return final_shares

    def _binary_search_scaling_factor(
        self,
        active_client_proxies: List[ClientProxy],
        active_stock: Dict[int, object],
        dropped_demand: Dict[int, object],
        server_round: int,
    ) -> float:
        """Find optimal scaling factor k using blind binary search.

        Returns:
            Scaling factor k in [0, 1] such that active_stock >= k * dropped_demand.
        """
        helper_proxy = active_client_proxies[0]
        k_min = 0.0
        k_max = 1.0
        feasibility_metrics: List[Dict[str, Scalar]] = []
        labels_to_check = list(dropped_demand.keys())

        for _ in range(self.binary_search_iterations):
            k_mid = (k_min + k_max) / 2.0
            blinded_checks = {}

            # encrypt k_mid and mask once per iteration (optimization)
            k_enc = self.ts.ckks_vector(self.context, [k_mid])
            mask_val = random.uniform(*self.mask_range)
            mask_enc = self.ts.ckks_vector(self.context, [mask_val])

            for label in labels_to_check:
                # use cached zero for missing labels
                stock_total = active_stock.get(label)
                if stock_total is None:
                    stock_total = self._get_zero_enc()
                else:
                    stock_total = stock_total.copy()  # Prevent degradation

                dropped_total = dropped_demand[label].copy()  # Prevent degradation
                # Compute: Active - (Dropped * k), then blind with mask
                diff = stock_total - (dropped_total * k_enc)
                blinded = diff * mask_enc
                blinded_checks[label] = blinded.serialize()

            ins = EvaluateIns(
                parameters=self.global_parameters,
                config={"check_global_feasibility": pickle.dumps(blinded_checks)},
            )

            res = None
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        helper_proxy.evaluate,
                        ins,
                        self.substitution_timeout,
                        server_round,
                    )
                    res = future.result()
            except Exception:
                k_min = k_mid
                break

            if res and res.metrics:
                feasibility_metrics.append(
                    {k: v for k, v in res.metrics.items() if isinstance(v, (int, float))}
                )
                self._log_client_metrics(helper_proxy.cid, res.metrics, server_round)

            is_feasible = bool(res.metrics.get("is_feasible", False)) if res else False
            if is_feasible:
                k_min = k_mid
            else:
                k_max = k_mid

        if feasibility_metrics:
            self._log_metric_average(
                feasibility_metrics,
                step=server_round,
                prefix="fhe/feasibility",
            )

        return k_min

    def _run_masked_interactive_protocol(
        self,
        active_cids: List[str],
        active_client_proxies: List[ClientProxy],
        dropped_demand: Dict[int, object],
        server_round: int,
    ) -> Dict[str, Dict[int, object]]:
        """Run masked interactive protocol to distribute shares among active clients.

        Returns:
            Dictionary mapping client IDs to their share allocations per label.
        """
        final_shares: Dict[str, Dict[int, object]] = {cid: {} for cid in active_cids}
        remaining_demand = {label: val.copy() for label, val in dropped_demand.items()}
        all_labels = set(remaining_demand.keys())
        active_set = {label: list(active_client_proxies) for label in all_labels}
        client_proxy_map = {p.cid: p for p in active_client_proxies}

        for i_loop in range(self.max_protocol_iters):
            blinded_checks_per_client: Dict[str, Dict[int, bytes]] = {
                cid: {} for cid in active_cids
            }
            has_checks = False

            for label in all_labels:
                if not active_set[label]:
                    continue

                target = remaining_demand[label]
                num_active = len(active_set[label])

                # cache inverse of num_active to avoid repeated encryption
                if num_active not in self.inv_num_cache:
                    self.inv_num_cache[num_active] = self.ts.ckks_vector(
                        self.context, [1.0 / num_active]
                    )
                inv_num_enc = self.inv_num_cache[num_active]
                fair_share = target.copy() * inv_num_enc

                for client in active_set[label]:
                    cid = client.cid
                    cid_key = self.cid_to_partition.get(cid, cid)
                    dist = self.label_distribution.get(str(cid_key), {})
                    if not dist:
                        dist = self.label_distribution.get(cid, {})

                    stock_enc = dist.get(label)
                    if stock_enc is None:
                        stock_enc = self._get_zero_enc()
                    else:
                        # copying to prevent in-place modulus switching degradation
                        stock_enc = stock_enc.copy()

                    # Generate random mask for blinding
                    mask_val = random.uniform(*self.mask_range)
                    mask_enc = self.ts.ckks_vector(self.context, [mask_val])
                    blinded_diff = (fair_share - stock_enc) * mask_enc
                    blinded_checks_per_client[cid][label] = blinded_diff.serialize()
                    has_checks = True

            if not has_checks:
                break

            # Per-client masked feasibility checks (parallel)
            iteration_metrics: List[Dict[str, Scalar]] = []
            with ThreadPoolExecutor() as executor:
                futures = {}
                for cid, checks in blinded_checks_per_client.items():
                    if not checks:
                        continue
                    ins = EvaluateIns(
                        parameters=self.global_parameters,
                        config={"blinded_diff": pickle.dumps(checks)},
                    )
                    futures[cid] = executor.submit(
                        client_proxy_map[cid].evaluate,
                        ins,
                        self.substitution_timeout,
                        server_round,
                    )

                for cid, future in futures.items():
                    try:
                        res = future.result()
                    except Exception:
                        continue

                    if res.metrics:
                        iteration_metrics.append(
                            {
                                k: v
                                for k, v in res.metrics.items()
                                if isinstance(v, (int, float))
                            }
                        )
                        self._log_client_metrics(cid, res.metrics, server_round)

                    is_capped_map = pickle.loads(
                        res.metrics.get("is_capped", pickle.dumps({}))
                    )
                    for label, is_capped in is_capped_map.items():
                        if is_capped:
                            # client is capped - assign stock as final share
                            cid_key = self.cid_to_partition.get(cid, cid)
                            dist = self.label_distribution.get(str(cid_key), {})
                            if not dist:
                                dist = self.label_distribution.get(cid, {})
                            stock = dist.get(label)
                            if stock is None:
                                stock = self._get_zero_enc()
                            else:
                                stock = stock.copy()
                            final_shares[cid][label] = stock
                            # subtract stock from remaining demand
                            remaining_demand[label] = remaining_demand[label] - stock
                            # remove from active set
                            proxy = client_proxy_map[cid]
                            if proxy in active_set[label]:
                                active_set[label].remove(proxy)

            if iteration_metrics:
                self._log_metric_average(
                    iteration_metrics,
                    step=server_round,
                    prefix=f"fhe/blinded_iter_{i_loop + 1}",
                )

        # Final assignment for remaining active clients
        for label in all_labels:
            active_clients = active_set[label]
            if not active_clients:
                continue
            target = remaining_demand[label]
            num_active = len(active_clients)
            if num_active not in self.inv_num_cache:
                self.inv_num_cache[num_active] = self.ts.ckks_vector(
                    self.context, [1.0 / num_active]
                )
            fair_share = target.copy() * self.inv_num_cache[num_active]
            for client in active_clients:
                final_shares[client.cid][label] = fair_share

        return final_shares

    def _trigger_subset_training(
        self,
        active_client_proxies: List[ClientProxy],
        final_shares: Dict[str, Dict[int, object]],
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        actual_dropped: List[str],
        active_cids: List[str],
    ) -> None:
        """Trigger subset training on active clients with computed shares."""
        all_labels = set()
        for shares in final_shares.values():
            all_labels.update(shares.keys())

        with ThreadPoolExecutor() as executor:
            futures = {}
            for client_proxy in active_client_proxies:
                cid = client_proxy.cid
                share_map = final_shares.get(cid, {})
                serialized_shares = {l: v.serialize() for l, v in share_map.items()}
                config = {
                    "subset_distribution": pickle.dumps(serialized_shares),
                    "custom_rpc": "handle_missing_clients",
                }
                fit_ins = FitIns(parameters=self.global_parameters, config=config)
                futures[cid] = executor.submit(
                    client_proxy.fit,
                    fit_ins,
                    self.substitution_timeout,
                    server_round,
                )
            outputs = []
            subset_metrics = []
            for cid, future in futures.items():
                res = future.result()
                outputs.append(res)
                if res.metrics:
                    subset_metrics.append(
                        {k: v for k, v in res.metrics.items() if isinstance(v, (int, float))}
                    )
                    self._log_client_metrics(cid, res.metrics, server_round)

        substituted_parameters_fitres = self.aggregate_substitution_parameters(outputs)
        results.append((None, substituted_parameters_fitres))

        log_metrics(
            {
                "diws_fhe/dropped_clients": float(len(actual_dropped)),
                "diws_fhe/active_clients": float(len(active_cids)),
                "diws_fhe/labels": float(len(all_labels)),
            },
            step=server_round,
        )
        if subset_metrics:
            self._log_metric_average(subset_metrics, step=server_round, prefix="fhe/subset")

    def aggregate_substitution_parameters(self, results: List[FitRes]) -> FitRes:
        results_with_proxies = [(None, fit_res) for fit_res in results]
        aggregated_parameters = aggregate_inplace(results_with_proxies)
        total_samples = sum(fit_res.num_examples for _, fit_res in results_with_proxies)

        return FitRes(
            parameters=ndarrays_to_parameters(aggregated_parameters),
            num_examples=total_samples,
            metrics={},
            status=None,
        )

    @staticmethod
    def _is_dropped_fitres(fit_res: FitRes) -> bool:
        if fit_res.num_examples == 0:
            return True
        if fit_res.metrics and (
            fit_res.metrics.get("disconnected") or fit_res.metrics.get("is_dropped")
        ):
            return True
        try:
            params = parameters_to_ndarrays(fit_res.parameters)
            if len(params) == 0:
                return True
        except Exception:
            return True
        return False

    @staticmethod
    def _log_metric_average(
        metrics_list: List[Dict[str, Scalar]],
        step: int,
        prefix: str,
    ) -> None:
        totals: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for metrics in metrics_list:
            for key, val in metrics.items():
                if not isinstance(val, (int, float)):
                    continue
                totals[key] = totals.get(key, 0.0) + float(val)
                counts[key] = counts.get(key, 0) + 1
        if not totals:
            return
        averaged = {k: totals[k] / counts[k] for k in totals}
        log_metrics(averaged, step=step, prefix=prefix)

    @staticmethod
    def _log_client_metrics(client_id: str, metrics: Dict[str, Scalar], step: int) -> None:
        numeric = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        if numeric:
            log_metrics(numeric, step=step, prefix=f"client/{client_id}")

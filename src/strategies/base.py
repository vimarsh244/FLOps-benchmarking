"""Base strategy utilities and common functions."""

from typing import List, Tuple, Dict, Optional, Callable, Union
import numpy as np

from flwr.common import (
    FitRes,
    EvaluateRes,
    Metrics,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Compute weighted average of metrics across clients.
    
    Args:
        metrics: list of (num_examples, metrics_dict) tuples
    
    Returns:
        Aggregated metrics dictionary
    """
    if not metrics:
        return {}
    
    # get all metric keys from first entry
    all_keys = set()
    for _, m in metrics:
        all_keys.update(m.keys())
    
    # compute weighted average for each metric
    aggregated = {}
    total_examples = sum(n for n, _ in metrics)
    
    if total_examples == 0:
        return {}
    
    for key in all_keys:
        weighted_sum = sum(
            n * m.get(key, 0.0) 
            for n, m in metrics 
            if isinstance(m.get(key), (int, float))
        )
        aggregated[key] = weighted_sum / total_examples
    
    return aggregated


def get_parameters_from_results(
    results: List[Tuple[ClientProxy, FitRes]]
) -> List[Tuple[NDArrays, int]]:
    """Extract parameters and num_examples from fit results.
    
    Args:
        results: list of (client_proxy, fit_result) tuples
    
    Returns:
        List of (parameters, num_examples) tuples
    """
    return [
        (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
        for _, fit_res in results
    ]


def aggregate_parameters(
    weights_results: List[Tuple[NDArrays, int]],
    inplace: bool = True,
) -> NDArrays:
    """Aggregate parameters using weighted average.
    
    Args:
        weights_results: list of (parameters, num_examples) tuples
        inplace: whether to use in-place aggregation
    
    Returns:
        Aggregated parameters
    """
    if not weights_results:
        raise ValueError("No results to aggregate")
    
    total_examples = sum(n for _, n in weights_results)
    
    if total_examples == 0:
        return weights_results[0][0]
    
    # weighted average
    aggregated = []
    for i in range(len(weights_results[0][0])):
        layer_sum = np.zeros_like(weights_results[0][0][i], dtype=np.float32)
        for weights, n in weights_results:
            layer_sum += (n / total_examples) * np.asarray(weights[i], dtype=np.float32)
        aggregated.append(layer_sum)
    
    return aggregated


def zeros_like(weights: NDArrays) -> NDArrays:
    """Create zero arrays with same shapes as weights."""
    return [np.zeros_like(w, dtype=np.float32) for w in weights]


def add_inplace(dst: NDArrays, src: NDArrays, alpha: float = 1.0) -> None:
    """Add src to dst in-place: dst += alpha * src."""
    for i in range(len(dst)):
        dst[i] = dst[i] + np.float32(alpha) * np.asarray(src[i], dtype=np.float32)


def copy_weights(weights: NDArrays) -> NDArrays:
    """Create a deep copy of weights."""
    return [np.asarray(w, dtype=np.float32).copy() for w in weights]


def compute_update(
    new_weights: NDArrays,
    old_weights: NDArrays,
) -> NDArrays:
    """Compute the difference (update) between new and old weights."""
    return [
        np.asarray(new, dtype=np.float32) - np.asarray(old, dtype=np.float32)
        for new, old in zip(new_weights, old_weights)
    ]


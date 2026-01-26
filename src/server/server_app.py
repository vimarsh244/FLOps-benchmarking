"""Flower server implementation for federated learning."""

from typing import Dict, List, Tuple, Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn

from flwr.common import (
    Context,
    Metrics,
    Parameters,
    ndarrays_to_parameters,
)
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import Strategy

from omegaconf import DictConfig

from src.strategies.fedavg import CustomFedAvg
from src.strategies.fedprox import CustomFedProx
from src.strategies.mifa import CustomMIFA
from src.strategies.clusteredfl import CustomClusteredFL
from src.strategies.scaffold import SCAFFOLD
from src.strategies.diws import DIWS
from src.strategies.diws_fhe import DIWSFHE
from src.strategies.fdms import FDMS
# use Flower's built-in adaptive optimizers - they're well-tested
from flwr.server.strategy import FedAdam, FedYogi, FedAdagrad


# strategy registry
STRATEGY_REGISTRY: Dict[str, type] = {
    "fedavg": CustomFedAvg,
    "fedprox": CustomFedProx,
    "mifa": CustomMIFA,
    "clusteredfl": CustomClusteredFL,
    "scaffold": SCAFFOLD,
    "diws": DIWS,
    "diws_fhe": DIWSFHE,
    "fdms": FDMS,
    # use Flower's built-in FedOpt strategies
    "fedadam": FedAdam,
    "fedyogi": FedYogi,
    "fedadagrad": FedAdagrad,
}


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Compute weighted average of metrics across clients.
    
    Args:
        metrics: List of (num_examples, metrics_dict) tuples
    
    Returns:
        Aggregated metrics dictionary
    """
    if not metrics:
        return {}
    
    # get all metric keys
    all_keys = set()
    for _, m in metrics:
        all_keys.update(m.keys())
    
    # compute weighted average
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


def get_initial_parameters(
    model: nn.Module,
) -> Parameters:
    """Get initial parameters from a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Flower Parameters object
    """
    weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
    return ndarrays_to_parameters(weights)


def create_strategy(
    config: DictConfig,
    initial_parameters: Optional[Parameters] = None,
    evaluate_fn: Optional[Callable] = None,
) -> Strategy:
    """Create a federated learning strategy from configuration.
    
    Args:
        config: Hydra configuration
        initial_parameters: Optional initial model parameters
        evaluate_fn: Optional server-side evaluation function
    
    Returns:
        Configured strategy
    
    Raises:
        ValueError: If strategy name is not in registry
    """
    strategy_name = config.strategy.name.lower()
    
    if strategy_name not in STRATEGY_REGISTRY:
        available = ", ".join(STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {available}")
    
    strategy_class = STRATEGY_REGISTRY[strategy_name]
    
    # common parameters for all strategies
    common_params = {
        "fraction_fit": config.server.fraction_fit,
        "fraction_evaluate": config.server.fraction_evaluate,
        "min_fit_clients": config.server.min_fit_clients,
        "min_evaluate_clients": config.server.min_evaluate_clients,
        "min_available_clients": config.server.min_available_clients,
        "accept_failures": config.server.accept_failures,
        "initial_parameters": initial_parameters,
        "evaluate_fn": evaluate_fn,
        "fit_metrics_aggregation_fn": weighted_average,
        "evaluate_metrics_aggregation_fn": weighted_average,
    }
    
    # strategy-specific parameters
    strategy_params = {}
    
    if strategy_name == "fedavg":
        strategy_params["inplace"] = config.strategy.get("inplace", True)
    
    elif strategy_name == "fedprox":
        strategy_params["proximal_mu"] = config.strategy.get("proximal_mu", 0.1)
    
    elif strategy_name == "mifa":
        strategy_params["base_server_lr"] = config.strategy.get("base_server_lr", 1.0)
        strategy_params["client_lr"] = config.client.get("learning_rate", 0.01)
        strategy_params["local_epochs"] = config.client.get("local_epochs", 1)
        strategy_params["batch_size"] = config.client.get("batch_size", 32)
        strategy_params["wait_for_all_clients_init"] = config.strategy.get(
            "wait_for_all_clients_init", True
        )
    
    elif strategy_name == "clusteredfl":
        strategy_params["cosine_similarity_threshold"] = config.strategy.get(
            "cosine_similarity_threshold", 0.7
        )
        strategy_params["min_cosine_similarity"] = config.strategy.get(
            "min_cosine_similarity", 0.3
        )
        strategy_params["split_warmup_rounds"] = config.strategy.get(
            "split_warmup_rounds", 5
        )
        strategy_params["split_cooldown_rounds"] = config.strategy.get(
            "split_cooldown_rounds", 3
        )
        strategy_params["min_clients_for_split"] = config.strategy.get(
            "min_clients_for_split", 3
        )
        strategy_params["min_cluster_size"] = config.strategy.get("min_cluster_size", 2)
    
    elif strategy_name == "scaffold":
        strategy_params["server_lr"] = config.strategy.get("server_lr", 1.0)
        strategy_params["client_lr"] = config.strategy.get("client_lr", 0.01)
        strategy_params["warm_start"] = config.strategy.get("warm_start", False)
    
    elif strategy_name == "fedadam":
        # Flower's FedAdam uses 'eta' for server learning rate
        strategy_params["eta"] = config.strategy.get("eta", config.strategy.get("server_lr", 0.1))
        strategy_params["beta_1"] = config.strategy.get("beta_1", 0.9)
        strategy_params["beta_2"] = config.strategy.get("beta_2", 0.99)
        strategy_params["tau"] = config.strategy.get("tau", 1e-9)
    
    elif strategy_name == "fedyogi":
        # Flower's FedYogi uses 'eta' for server learning rate
        strategy_params["eta"] = config.strategy.get("eta", config.strategy.get("server_lr", 0.01))
        strategy_params["beta_1"] = config.strategy.get("beta_1", 0.9)
        strategy_params["beta_2"] = config.strategy.get("beta_2", 0.99)
        strategy_params["tau"] = config.strategy.get("tau", 1e-3)
    
    elif strategy_name == "fedadagrad":
        # Flower's FedAdagrad uses 'eta' for server learning rate
        strategy_params["eta"] = config.strategy.get("eta", config.strategy.get("server_lr", 0.1))
        strategy_params["tau"] = config.strategy.get("tau", 1e-9)
    
    if strategy_name == "diws":
        base_strategy = CustomFedAvg(
            **common_params,
            inplace=config.strategy.get("inplace", True),
        )
        return DIWS(
            aggregator_strategy=base_strategy,
            substitution_timeout=config.strategy.get("substitution_timeout", 600.0),
        )

    if strategy_name == "diws_fhe":
        base_strategy = CustomFedAvg(
            **common_params,
            inplace=config.strategy.get("inplace", True),
        )
        fhe_cfg = config.strategy.get("fhe", {})
        coeff_mod_bit_sizes = fhe_cfg.get("coeff_mod_bit_sizes")
        if coeff_mod_bit_sizes is not None:
            coeff_mod_bit_sizes = list(coeff_mod_bit_sizes)
        return DIWSFHE(
            aggregator_strategy=base_strategy,
            substitution_timeout=config.strategy.get("substitution_timeout", 600.0),
            server_context_path=fhe_cfg.get("server_context_path", "server_context.pkl"),
            client_context_path=fhe_cfg.get("client_context_path", "client_context.pkl"),
            poly_modulus_degree=fhe_cfg.get("poly_modulus_degree", 16384),
            coeff_mod_bit_sizes=coeff_mod_bit_sizes,
            global_scale_bits=fhe_cfg.get("global_scale_bits", 29),
            binary_search_iterations=fhe_cfg.get("binary_search_iterations", 5),
            mask_range=tuple(fhe_cfg.get("mask_range", [10.0, 100.0])),
            max_protocol_iters=fhe_cfg.get("max_protocol_iters", 3),
            feasibility_epsilon=fhe_cfg.get("feasibility_epsilon", 0.01),
        )

    if strategy_name == "fdms":
        base_strategy = CustomFedAvg(
            **common_params,
            inplace=config.strategy.get("inplace", True),
        )
        return FDMS(
            aggregator_strategy=base_strategy,
            min_common_rounds=config.strategy.get("min_common_rounds", 1),
            fallback_mode=config.strategy.get("fallback_mode", "mean"),
            similarity_eps=config.strategy.get("similarity_eps", 1e-12),
        )

    return strategy_class(**common_params, **strategy_params)


def create_server_fn(config: DictConfig):
    """Create a server function for Flower simulation.
    
    Args:
        config: Hydra configuration
    
    Returns:
        Server function for Flower ServerApp
    """
    from src.models.registry import get_model_from_config
    
    def server_fn(context: Context) -> ServerAppComponents:
        # create model for initial parameters
        model = get_model_from_config(config.model, config.dataset)
        initial_parameters = get_initial_parameters(model)
        
        # create strategy
        strategy = create_strategy(
            config=config,
            initial_parameters=initial_parameters,
        )
        
        # server config
        num_rounds = config.server.num_rounds
        server_config = ServerConfig(num_rounds=num_rounds)
        
        return ServerAppComponents(strategy=strategy, config=server_config)
    
    return server_fn


def create_server_app(config: DictConfig) -> ServerApp:
    """Create a Flower ServerApp from configuration.
    
    Args:
        config: Hydra configuration
    
    Returns:
        Configured ServerApp
    """
    server_fn = create_server_fn(config)
    return ServerApp(server_fn=server_fn)


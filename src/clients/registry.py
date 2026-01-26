"""Client registry for Hydra-based client type selection.

This module provides a centralized way to select the appropriate client
implementation based on the strategy being used. Different strategies
(e.g., SCAFFOLD) require different client-side logic.
"""

from typing import Any, Callable

from omegaconf import DictConfig

from src.clients.base_client import FlowerClient
from src.clients.personalized_client import PersonalizedClient
from src.clients.scaffold_client import ScaffoldClient

# mapping of client types to their classes
CLIENT_REGISTRY = {
    "base": FlowerClient,
    "default": FlowerClient,
    "fedavg": FlowerClient,
    "fedprox": FlowerClient,
    "fedopt": FlowerClient,
    "fedadam": FlowerClient,
    "fedyogi": FlowerClient,
    "fedadagrad": FlowerClient,
    "mifa": FlowerClient,
    "clusteredfl": FlowerClient,
    "scaffold": ScaffoldClient,
    "fedper": PersonalizedClient,
    "personalized": PersonalizedClient,
}


def get_client_class(client_type: str):
    """Get the client class for a given client type.

    Args:
        client_type: The type of client to use (e.g., "base", "scaffold")

    Returns:
        The client class to instantiate

    Raises:
        ValueError: If the client type is not recognized
    """
    client_type_lower = client_type.lower()

    if client_type_lower not in CLIENT_REGISTRY:
        available = ", ".join(sorted(CLIENT_REGISTRY.keys()))
        raise ValueError(f"Unknown client type: {client_type}. Available types: {available}")

    return CLIENT_REGISTRY[client_type_lower]


def get_client_type_for_strategy(strategy_name: str) -> str:
    """Determine the appropriate client type based on strategy.

    Args:
        strategy_name: The name of the FL strategy being used

    Returns:
        The client type string to use
    """
    strategy_lower = strategy_name.lower()

    # scaffold requires its own client
    if "scaffold" in strategy_lower:
        return "scaffold"

    # fedper requires personalized client
    if "fedper" in strategy_lower:
        return "personalized"

    # all other strategies use the base client
    return "base"


def create_client_fn(config: DictConfig) -> Callable:
    """Create a client function for Flower simulation.

    This function selects the appropriate client implementation based on
    the configuration. It checks:
    1. Explicit client.client_type in config
    2. Infers from strategy name if not specified

    Args:
        config: Hydra configuration

    Returns:
        Client function for Flower ClientApp
    """
    from flwr.common import Context

    from src.datasets.loader import load_data
    from src.models.registry import get_model_from_config
    from src.scenarios.registry import get_scenario

    # determine client type
    # check if explicitly set in config
    client_type = config.client.get("client_type", None)

    # if not set, infer from strategy
    if client_type is None:
        strategy_name = config.strategy.get("name", "fedavg")
        client_type = get_client_type_for_strategy(strategy_name)

    # get the client class
    ClientClass = get_client_class(client_type)

    # create scenario handler
    scenario = get_scenario(config.scenario)

    def client_fn(context: Context):
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]

        # load data
        trainloader, valloader = load_data(
            partition_id=partition_id,
            num_partitions=num_partitions,
            dataset_cfg=config.dataset,
            partitioner_cfg=config.partitioner,
            batch_size=config.client.batch_size,
            test_fraction=config.evaluation.test_fraction,
        )

        # create model
        model = get_model_from_config(config.model, config.dataset)

        # create client using the selected class
        client = ClientClass(
            model=model,
            trainloader=trainloader,
            valloader=valloader,
            partition_id=partition_id,
            config=config,
            scenario_handler=scenario,
            context=context, 
        )

        return client.to_client()

    return client_fn

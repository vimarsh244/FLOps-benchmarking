"""Federated learning strategy implementations."""

from src.strategies.fedavg import CustomFedAvg
from src.strategies.fedprox import CustomFedProx
from src.strategies.mifa import CustomMIFA
from src.strategies.clusteredfl import CustomClusteredFL
from src.strategies.scaffold import SCAFFOLD
# use Flower's built-in FedOpt strategies
from flwr.server.strategy import FedAdam, FedYogi, FedAdagrad

__all__ = [
    "CustomFedAvg",
    "CustomFedProx",
    "CustomMIFA",
    "CustomClusteredFL",
    "SCAFFOLD",
    "FedAdam",
    "FedYogi",
    "FedAdagrad",
]


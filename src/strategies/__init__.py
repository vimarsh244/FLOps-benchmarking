"""Federated learning strategy implementations."""

from src.strategies.fedavg import CustomFedAvg
from src.strategies.fedprox import CustomFedProx
from src.strategies.mifa import CustomMIFA
from src.strategies.clusteredfl import CustomClusteredFL
from src.strategies.scaffold import SCAFFOLD
from src.strategies.diws import DIWS
from src.strategies.diws_fhe import DIWSFHE
from src.strategies.fdms import FDMS

# use Flower's built-in FedOpt strategies
from flwr.server.strategy import FedAdam, FedYogi, FedAdagrad

__all__ = [
    "CustomFedAvg",
    "CustomFedProx",
    "CustomMIFA",
    "CustomClusteredFL",
    "SCAFFOLD",
    "DIWS",
    "DIWSFHE",
    "FDMS",
    "FedAdam",
    "FedYogi",
    "FedAdagrad",
]

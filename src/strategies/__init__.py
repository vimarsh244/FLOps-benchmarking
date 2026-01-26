"""Federated learning strategy implementations."""

# use Flower's built-in FedOpt strategies
from flwr.server.strategy import FedAdagrad, FedAdam, FedYogi

from src.strategies.clusteredfl import CustomClusteredFL
from src.strategies.ditto import Ditto
from src.strategies.diws import DIWS
from src.strategies.fdms import FDMS
from src.strategies.fedavg import CustomFedAvg
from src.strategies.fedper import FedPer
from src.strategies.fedprox import CustomFedProx
from src.strategies.mifa import CustomMIFA
from src.strategies.scaffold import SCAFFOLD

__all__ = [
    "CustomFedAvg",
    "CustomFedProx",
    "CustomMIFA",
    "CustomClusteredFL",
    "SCAFFOLD",
    "DIWS",
    "FDMS",
    "FedPer",
    "FedAdam",
    "FedYogi",
    "FedAdagrad",
]

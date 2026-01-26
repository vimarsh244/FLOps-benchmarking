"""Scenario handlers for federated learning experiments."""

from src.scenarios.base import BaseScenario
from src.scenarios.node_drop import NodeDropScenario
from src.scenarios.timeout import TimeoutScenario
from src.scenarios.registry import get_scenario

__all__ = [
    "BaseScenario",
    "NodeDropScenario",
    "TimeoutScenario",
    "get_scenario",
]

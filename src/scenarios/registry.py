"""Scenario registry for easy scenario creation."""

from typing import Dict, Type, Optional
from omegaconf import DictConfig

from src.scenarios.base import BaseScenario, NoOpScenario
from src.scenarios.node_drop import NodeDropScenario
from src.scenarios.timeout import TimeoutScenario

# registry mapping scenario names to classes
SCENARIO_REGISTRY: Dict[str, Type[BaseScenario]] = {
    "baseline": NoOpScenario,
    "node_drop": NodeDropScenario,
    "node_drop_standard": NodeDropScenario,
    "node_drop_standard_20clients": NodeDropScenario,
    "timeout": TimeoutScenario,
}


def get_scenario(
    scenario_cfg: DictConfig,
) -> BaseScenario:
    """Create a scenario from Hydra configuration.

    Args:
        scenario_cfg: Scenario configuration from Hydra

    Returns:
        Initialized scenario handler

    Raises:
        ValueError: If scenario name is not in registry
    """
    name = scenario_cfg.get("name", "baseline").lower()

    if name not in SCENARIO_REGISTRY:
        available = ", ".join(SCENARIO_REGISTRY.keys())
        raise ValueError(f"Unknown scenario: {name}. Available: {available}")

    scenario_class = SCENARIO_REGISTRY[name]
    return scenario_class(config=scenario_cfg)


def combine_scenarios(*scenarios: BaseScenario) -> "CombinedScenario":
    """Combine multiple scenarios into one.

    A client will only participate if ALL scenarios allow it.

    Args:
        *scenarios: Scenario handlers to combine

    Returns:
        Combined scenario handler
    """
    return CombinedScenario(list(scenarios))


class CombinedScenario(BaseScenario):
    """Combines multiple scenarios - client participates only if all allow."""

    def __init__(self, scenarios: list):
        super().__init__()
        self.scenarios = scenarios
        self.enabled = any(s.enabled for s in scenarios)

    def should_client_participate(
        self,
        client_id: int,
        current_round: int,
        **kwargs,
    ) -> bool:
        """Client participates only if ALL scenarios allow it."""
        for scenario in self.scenarios:
            if not scenario.should_client_participate(client_id, current_round, **kwargs):
                return False
        return True

    def get_client_config(
        self,
        client_id: int,
        current_round: int,
        **kwargs,
    ) -> dict:
        """Merge configs from all scenarios."""
        config = {}
        for scenario in self.scenarios:
            config.update(scenario.get_client_config(client_id, current_round, **kwargs))
        return config

    def on_round_start(self, current_round: int, **kwargs) -> None:
        for scenario in self.scenarios:
            scenario.on_round_start(current_round, **kwargs)

    def on_round_end(self, current_round: int, **kwargs) -> None:
        for scenario in self.scenarios:
            scenario.on_round_end(current_round, **kwargs)

    def __repr__(self) -> str:
        scenario_names = [s.__class__.__name__ for s in self.scenarios]
        return f"CombinedScenario({scenario_names})"

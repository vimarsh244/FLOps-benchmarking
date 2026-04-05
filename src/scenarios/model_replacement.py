"""Model replacement (A3) attack scenario.

This scenario marks a subset of clients as adversaries and provides
configuration for scaling their local updates before they are sent to
the server.
"""

from typing import Any, Dict, Optional, Set
import random

from omegaconf import DictConfig

from src.scenarios.base import BaseScenario


class ModelReplacementScenario(BaseScenario):
    """Scenario implementing model replacement attack configuration.

    Configuration:
        enabled: bool
        adversary_fraction: float in [0,1]
        num_clients: int
        selection_mode: first_n | random
        fixed_adversary_ids: optional explicit list of ids
        replacement_scale: optional float; if null uses clients_per_round
        clients_per_round: int used when replacement_scale is null
        seed: random seed for random selection mode
    """

    def __init__(self, config: Optional[DictConfig] = None):
        super().__init__(config)
        self.adversary_fraction = 0.2
        self.num_clients = 0
        self.selection_mode = "first_n"
        self.fixed_adversary_ids: Set[int] = set()
        self.replacement_scale: Optional[float] = None
        self.clients_per_round = 10
        self.seed = 0
        self._adversary_ids: Set[int] = set()

        if config:
            self._parse_config(config)

        if self.enabled:
            self._build_adversary_ids()

    def _parse_config(self, config: DictConfig) -> None:
        self.adversary_fraction = float(config.get("adversary_fraction", 0.2))
        self.num_clients = int(config.get("num_clients", 0))
        self.selection_mode = str(config.get("selection_mode", "first_n")).lower()
        self.fixed_adversary_ids = set(int(i) for i in config.get("fixed_adversary_ids", []))
        scale = config.get("replacement_scale", None)
        self.replacement_scale = None if scale is None else float(scale)
        self.clients_per_round = int(config.get("clients_per_round", 10))
        self.seed = int(config.get("seed", 0))

    def _build_adversary_ids(self) -> None:
        if self.fixed_adversary_ids:
            self._adversary_ids = set(self.fixed_adversary_ids)
            return

        if self.num_clients <= 0:
            self._adversary_ids = set()
            return

        num_adversaries = int(round(self.adversary_fraction * self.num_clients))
        num_adversaries = max(0, min(num_adversaries, self.num_clients))

        if self.selection_mode == "random":
            rng = random.Random(self.seed)
            self._adversary_ids = set(rng.sample(range(self.num_clients), num_adversaries))
        else:
            self._adversary_ids = set(range(num_adversaries))

    def is_adversary(self, client_id: int) -> bool:
        if not self.enabled:
            return False
        return client_id in self._adversary_ids

    def should_client_participate(
        self,
        client_id: int,
        current_round: int,
        **kwargs: Any,
    ) -> bool:
        return True

    def get_client_config(
        self,
        client_id: int,
        current_round: int,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        scale = self.replacement_scale
        if scale is None:
            scale = float(self.clients_per_round)

        return {
            "attack_type": "model_replacement",
            "is_adversary": self.is_adversary(client_id),
            "replacement_scale": scale,
        }

    def __repr__(self) -> str:
        return (
            "ModelReplacementScenario("
            f"enabled={self.enabled}, adversaries={len(self._adversary_ids)}, "
            f"scale={self.replacement_scale if self.replacement_scale is not None else self.clients_per_round}"
            ")"
        )

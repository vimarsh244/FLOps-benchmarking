"""Timeout scenario handler.

Simulates timeouts for straggler nodes.
"""

import time
import random
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from omegaconf import DictConfig

from src.scenarios.base import BaseScenario


@dataclass
class StragglerConfig:
    """Configuration for straggler simulation."""
    enabled: bool = True
    probability: float = 0.2
    delay_multiplier: float = 3.0
    fixed_straggler_ids: Set[int] = None
    
    def __post_init__(self):
        if self.fixed_straggler_ids is None:
            self.fixed_straggler_ids = set()


class TimeoutScenario(BaseScenario):
    """Scenario that simulates timeouts for straggler nodes.
    
    Configuration:
        timeout_seconds: maximum time to wait for client response
        simulate_stragglers:
            enabled: bool
            straggler_probability: float
            delay_multiplier: float
            fixed_straggler_ids: list[int]
        on_timeout:
            action: skip | retry | fail
            max_retries: int
            retry_delay: float
    """

    def __init__(self, config: Optional[DictConfig] = None):
        super().__init__(config)
        
        self.timeout_seconds = 30.0
        self.straggler_config = StragglerConfig()
        self.on_timeout_action = "skip"
        self.max_retries = 1
        self.retry_delay = 5.0
        
        # track stragglers per round
        self._round_stragglers: Dict[int, Set[int]] = {}
        self._rng = random.Random(42)
        
        if self.enabled and config:
            self._parse_config(config)

    def _parse_config(self, config: DictConfig) -> None:
        """Parse configuration."""
        self.timeout_seconds = config.get("timeout_seconds", 30.0)
        
        if "simulate_stragglers" in config:
            straggler_cfg = config.simulate_stragglers
            fixed_ids = set(straggler_cfg.get("fixed_straggler_ids", []))
            self.straggler_config = StragglerConfig(
                enabled=straggler_cfg.get("enabled", True),
                probability=straggler_cfg.get("straggler_probability", 0.2),
                delay_multiplier=straggler_cfg.get("delay_multiplier", 3.0),
                fixed_straggler_ids=fixed_ids,
            )
        
        if "on_timeout" in config:
            timeout_cfg = config.on_timeout
            self.on_timeout_action = timeout_cfg.get("action", "skip")
            self.max_retries = timeout_cfg.get("max_retries", 1)
            self.retry_delay = timeout_cfg.get("retry_delay", 5.0)

    def should_client_participate(
        self,
        client_id: int,
        current_round: int,
        **kwargs: Any,
    ) -> bool:
        """All clients can participate, but some may timeout."""
        return True

    def is_straggler(self, client_id: int, current_round: int) -> bool:
        """Determine if a client is a straggler for the current round.
        
        Args:
            client_id: Client partition ID
            current_round: Current training round
        
        Returns:
            True if client is a straggler
        """
        if not self.enabled or not self.straggler_config.enabled:
            return False
        
        # check fixed stragglers
        if client_id in self.straggler_config.fixed_straggler_ids:
            return True
        
        # check round-specific stragglers (cached)
        if current_round not in self._round_stragglers:
            self._round_stragglers[current_round] = set()
        
        if client_id in self._round_stragglers[current_round]:
            return True
        
        # probabilistic determination (deterministic based on seed)
        # use round and client_id for reproducibility
        seed = current_round * 1000 + client_id
        self._rng.seed(seed)
        if self._rng.random() < self.straggler_config.probability:
            self._round_stragglers[current_round].add(client_id)
            return True
        
        return False

    def get_client_config(
        self,
        client_id: int,
        current_round: int,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Get config for client including straggler delay."""
        config = {}
        
        if self.is_straggler(client_id, current_round):
            config["is_straggler"] = True
            config["delay_multiplier"] = self.straggler_config.delay_multiplier
        else:
            config["is_straggler"] = False
            config["delay_multiplier"] = 1.0
        
        config["timeout_seconds"] = self.timeout_seconds
        
        return config

    def simulate_straggler_delay(
        self,
        client_id: int,
        current_round: int,
        base_duration: float,
    ) -> float:
        """Simulate additional delay for straggler clients.
        
        Args:
            client_id: Client partition ID
            current_round: Current training round
            base_duration: Base training duration
        
        Returns:
            Actual duration with straggler delay applied
        """
        if self.is_straggler(client_id, current_round):
            return base_duration * self.straggler_config.delay_multiplier
        return base_duration

    def would_timeout(
        self,
        client_id: int,
        current_round: int,
        training_duration: float,
    ) -> bool:
        """Check if a client would timeout given their training duration.
        
        Args:
            client_id: Client partition ID
            current_round: Current training round
            training_duration: Expected training duration
        
        Returns:
            True if client would timeout
        """
        if not self.enabled:
            return False
        
        actual_duration = self.simulate_straggler_delay(
            client_id, current_round, training_duration
        )
        return actual_duration > self.timeout_seconds

    def get_stragglers_for_round(self, current_round: int, num_clients: int) -> Set[int]:
        """Get set of straggler client IDs for a given round.
        
        Args:
            current_round: Current training round
            num_clients: Total number of clients
        
        Returns:
            Set of client IDs that are stragglers in this round
        """
        stragglers = set()
        for client_id in range(num_clients):
            if self.is_straggler(client_id, current_round):
                stragglers.add(client_id)
        return stragglers

    def on_round_start(self, current_round: int, **kwargs: Any) -> None:
        """Clear straggler cache for new round."""
        # keep only recent rounds to avoid memory leak
        rounds_to_keep = 5
        old_rounds = [r for r in self._round_stragglers if r < current_round - rounds_to_keep]
        for r in old_rounds:
            del self._round_stragglers[r]

    def __repr__(self) -> str:
        if not self.enabled:
            return "TimeoutScenario(enabled=False)"
        
        return (
            f"TimeoutScenario(timeout={self.timeout_seconds}s, "
            f"straggler_prob={self.straggler_config.probability})"
        )


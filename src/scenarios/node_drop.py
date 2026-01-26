"""Node drop scenario handler.

Simulates clients disconnecting and rejoining at specific rounds.
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from omegaconf import DictConfig, ListConfig

from src.scenarios.base import BaseScenario


@dataclass
class DropEvent:
    """Represents a node drop event."""

    client_ids: Set[int]
    disconnect_round: int
    rejoin_round: int


class NodeDropScenario(BaseScenario):
    """Scenario that simulates node disconnection and rejoining.

    Can be configured with explicit drop events or auto-mode based on partition ID.

    Configuration:
        drop_events: list of {client_ids, disconnect_round, rejoin_round}
        auto_mode:
            enabled: bool
            start_partition_id: int
            disconnect_offset: int
            rejoin_round: int
    """

    def __init__(self, config: Optional[DictConfig] = None):
        super().__init__(config)
        self.drop_events: List[DropEvent] = []
        self.auto_mode_enabled = False
        self.auto_start_partition = 2
        self.auto_disconnect_offset = 3
        self.auto_rejoin_round = 31

        if self.enabled and config:
            self._parse_config(config)

    def _parse_config(self, config: DictConfig) -> None:
        """Parse configuration to build drop events."""
        # parse explicit drop events
        if "drop_events" in config:
            for event in config.drop_events:
                client_ids = set(event.get("client_ids", []))
                self.drop_events.append(
                    DropEvent(
                        client_ids=client_ids,
                        disconnect_round=event.get("disconnect_round", 5),
                        rejoin_round=event.get("rejoin_round", 31),
                    )
                )

        # parse auto mode
        if "auto_mode" in config and config.auto_mode.get("enabled", False):
            self.auto_mode_enabled = True
            self.auto_start_partition = config.auto_mode.get("start_partition_id", 2)
            self.auto_disconnect_offset = config.auto_mode.get("disconnect_offset", 3)
            self.auto_rejoin_round = config.auto_mode.get("rejoin_round", 31)

    def _is_dropped_explicit(self, client_id: int, current_round: int) -> bool:
        """Check if client is dropped based on explicit events."""
        for event in self.drop_events:
            if client_id in event.client_ids:
                if event.disconnect_round <= current_round < event.rejoin_round:
                    return True
        return False

    def _is_dropped_auto(self, client_id: int, current_round: int) -> bool:
        """Check if client is dropped based on auto mode."""
        if not self.auto_mode_enabled:
            return False

        if client_id < self.auto_start_partition:
            return False

        # disconnect round = partition_id + offset
        disconnect_round = client_id + self.auto_disconnect_offset
        rejoin_round = self.auto_rejoin_round

        return disconnect_round <= current_round < rejoin_round

    def is_client_dropped(self, client_id: int, current_round: int) -> bool:
        """Check if a client is currently dropped.

        Args:
            client_id: Client partition ID
            current_round: Current training round

        Returns:
            True if client is dropped (disconnected), False otherwise
        """
        if not self.enabled:
            return False

        return self._is_dropped_explicit(client_id, current_round) or self._is_dropped_auto(
            client_id, current_round
        )

    def should_client_participate(
        self,
        client_id: int,
        current_round: int,
        **kwargs: Any,
    ) -> bool:
        """Determine if client should participate (not dropped)."""
        return not self.is_client_dropped(client_id, current_round)

    def get_drop_info(self, client_id: int) -> Dict[str, Any]:
        """Get drop information for a client.

        Args:
            client_id: Client partition ID

        Returns:
            Dictionary with disconnect_round, rejoin_round if applicable
        """
        # check explicit events first
        for event in self.drop_events:
            if client_id in event.client_ids:
                return {
                    "will_drop": True,
                    "disconnect_round": event.disconnect_round,
                    "rejoin_round": event.rejoin_round,
                }

        # check auto mode
        if self.auto_mode_enabled and client_id >= self.auto_start_partition:
            disconnect_round = client_id + self.auto_disconnect_offset
            return {
                "will_drop": True,
                "disconnect_round": disconnect_round,
                "rejoin_round": self.auto_rejoin_round,
            }

        return {"will_drop": False}

    def get_dropped_clients_for_round(self, current_round: int, num_clients: int) -> Set[int]:
        """Get set of dropped client IDs for a given round.

        Args:
            current_round: Current training round
            num_clients: Total number of clients

        Returns:
            Set of client IDs that are dropped in this round
        """
        dropped = set()
        for client_id in range(num_clients):
            if self.is_client_dropped(client_id, current_round):
                dropped.add(client_id)
        return dropped

    def __repr__(self) -> str:
        if not self.enabled:
            return "NodeDropScenario(enabled=False)"

        info = []
        if self.drop_events:
            info.append(f"events={len(self.drop_events)}")
        if self.auto_mode_enabled:
            info.append(f"auto_mode=True")

        return f"NodeDropScenario({', '.join(info)})"

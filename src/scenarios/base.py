"""Base scenario handler."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from omegaconf import DictConfig


class BaseScenario(ABC):
    """Base class for scenario handlers.
    
    Scenarios define special behaviors during federated learning,
    such as node disconnections, timeouts, or custom client selection.
    """

    def __init__(self, config: Optional[DictConfig] = None):
        """Initialize scenario.
        
        Args:
            config: Hydra configuration for the scenario
        """
        self.config = config or {}
        self.enabled = self.config.get("enabled", False)

    @abstractmethod
    def should_client_participate(
        self,
        client_id: int,
        current_round: int,
        **kwargs: Any,
    ) -> bool:
        """Determine if a client should participate in the current round.
        
        Args:
            client_id: ID of the client (partition_id)
            current_round: Current training round
            **kwargs: Additional context
        
        Returns:
            True if client should participate, False otherwise
        """
        pass

    def get_client_config(
        self,
        client_id: int,
        current_round: int,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Get additional configuration for a client.
        
        Args:
            client_id: ID of the client
            current_round: Current training round
            **kwargs: Additional context
        
        Returns:
            Dictionary of configuration values to pass to client
        """
        return {}

    def on_round_start(self, current_round: int, **kwargs: Any) -> None:
        """Called at the start of each round.
        
        Args:
            current_round: Current training round
            **kwargs: Additional context
        """
        pass

    def on_round_end(self, current_round: int, **kwargs: Any) -> None:
        """Called at the end of each round.
        
        Args:
            current_round: Current training round
            **kwargs: Additional context
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(enabled={self.enabled})"


class NoOpScenario(BaseScenario):
    """No-op scenario that allows all clients to participate.
    
    Used for baseline experiments with no special behavior.
    """

    def should_client_participate(
        self,
        client_id: int,
        current_round: int,
        **kwargs: Any,
    ) -> bool:
        """Always returns True - all clients participate."""
        return True


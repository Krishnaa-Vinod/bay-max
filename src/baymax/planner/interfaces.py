"""Planner module interfaces."""

from abc import ABC, abstractmethod

from baymax.core.enums import ResponseStrategy
from baymax.schemas.memory import MemoryQueryResult
from baymax.state.models import InteractionState


class ResponsePlanner(ABC):
    """Interface for choosing a response strategy."""

    @abstractmethod
    def plan(
        self, state: InteractionState, memories: MemoryQueryResult
    ) -> ResponseStrategy:
        """Choose a response strategy given current state and retrieved memories."""
        ...

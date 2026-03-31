"""Planner module interfaces."""

from abc import ABC, abstractmethod

from baymax.core.enums import ResponseStrategy, TurnIntent
from baymax.schemas.memory import MemoryQueryResult
from baymax.state.models import InteractionState


class ResponsePlanner(ABC):
    """Interface for choosing a response strategy."""

    @abstractmethod
    def plan(
        self,
        state: InteractionState,
        memories: MemoryQueryResult,
        context: str = "",
        turn_intent: TurnIntent | None = None,
        proactive: bool = False,
    ) -> ResponseStrategy:
        """Choose a response strategy given current state and retrieved memories."""
        ...

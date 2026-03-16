"""Dialogue module interfaces."""

from abc import ABC, abstractmethod

from baymax.core.enums import ResponseStrategy
from baymax.schemas.memory import MemoryQueryResult
from baymax.schemas.response import SupportiveResponse
from baymax.state.models import InteractionState


class DialogueProvider(ABC):
    """Interface for generating supportive responses."""

    @abstractmethod
    def generate(
        self,
        strategy: ResponseStrategy,
        state: InteractionState,
        memories: MemoryQueryResult,
    ) -> SupportiveResponse:
        """Generate a supportive response based on strategy, state, and memories."""
        ...

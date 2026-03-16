"""Dialogue module interfaces."""

from abc import ABC, abstractmethod

from baymax.core.enums import ResponseStrategy
from baymax.schemas.memory import MemoryQueryResult
from baymax.schemas.response import GroundedPromptContext, SupportiveResponse
from baymax.state.models import InteractionState


class DialogueProvider(ABC):
    """Interface for generating supportive responses."""

    @abstractmethod
    def generate(
        self,
        strategy: ResponseStrategy,
        state: InteractionState,
        memories: MemoryQueryResult,
        prompt_context: GroundedPromptContext | None = None,
    ) -> SupportiveResponse:
        """Generate a supportive response based on strategy, state, and memories.

        Args:
            strategy: The planned response strategy.
            state: Current interaction state.
            memories: Retrieved memories for this user.
            prompt_context: Optional grounded prompt context (used by LLM backends).

        Returns:
            A SupportiveResponse including backend/model metadata.
        """
        ...

    @property
    def backend_name(self) -> str:
        """Return the identifier for this backend."""
        return "unknown"

    @property
    def model_name(self) -> str:
        """Return the model name used by this backend."""
        return ""

    def is_available(self) -> bool:
        """Return True if this backend is ready to generate responses."""
        return True

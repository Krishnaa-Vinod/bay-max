"""State manager for interaction sessions."""

from uuid import UUID

from baymax.core.enums import EmotionLabel, EngagementLevel
from baymax.state.models import InteractionState


class StateManager:
    """Manages the current interaction state for active sessions."""

    def __init__(self) -> None:
        self._states: dict[UUID, InteractionState] = {}

    def get_or_create(
        self, session_id: UUID, user_id: UUID | None = None
    ) -> InteractionState:
        """Get existing state or create a new one for the session."""
        if session_id not in self._states:
            self._states[session_id] = InteractionState(
                session_id=session_id,
                user_id=user_id,
            )
        return self._states[session_id]

    def update_engagement(
        self, session_id: UUID, level: EngagementLevel
    ) -> InteractionState | None:
        """Update engagement level for a session."""
        state = self._states.get(session_id)
        if state:
            state.engagement_level = level
        return state

    def update_emotion(
        self, session_id: UUID, emotion: EmotionLabel
    ) -> InteractionState | None:
        """Update dominant emotion for a session."""
        state = self._states.get(session_id)
        if state:
            state.dominant_emotion = emotion
        return state

    def increment_turn(self, session_id: UUID) -> InteractionState | None:
        """Increment the turn count for a session."""
        state = self._states.get(session_id)
        if state:
            state.turn_count += 1
        return state

    def remove(self, session_id: UUID) -> None:
        """Remove state for an ended session."""
        self._states.pop(session_id, None)

"""State manager for interaction sessions."""

from datetime import datetime
from uuid import UUID

from baymax.core.enums import EmotionLabel, EngagementLevel
from baymax.schemas.perception import EngagementResult
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

    def update_recognition(
        self,
        session_id: UUID,
        *,
        user_id: UUID | None = None,
        user_display_name: str | None = None,
        identity_confidence: float = 0.0,
        face_count: int = 0,
        active_track_ids: list[str] | None = None,
        frame_summary: str = "",
    ) -> InteractionState | None:
        """Update state with face recognition results."""
        state = self._states.get(session_id)
        if state is None:
            return None
        if user_id is not None:
            state.user_id = user_id
            state.is_new_user = False
        if user_display_name is not None:
            state.user_display_name = user_display_name
        state.identity_confidence = identity_confidence
        state.face_count = face_count
        state.last_seen_at = datetime.utcnow()
        if active_track_ids is not None:
            state.active_track_ids = active_track_ids
        state.last_frame_summary = frame_summary
        return state

    def update_body_state(
        self,
        session_id: UUID,
        engagement_result: EngagementResult,
    ) -> InteractionState | None:
        """Update state with body-state and engagement results."""
        state = self._states.get(session_id)
        if state is None:
            return None
        state.pose_visible = engagement_result.pose_visible
        state.posture = engagement_result.posture
        state.lean = engagement_result.lean
        state.motion = engagement_result.motion
        state.engagement_level = engagement_result.level
        state.engagement_score = engagement_result.score
        return state

    def remove(self, session_id: UUID) -> None:
        """Remove state for an ended session."""
        self._states.pop(session_id, None)

"""Interaction state models."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from baymax.core.enums import EmotionLabel, EngagementLevel, ResponseStrategy


class InteractionState(BaseModel):
    """Current state of an interaction session."""

    session_id: UUID
    user_id: UUID | None = None
    user_display_name: str | None = None
    engagement_level: EngagementLevel = EngagementLevel.MEDIUM
    dominant_emotion: EmotionLabel = EmotionLabel.NEUTRAL
    turn_count: int = 0
    suggested_strategy: ResponseStrategy = ResponseStrategy.GREET
    memory_summary: str = ""
    is_new_user: bool = True

    # --- Iteration 002: recognition-oriented fields ---
    identity_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    face_count: int = 0
    last_seen_at: datetime | None = None
    active_track_ids: list[str] = Field(default_factory=list)
    last_frame_summary: str = ""

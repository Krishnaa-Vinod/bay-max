"""Interaction state models."""

from uuid import UUID

from pydantic import BaseModel

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

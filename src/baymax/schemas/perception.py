"""Perception-related Pydantic schemas."""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from baymax.core.enums import EmotionLabel, EngagementLevel, PerceptionEventType


class PerceptionEvent(BaseModel):
    """A single perception event from the vision pipeline."""

    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    event_type: PerceptionEventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_id: UUID | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    engagement_level: EngagementLevel | None = None
    emotion_label: EmotionLabel | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    """A factual observation derived from perception. Observations are raw facts,
    not inferences."""

    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    user_id: UUID | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    content: str = Field(..., description="Natural-language description of the observation.")
    source: str = Field(default="perception", description="Module that produced this observation.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence_refs: list[str] = Field(
        default_factory=list,
        description="References to supporting data (event IDs, frame numbers, etc.).",
    )

"""Session-related Pydantic schemas."""

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from baymax.core.enums import SessionStatus


class SessionCreate(BaseModel):
    """Request schema for creating a new session."""

    user_id: UUID | None = Field(
        default=None, description="User ID if known; None for anonymous sessions."
    )


class Session(BaseModel):
    """Interaction session model."""

    id: UUID = Field(default_factory=uuid4)
    user_id: UUID | None = None
    status: SessionStatus = SessionStatus.ACTIVE
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: datetime | None = None
    frame_count: int = 0
    notes: str = ""

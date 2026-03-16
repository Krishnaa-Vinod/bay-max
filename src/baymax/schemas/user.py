"""User-related Pydantic schemas."""

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class UserProfileCreate(BaseModel):
    """Request schema for creating a new user."""

    display_name: str = Field(..., min_length=1, max_length=200)
    notes: str = Field(default="", max_length=1000)


class UserProfile(BaseModel):
    """Full user profile model."""

    id: UUID = Field(default_factory=uuid4)
    display_name: str
    notes: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True


class FaceEnrollment(BaseModel):
    """Placeholder model for face enrollment metadata."""

    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    enrolled_at: datetime = Field(default_factory=datetime.utcnow)
    encoding_ref: str = Field(
        default="",
        description="Reference path or key for the stored face encoding (not stored in repo).",
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    status: str = Field(default="pending", description="pending | enrolled | revoked")

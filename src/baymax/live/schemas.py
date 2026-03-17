"""Pydantic schemas for the live runtime module."""

from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class LiveSourceType(StrEnum):
    WEBCAM = "webcam"
    REPLAY = "replay"


class SessionLifecycleState(StrEnum):
    IDLE = "idle"
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"


class CompanionEventType(StrEnum):
    PERSON_ARRIVED = "person_arrived"
    RECOGNIZED_USER_ARRIVED = "recognized_user_arrived"
    UNKNOWN_USER_ARRIVED = "unknown_user_arrived"
    PERSON_DEPARTED = "person_departed"
    SESSION_PAUSED = "session_paused"
    SESSION_RESUMED = "session_resumed"
    RECOGNITION_GAINED = "recognition_gained"
    RECOGNITION_LOST = "recognition_lost"
    POSTURE_CHANGED = "posture_changed"
    ENGAGEMENT_CHANGED = "engagement_changed"
    QUIET_COMPANIONSHIP_DUE = "quiet_companionship_due"


class LiveRuntimeStatus(BaseModel):
    """Current status of the live runtime."""

    live_mode_active: bool = False
    source: str = ""
    session_status: SessionLifecycleState = SessionLifecycleState.IDLE
    current_user_id: UUID | None = None
    current_user_display_name: str | None = None
    last_event: str | None = None
    last_event_at: datetime | None = None
    last_response_text: str | None = None
    last_response_at: datetime | None = None
    frame_count: int = 0
    analysis_count: int = 0
    uptime_sec: float = 0.0
    # Iteration 007 — speech state
    last_spoken_text: str = ""
    speech_queue_depth: int = 0
    tts_backend: str = ""
    tts_voice: str = ""


class LiveFrameResult(BaseModel):
    """Result from processing a single live frame."""

    frame_index: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    faces_detected: int = 0
    recognized_user_id: UUID | None = None
    recognized_display_name: str | None = None
    identity_confidence: float = 0.0
    posture: str = "unknown"
    engagement: str = "medium"
    engagement_score: float = 0.5
    is_analysis_frame: bool = False


class SessionLifecycleEvent(BaseModel):
    """A session lifecycle transition event."""

    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_type: str  # e.g. "session_started", "session_paused", "session_resumed", "session_ended"
    session_id: UUID | None = None
    user_id: UUID | None = None
    reason: str = ""


class CompanionEvent(BaseModel):
    """A meaningful event detected by the event engine."""

    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_type: CompanionEventType
    session_id: UUID | None = None
    user_id: UUID | None = None
    user_display_name: str | None = None
    confidence: float = 0.0
    details: dict[str, str] = Field(default_factory=dict)
    suppressed: bool = False
    suppression_reason: str = ""


class ProactiveDecision(BaseModel):
    """Decision on whether to proactively respond to an event."""

    event: CompanionEvent
    should_respond: bool = False
    trigger_reason: str = ""
    suppressed: bool = False
    suppression_reason: str = ""
    cooldown_remaining_sec: float = 0.0


class CooldownState(BaseModel):
    """Tracks cooldown timers for proactive responses."""

    last_any_response_at: datetime | None = None
    last_proactive_response_at: datetime | None = None
    last_greeting_at: datetime | None = None
    last_quiet_companionship_at: datetime | None = None
    last_event_responses: dict[str, datetime] = Field(default_factory=dict)


class LiveArtifactRef(BaseModel):
    """Reference to an artifact produced during a live run."""

    path: str
    artifact_type: str  # event_log, response_log, session_timeline, snapshot, manifest, summary
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    session_id: UUID | None = None


class LiveRunSummary(BaseModel):
    """Summary of a complete live run."""

    run_id: UUID = Field(default_factory=uuid4)
    source: str = ""
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: datetime | None = None
    total_frames: int = 0
    total_analyses: int = 0
    total_events: int = 0
    total_responses: int = 0
    total_suppressions: int = 0
    sessions_created: int = 0
    sessions_resumed: int = 0
    users_recognized: list[str] = Field(default_factory=list)
    artifacts: list[LiveArtifactRef] = Field(default_factory=list)
    backend: str = ""
    model_name: str = ""
    errors: list[str] = Field(default_factory=list)

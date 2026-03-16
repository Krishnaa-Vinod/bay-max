"""Core enums and constants for Bay-Max."""

from enum import StrEnum


class SessionStatus(StrEnum):
    """Status of an interaction session."""

    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"


class EngagementLevel(StrEnum):
    """Estimated engagement level of the user."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ABSENT = "absent"


class EmotionLabel(StrEnum):
    """High-level emotion labels for perception events."""

    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANXIOUS = "anxious"
    FRUSTRATED = "frustrated"
    TIRED = "tired"
    UNKNOWN = "unknown"


class MemoryType(StrEnum):
    """Types of memory stored by the system."""

    OBSERVATION = "observation"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    INTERVENTION = "intervention"


class MemoryStatus(StrEnum):
    """Status of a memory record."""

    ACTIVE = "active"
    ARCHIVED = "archived"
    CORRECTED = "corrected"
    DELETED = "deleted"


class ResponseStrategy(StrEnum):
    """Strategy for generating a supportive response."""

    GREET = "greet"
    CHECK_IN = "check_in"
    ENCOURAGE = "encourage"
    EMPATHIZE = "empathize"
    RECALL = "recall"
    SUGGEST = "suggest"
    FAREWELL = "farewell"


class PerceptionEventType(StrEnum):
    """Types of perception events."""

    FACE_DETECTED = "face_detected"
    FACE_RECOGNIZED = "face_recognized"
    FACE_LOST = "face_lost"
    ENGAGEMENT_CHANGE = "engagement_change"
    EMOTION_CHANGE = "emotion_change"
    POSE_CHANGE = "pose_change"

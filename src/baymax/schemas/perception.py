"""Perception-related Pydantic schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from baymax.core.enums import (
    EmotionLabel,
    EngagementLevel,
    LeanLabel,
    MotionLevel,
    PerceptionEventType,
    PostureLabel,
)


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


# --- Iteration 002: Structured face detection / recognition models ---


class FaceBoundingBox(BaseModel):
    """Pixel-coordinate bounding box for a detected face."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


class DetectedFace(BaseModel):
    """A single face detected in a frame."""

    bbox: FaceBoundingBox
    detection_confidence: float = Field(ge=0.0, le=1.0)
    embedding: list[float] | None = Field(
        default=None, description="Face embedding vector (excluded from API responses)."
    )
    backend: str = "facenet_pytorch"


class RecognizedFace(BaseModel):
    """A detected face matched to an enrolled user."""

    user_id: UUID
    user_display_name: str | None = None
    bbox: FaceBoundingBox
    detection_confidence: float = Field(ge=0.0, le=1.0)
    match_confidence: float = Field(ge=0.0, le=1.0)
    backend: str = "facenet_pytorch"


class UnknownFace(BaseModel):
    """A detected face that could not be matched to any enrolled user."""

    bbox: FaceBoundingBox
    detection_confidence: float = Field(ge=0.0, le=1.0)
    best_match_score: float | None = Field(
        default=None, description="Highest similarity score among enrolled users, if any."
    )


class FaceEmbeddingRecord(BaseModel):
    """Stored face embedding for an enrolled user."""

    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    embedding: list[float]
    backend: str = "facenet_pytorch"
    model_name: str = "InceptionResnetV1-vggface2"
    enrolled_at: datetime = Field(default_factory=datetime.utcnow)


class FrameAnalysisResult(BaseModel):
    """Complete result of analyzing a single frame."""

    session_id: UUID
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    faces_detected: int = 0
    recognized_faces: list[RecognizedFace] = Field(default_factory=list)
    unknown_faces: list[UnknownFace] = Field(default_factory=list)
    pose_result: PoseResult | None = None
    engagement: EngagementResult | None = None
    state: dict[str, Any] = Field(default_factory=dict)
    observations_written: int = 0
    latency_ms: float = 0.0
    annotation_artifact: AnnotatedArtifactRef | None = None


class RecognitionObservation(BaseModel):
    """An observation specifically about a recognition event."""

    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    user_id: UUID | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_type: str = Field(
        default="first_recognition",
        description=(
            "first_recognition | unknown_to_known | known_to_unknown"
            " | user_switch | confidence_change"
        ),
    )
    content: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence_refs: list[str] = Field(default_factory=list)


# --- Iteration 003: Pose estimation and engagement models ---


class PoseLandmark2D(BaseModel):
    """A single 2D pose landmark with visibility."""

    x: float = Field(description="Normalized x coordinate (0.0-1.0)")
    y: float = Field(description="Normalized y coordinate (0.0-1.0)")
    visibility: float = Field(default=0.0, ge=0.0, le=1.0)
    name: str = ""


class PoseResult(BaseModel):
    """Result of pose estimation on a single frame."""

    pose_present: bool = False
    landmarks: list[PoseLandmark2D] = Field(default_factory=list)
    landmark_count: int = 0
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    backend: str = "mediapipe"
    model_name: str = "pose_landmarker"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class EngagementResult(BaseModel):
    """Computed engagement level with supporting evidence."""

    level: EngagementLevel = EngagementLevel.MEDIUM
    posture: PostureLabel = PostureLabel.UNKNOWN
    lean: LeanLabel = LeanLabel.UNKNOWN
    motion: MotionLevel = MotionLevel.UNKNOWN
    face_visible: bool = False
    face_recognized: bool = False
    pose_visible: bool = False
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AnnotatedArtifactRef(BaseModel):
    """Reference to a saved annotated debug artifact."""

    path: str
    artifact_type: str = "annotated_frame"
    session_id: UUID | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BodyStateObservation(BaseModel):
    """An observation about body-state or engagement changes."""

    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    user_id: UUID | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_type: str = Field(
        default="posture_change",
        description="posture_change | engagement_change | pose_first_seen | pose_lost",
    )
    content: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence_refs: list[str] = Field(default_factory=list)


# --- Iteration 009: Affect analysis models ---


class AffectEvidence(BaseModel):
    """Evidence supporting an emotion analysis result."""

    landmark_indices: list[int] = Field(default_factory=list, description="Relevant facial landmark indices")
    action_units: dict[str, float] = Field(default_factory=dict, description="Action unit activations (if available)")
    blendshapes: dict[str, float] = Field(default_factory=dict, description="MediaPipe face blendshapes (if available)")
    processing_method: str = Field(default="unknown", description="Backend method used")
    processing_time_ms: float = Field(default=0.0, description="Analysis processing time")
    frame_quality_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Face region quality")


class EmotionResult(BaseModel):
    """Result of facial affect analysis for a single frame or region."""

    backend: str = Field(description="Backend used (null, mediapipe, etc.)")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall confidence in the analysis")
    valence: float = Field(default=0.0, ge=-1.0, le=1.0, description="Valence dimension (-1=negative, +1=positive)")
    arousal: float = Field(default=0.0, ge=0.0, le=1.0, description="Arousal dimension (0=calm, 1=activated)")
    dominant_emotion_soft_label: str | None = Field(default=None, description="Soft label like 'appears calm' or 'seems energized' - not diagnostic")
    evidence: AffectEvidence = Field(default_factory=AffectEvidence)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    success: bool = Field(default=False, description="Whether analysis completed successfully")


class SmoothedAffectState(BaseModel):
    """Smoothed affect state tracked over time."""

    valence: float = Field(default=0.0, ge=-1.0, le=1.0, description="Smoothed valence")
    arousal: float = Field(default=0.0, ge=0.0, le=1.0, description="Smoothed arousal")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence in current smoothed state")
    stable_duration_sec: float = Field(default=0.0, description="How long state has been stable")
    last_update: datetime = Field(default_factory=datetime.utcnow)
    enabled: bool = Field(default=False, description="Whether affect analysis is enabled")
    sample_count: int = Field(default=0, description="Number of samples contributing to this state")


class AffectMemoryObservation(BaseModel):
    """Session-level affect summary for memory consolidation."""

    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    user_id: UUID | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    content: str = Field(description="Non-clinical description of observed affect patterns")
    valence_summary: str = Field(description="Summary of valence patterns (e.g., 'generally positive', 'mixed')")
    arousal_summary: str = Field(description="Summary of arousal patterns (e.g., 'calm throughout', 'energetic')")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this summary")
    duration_minutes: float = Field(description="Duration of the observation period")
    stability_evidence: list[str] = Field(default_factory=list, description="Evidence for stability")


class AffectTimelinePoint(BaseModel):
    """Single point in an affect timeline for artifact logging."""

    timestamp: datetime
    valence: float = Field(ge=-1.0, le=1.0)
    arousal: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    raw_valence: float | None = Field(default=None, ge=-1.0, le=1.0, description="Raw unsmoothed valence")
    raw_arousal: float | None = Field(default=None, ge=0.0, le=1.0, description="Raw unsmoothed arousal")
    backend: str = Field(default="unknown")
    strategy_hint: str | None = Field(default=None, description="Any strategy adjustment noted")

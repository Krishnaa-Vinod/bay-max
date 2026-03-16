"""Perception module interfaces for face detection, recognition, and engagement."""

from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID

from baymax.core.enums import EmotionLabel, EngagementLevel


class FaceDetector(ABC):
    """Interface for face detection in frames."""

    @abstractmethod
    def detect(self, frame: Any) -> list[dict[str, Any]]:
        """Detect faces in a frame. Returns list of face bounding boxes."""
        ...


class FaceRecognizer(ABC):
    """Interface for face recognition against enrolled users."""

    @abstractmethod
    def recognize(self, frame: Any, face_bbox: dict[str, Any]) -> UUID | None:
        """Attempt to match a detected face to an enrolled user. Returns user_id or None."""
        ...


class EngagementEstimator(ABC):
    """Interface for estimating user engagement level."""

    @abstractmethod
    def estimate(self, frame: Any, face_bbox: dict[str, Any]) -> EngagementLevel:
        """Estimate the engagement level from a face region."""
        ...


class EmotionEstimator(ABC):
    """Interface for estimating user emotion."""

    @abstractmethod
    def estimate(self, frame: Any, face_bbox: dict[str, Any]) -> EmotionLabel:
        """Estimate the dominant emotion from a face region."""
        ...


class StubFaceDetector(FaceDetector):
    """Returns a single placeholder face detection."""

    def detect(self, frame: Any) -> list[dict[str, Any]]:
        return [{"x": 100, "y": 100, "w": 200, "h": 200, "confidence": 0.95}]


class StubFaceRecognizer(FaceRecognizer):
    """Always returns None (unrecognized)."""

    def recognize(self, frame: Any, face_bbox: dict[str, Any]) -> UUID | None:
        return None


class StubEngagementEstimator(EngagementEstimator):
    """Always returns medium engagement."""

    def estimate(self, frame: Any, face_bbox: dict[str, Any]) -> EngagementLevel:
        return EngagementLevel.MEDIUM


class StubEmotionEstimator(EmotionEstimator):
    """Always returns neutral emotion."""

    def estimate(self, frame: Any, face_bbox: dict[str, Any]) -> EmotionLabel:
        return EmotionLabel.NEUTRAL

"""Perception module interfaces for face detection, recognition, and engagement."""

from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID

import numpy as np

from baymax.core.enums import EmotionLabel, EngagementLevel
from baymax.schemas.perception import DetectedFace, FaceEmbeddingRecord


class FaceDetector(ABC):
    """Interface for face detection in frames."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[DetectedFace]:
        """Detect faces in a BGR/RGB numpy frame.

        Returns list of DetectedFace with bounding boxes and embeddings.
        """
        ...

    @abstractmethod
    def load_model(self) -> None:
        """Load/initialize the detection model."""
        ...


class FaceRecognizer(ABC):
    """Interface for face recognition against enrolled users."""

    @abstractmethod
    def recognize(
        self,
        embedding: list[float],
        enrolled: list[FaceEmbeddingRecord],
        threshold: float = 0.75,
    ) -> tuple[UUID | None, float]:
        """Match a face embedding against enrolled embeddings.

        Returns (user_id, confidence) or (None, best_score).
        """
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


# --- Stub implementations for testing / fallback ---


class StubFaceDetector(FaceDetector):
    """Returns an empty list (no faces detected)."""

    def detect(self, frame: np.ndarray) -> list[DetectedFace]:
        return []

    def load_model(self) -> None:
        pass


class StubFaceRecognizer(FaceRecognizer):
    """Always returns None (unrecognized)."""

    def recognize(
        self,
        embedding: list[float],
        enrolled: list[FaceEmbeddingRecord],
        threshold: float = 0.75,
    ) -> tuple[UUID | None, float]:
        return None, 0.0


class StubEngagementEstimator(EngagementEstimator):
    """Always returns medium engagement."""

    def estimate(self, frame: Any, face_bbox: dict[str, Any]) -> EngagementLevel:
        return EngagementLevel.MEDIUM


class StubEmotionEstimator(EmotionEstimator):
    """Always returns neutral emotion."""

    def estimate(self, frame: Any, face_bbox: dict[str, Any]) -> EmotionLabel:
        return EmotionLabel.NEUTRAL

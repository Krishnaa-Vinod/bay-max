"""Facial affect analysis backend abstraction for Bay-Max."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import cv2
import numpy as np

from baymax.schemas.perception import AffectEvidence, EmotionResult

logger = logging.getLogger(__name__)


class EmotionAnalyzer(ABC):
    """Interface for facial affect analysis backends."""

    @abstractmethod
    def name(self) -> str:
        """Return the backend name."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is available and ready."""
        pass

    @abstractmethod
    def analyze(self, frame: np.ndarray, face_bbox: dict[str, Any]) -> EmotionResult:
        """Analyze facial affect in the given face region.

        Args:
            frame: BGR/RGB numpy frame
            face_bbox: Face bounding box dict with keys: x1, y1, x2, y2

        Returns:
            EmotionResult with valence, arousal, confidence, and evidence
        """
        pass

    @abstractmethod
    def model_info(self) -> dict[str, Any]:
        """Return backend model information."""
        pass

    def load_model(self) -> None:
        """Load/initialize the analysis model. Optional override."""
        pass


class NullEmotionAnalyzer(EmotionAnalyzer):
    """Null backend that returns unknown/neutral affect - always available."""

    def name(self) -> str:
        return "null"

    def is_available(self) -> bool:
        return True

    def analyze(self, frame: np.ndarray, face_bbox: dict[str, Any]) -> EmotionResult:
        """Always returns neutral affect with low confidence."""
        return EmotionResult(
            backend="null",
            confidence=0.0,
            valence=0.0,
            arousal=0.0,
            dominant_emotion_soft_label=None,
            evidence=AffectEvidence(
                processing_method="null_backend",
                processing_time_ms=0.0,
                frame_quality_score=0.0
            ),
            timestamp=datetime.utcnow(),
            success=True
        )

    def model_info(self) -> dict[str, Any]:
        return {
            "backend": "null",
            "model": "NullEmotionAnalyzer",
            "version": "1.0",
            "description": "Always returns neutral affect - testing/fallback backend"
        }


class MediaPipeEmotionAnalyzer(EmotionAnalyzer):
    """MediaPipe Face Landmarker backend using blendshapes for affect estimation."""

    def __init__(self, model_path: str | None = None, device: str = "cpu"):
        """Initialize MediaPipe emotion analyzer.

        Args:
            model_path: Path to MediaPipe face landmarker model (auto-downloads if None)
            device: Device to run on (MediaPipe handles CPU/GPU automatically)
        """
        self.model_path = model_path
        self.device = device
        self._detector = None
        self._model_loaded = False

    def name(self) -> str:
        return "mediapipe"

    def is_available(self) -> bool:
        """Check if MediaPipe is available."""
        try:
            import mediapipe as mp
            return True
        except ImportError:
            return False

    def load_model(self) -> None:
        """Load MediaPipe Face Landmarker model."""
        if self._model_loaded:
            return

        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            # Use default model if no path specified
            if self.model_path is None:
                # MediaPipe will auto-download the model
                options = vision.FaceLandmarkerOptions(
                    base_options=mp_python.BaseOptions(
                        model_asset_path="face_landmarker.task"  # Auto-download
                    ),
                    output_face_blendshapes=True,
                    output_facial_transformation_matrixes=False,
                    num_faces=1
                )
            else:
                options = vision.FaceLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=self.model_path),
                    output_face_blendshapes=True,
                    output_facial_transformation_matrixes=False,
                    num_faces=1
                )

            self._detector = vision.FaceLandmarker.create_from_options(options)
            self._model_loaded = True
            logger.info("MediaPipe Face Landmarker loaded successfully")

        except Exception as e:
            logger.error("Failed to load MediaPipe Face Landmarker: %s", e)
            self._detector = None
            self._model_loaded = False
            raise

    def analyze(self, frame: np.ndarray, face_bbox: dict[str, Any]) -> EmotionResult:
        """Analyze affect using MediaPipe Face Landmarker blendshapes."""
        start_time = datetime.utcnow()

        if not self._model_loaded or self._detector is None:
            try:
                self.load_model()
            except Exception as e:
                logger.error("Failed to load model for analysis: %s", e)
                return self._failed_result(f"Model load error: {e}")

        try:
            import mediapipe as mp

            # Convert BGR to RGB if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame = frame

            # Extract face region
            x1, y1 = int(face_bbox.get("x1", 0)), int(face_bbox.get("y1", 0))
            x2, y2 = int(face_bbox.get("x2", frame.shape[1])), int(face_bbox.get("y2", frame.shape[0]))

            # Ensure valid bounds
            h, w = frame.shape[:2]
            x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
            y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))

            if x2 <= x1 or y2 <= y1:
                return self._failed_result("Invalid face bounding box")

            # Create MediaPipe Image from the full frame (landmarker works better with full context)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Detect face landmarks and blendshapes
            result = self._detector.detect(mp_image)

            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            if not result.face_landmarks or not result.face_blendshapes:
                return EmotionResult(
                    backend="mediapipe",
                    confidence=0.0,
                    valence=0.0,
                    arousal=0.0,
                    dominant_emotion_soft_label=None,
                    evidence=AffectEvidence(
                        processing_method="mediapipe_face_landmarker",
                        processing_time_ms=processing_time,
                        frame_quality_score=0.0
                    ),
                    timestamp=datetime.utcnow(),
                    success=True
                )

            # Extract blendshapes
            face_blendshapes = result.face_blendshapes[0]  # First face
            blendshape_dict = {
                category.category_name: category.score
                for category in face_blendshapes
            }

            # Estimate valence and arousal from blendshapes
            valence, arousal, confidence, soft_label = self._estimate_affect_from_blendshapes(blendshape_dict)

            # Calculate face region quality score
            face_quality = self._calculate_face_quality(blendshape_dict, face_bbox, frame.shape)

            return EmotionResult(
                backend="mediapipe",
                confidence=confidence,
                valence=valence,
                arousal=arousal,
                dominant_emotion_soft_label=soft_label,
                evidence=AffectEvidence(
                    blendshapes=blendshape_dict,
                    processing_method="mediapipe_face_landmarker",
                    processing_time_ms=processing_time,
                    frame_quality_score=face_quality
                ),
                timestamp=datetime.utcnow(),
                success=True
            )

        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.error("MediaPipe emotion analysis failed: %s", e)
            return self._failed_result(f"Analysis error: {e}", processing_time)

    def _estimate_affect_from_blendshapes(self, blendshapes: dict[str, float]) -> tuple[float, float, float, str | None]:
        """Estimate valence/arousal from MediaPipe blendshapes.

        This uses a simple heuristic approach based on known relationships between
        facial expressions and emotional dimensions. Not clinically validated.
        """
        # Positive affect indicators
        smile_left = blendshapes.get("mouthSmileLeft", 0.0)
        smile_right = blendshapes.get("mouthSmileRight", 0.0)
        cheek_puff = blendshapes.get("cheekPuff", 0.0)
        eye_squint_left = blendshapes.get("eyeSquintLeft", 0.0)
        eye_squint_right = blendshapes.get("eyeSquintRight", 0.0)

        # Negative affect indicators
        frown_left = blendshapes.get("mouthFrownLeft", 0.0)
        frown_right = blendshapes.get("mouthFrownRight", 0.0)
        mouth_lower_down_left = blendshapes.get("mouthLowerDownLeft", 0.0)
        mouth_lower_down_right = blendshapes.get("mouthLowerDownRight", 0.0)

        # Arousal indicators
        eye_wide_left = blendshapes.get("eyeWideLeft", 0.0)
        eye_wide_right = blendshapes.get("eyeWideRight", 0.0)
        jaw_open = blendshapes.get("jawOpen", 0.0)
        mouth_open = blendshapes.get("mouthOpen", 0.0)

        # Calculate valence (-1 to +1)
        positive_cues = (smile_left + smile_right) / 2.0 + cheek_puff * 0.5
        positive_cues += (eye_squint_left + eye_squint_right) * 0.3  # Duchenne-like

        negative_cues = (frown_left + frown_right) / 2.0
        negative_cues += (mouth_lower_down_left + mouth_lower_down_right) / 2.0

        valence = positive_cues - negative_cues
        valence = max(-1.0, min(1.0, valence))  # Clamp to [-1, 1]

        # Calculate arousal (0 to 1)
        arousal_cues = (eye_wide_left + eye_wide_right) / 2.0
        arousal_cues += jaw_open * 0.7 + mouth_open * 0.5
        arousal_cues += abs(valence) * 0.3  # Strong emotions tend to be higher arousal

        arousal = max(0.0, min(1.0, arousal_cues))

        # Calculate confidence based on strength of indicators
        max_positive = max(smile_left, smile_right, cheek_puff)
        max_negative = max(frown_left, frown_right, mouth_lower_down_left, mouth_lower_down_right)
        max_arousal = max(eye_wide_left, eye_wide_right, jaw_open, mouth_open)

        confidence = max(max_positive, max_negative, max_arousal)
        confidence = max(0.0, min(1.0, confidence))

        # Generate soft label (non-clinical)
        soft_label = None
        if confidence > 0.4:
            if valence > 0.3 and arousal > 0.4:
                soft_label = "appears energetic and positive"
            elif valence > 0.3 and arousal < 0.3:
                soft_label = "appears calm and content"
            elif valence < -0.3 and arousal > 0.4:
                soft_label = "appears tense or concerned"
            elif valence < -0.3 and arousal < 0.3:
                soft_label = "appears subdued or thoughtful"
            elif arousal < 0.2:
                soft_label = "appears calm"
            elif arousal > 0.6:
                soft_label = "appears engaged and alert"

        return valence, arousal, confidence, soft_label

    def _calculate_face_quality(self, blendshapes: dict[str, float], face_bbox: dict[str, Any], frame_shape: tuple) -> float:
        """Calculate a quality score for the face region."""
        # Simple quality heuristic based on face size and visibility
        bbox_width = face_bbox.get("x2", 0) - face_bbox.get("x1", 0)
        bbox_height = face_bbox.get("y2", 0) - face_bbox.get("y1", 0)

        frame_area = frame_shape[0] * frame_shape[1]
        face_area = bbox_width * bbox_height
        face_ratio = face_area / frame_area if frame_area > 0 else 0

        # Prefer faces that are not too small or too large
        size_quality = 1.0 - abs(face_ratio - 0.15)  # Optimal around 15% of frame
        size_quality = max(0.0, min(1.0, size_quality))

        # Check for neutral positioning (not extremely rotated)
        head_rotation_indicators = [
            blendshapes.get("headYaw", 0.0),
            blendshapes.get("headPitch", 0.0),
            blendshapes.get("headRoll", 0.0)
        ]
        rotation_quality = 1.0 - min(1.0, sum(abs(r) for r in head_rotation_indicators) / 3.0)

        return (size_quality + rotation_quality) / 2.0

    def _failed_result(self, error_message: str, processing_time: float = 0.0) -> EmotionResult:
        """Create a failed analysis result."""
        return EmotionResult(
            backend="mediapipe",
            confidence=0.0,
            valence=0.0,
            arousal=0.0,
            dominant_emotion_soft_label=None,
            evidence=AffectEvidence(
                processing_method="mediapipe_face_landmarker_failed",
                processing_time_ms=processing_time,
                frame_quality_score=0.0
            ),
            timestamp=datetime.utcnow(),
            success=False
        )

    def model_info(self) -> dict[str, Any]:
        return {
            "backend": "mediapipe",
            "model": "Face Landmarker",
            "version": "MediaPipe 0.10.32+",
            "description": "MediaPipe Face Landmarker with blendshape-based affect estimation",
            "device": self.device,
            "model_path": self.model_path or "auto-download"
        }


def get_emotion_analyzer(backend: str = "null", **kwargs) -> EmotionAnalyzer:
    """Factory function to create emotion analyzer backends.

    Args:
        backend: Backend name ("null", "mediapipe")
        **kwargs: Backend-specific arguments

    Returns:
        EmotionAnalyzer instance

    Raises:
        ValueError: If backend is unknown or unavailable
    """
    if backend == "null":
        return NullEmotionAnalyzer()
    elif backend == "mediapipe":
        analyzer = MediaPipeEmotionAnalyzer(**kwargs)
        if not analyzer.is_available():
            raise ValueError("MediaPipe backend is not available (import failed)")
        return analyzer
    else:
        raise ValueError(f"Unknown emotion backend: {backend}")
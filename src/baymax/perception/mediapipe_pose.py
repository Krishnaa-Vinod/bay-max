"""MediaPipe Pose Landmarker adapter for pose estimation."""

import logging

import numpy as np

from baymax.schemas.perception import PoseLandmark2D, PoseResult

from .pose_interface import PoseEstimator

logger = logging.getLogger(__name__)

# MediaPipe landmark names for the 33-point Pose model
_MP_LANDMARK_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]


class MediaPipePoseEstimator(PoseEstimator):
    """Pose estimation using MediaPipe Pose.

    Uses the legacy mp.solutions.pose API which is widely available
    and works in image mode for single-frame analysis.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ) -> None:
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._model_complexity = model_complexity
        self._pose = None

    def load_model(self) -> None:
        """Initialize MediaPipe Pose."""
        try:
            import mediapipe as mp

            self._pose = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=self._model_complexity,
                min_detection_confidence=self._min_detection_confidence,
                min_tracking_confidence=self._min_tracking_confidence,
            )
            logger.info("MediaPipe Pose loaded (complexity=%d)", self._model_complexity)
        except ImportError:
            logger.error("mediapipe is not installed. Install with: pip install mediapipe")
            raise
        except Exception:
            logger.error("Failed to initialize MediaPipe Pose", exc_info=True)
            raise

    def estimate(self, frame: np.ndarray) -> PoseResult:
        """Estimate pose from an RGB numpy frame.

        Args:
            frame: H x W x 3 numpy array in RGB format.

        Returns:
            PoseResult with landmarks if a person is detected,
            otherwise PoseResult(pose_present=False).
        """
        if self._pose is None:
            self.load_model()

        try:
            results = self._pose.process(frame)
        except Exception:
            logger.warning("MediaPipe Pose processing failed", exc_info=True)
            return PoseResult(pose_present=False, backend="mediapipe")

        if results.pose_landmarks is None:
            return PoseResult(pose_present=False, backend="mediapipe")

        landmarks = []
        total_visibility = 0.0
        for i, lm in enumerate(results.pose_landmarks.landmark):
            name = _MP_LANDMARK_NAMES[i] if i < len(_MP_LANDMARK_NAMES) else f"landmark_{i}"
            landmarks.append(
                PoseLandmark2D(
                    x=lm.x,
                    y=lm.y,
                    visibility=lm.visibility,
                    name=name,
                )
            )
            total_visibility += lm.visibility

        avg_visibility = total_visibility / len(landmarks) if landmarks else 0.0

        return PoseResult(
            pose_present=True,
            landmarks=landmarks,
            landmark_count=len(landmarks),
            confidence=round(avg_visibility, 4),
            backend="mediapipe",
            model_name="mp_pose_legacy",
        )

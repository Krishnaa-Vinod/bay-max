"""Pose estimation interface and stub implementation."""

from abc import ABC, abstractmethod

import numpy as np

from baymax.schemas.perception import PoseResult


class PoseEstimator(ABC):
    """Interface for pose estimation from image frames.

    Implementations should return structured PoseResult objects.
    The interface is backend-agnostic to allow swapping MediaPipe
    for RTMPose or other backends in the future.
    """

    @abstractmethod
    def estimate(self, frame: np.ndarray) -> PoseResult:
        """Estimate body pose from a BGR/RGB numpy frame.

        Args:
            frame: H x W x 3 numpy array.

        Returns:
            PoseResult with landmarks and confidence.
            If no person is detected, returns PoseResult(pose_present=False).
        """
        ...

    @abstractmethod
    def load_model(self) -> None:
        """Load/initialize the pose estimation model."""
        ...


class StubPoseEstimator(PoseEstimator):
    """Returns empty pose result (no pose detected). Used for testing/fallback."""

    def estimate(self, frame: np.ndarray) -> PoseResult:
        return PoseResult(pose_present=False, backend="stub")

    def load_model(self) -> None:
        pass

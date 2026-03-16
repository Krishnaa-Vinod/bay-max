"""Annotation utilities for drawing debug overlays on frames.

Draws face bounding boxes, track IDs, pose landmarks, and body-state labels
onto images for visual debugging and local manual inspection.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from baymax.schemas.perception import (
    AnnotatedArtifactRef,
    EngagementResult,
    PoseResult,
    RecognizedFace,
    UnknownFace,
)

logger = logging.getLogger(__name__)

# MediaPipe Pose skeleton connections for drawing
_POSE_CONNECTIONS = [
    (11, 12),  # shoulders
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    (11, 23), (12, 24),  # torso sides
    (23, 24),  # hips
    (23, 25), (25, 27),  # left leg
    (24, 26), (26, 28),  # right leg
    (0, 11), (0, 12),  # nose to shoulders (approx)
]


def draw_annotations(
    frame: np.ndarray,
    recognized_faces: list[RecognizedFace] | None = None,
    unknown_faces: list[UnknownFace] | None = None,
    pose_result: PoseResult | None = None,
    engagement: EngagementResult | None = None,
) -> np.ndarray:
    """Draw debug annotations onto a frame.

    Args:
        frame: RGB numpy array (H, W, 3).
        recognized_faces: Recognized faces with bounding boxes.
        unknown_faces: Unknown faces with bounding boxes.
        pose_result: Pose estimation result with landmarks.
        engagement: Engagement heuristic result.

    Returns:
        Annotated copy of the frame.
    """
    try:
        import cv2
    except ImportError:
        logger.warning("opencv-python not installed, skipping annotations")
        return frame

    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Draw recognized face boxes (green)
    if recognized_faces:
        for face in recognized_faces:
            x1, y1, x2, y2 = (
                int(face.bbox.x1), int(face.bbox.y1),
                int(face.bbox.x2), int(face.bbox.y2),
            )
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = face.user_display_name or str(face.user_id)[:8]
            conf_str = f"{face.match_confidence:.2f}"
            cv2.putText(
                annotated, f"{label} ({conf_str})",
                (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )

    # Draw unknown face boxes (red)
    if unknown_faces:
        for face in unknown_faces:
            x1, y1, x2, y2 = (
                int(face.bbox.x1), int(face.bbox.y1),
                int(face.bbox.x2), int(face.bbox.y2),
            )
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                annotated, "Unknown",
                (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
            )

    # Draw pose landmarks and skeleton
    if pose_result and pose_result.pose_present and pose_result.landmarks:
        lms = pose_result.landmarks
        # Draw landmarks as circles
        for lm in lms:
            if lm.visibility >= 0.5:
                px, py = int(lm.x * w), int(lm.y * h)
                cv2.circle(annotated, (px, py), 3, (255, 255, 0), -1)

        # Draw skeleton connections
        for i, j in _POSE_CONNECTIONS:
            if i < len(lms) and j < len(lms):
                lm_a, lm_b = lms[i], lms[j]
                if lm_a.visibility >= 0.5 and lm_b.visibility >= 0.5:
                    pa = (int(lm_a.x * w), int(lm_a.y * h))
                    pb = (int(lm_b.x * w), int(lm_b.y * h))
                    cv2.line(annotated, pa, pb, (255, 255, 0), 1)

    # Draw body-state labels
    if engagement:
        labels = [
            f"Engagement: {engagement.level} ({engagement.score:.2f})",
            f"Posture: {engagement.posture}",
            f"Lean: {engagement.lean}",
            f"Motion: {engagement.motion}",
        ]
        y_offset = 25
        for label in labels:
            cv2.putText(
                annotated, label,
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
            )
            cv2.putText(
                annotated, label,
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1,
            )
            y_offset += 22

    return annotated


def save_annotated_artifact(
    annotated_frame: np.ndarray,
    artifact_dir: str,
    session_id: str | None = None,
    prefix: str = "frame",
) -> AnnotatedArtifactRef | None:
    """Save an annotated frame to the artifact directory.

    Args:
        annotated_frame: Annotated RGB numpy array.
        artifact_dir: Directory to save artifacts to.
        session_id: Optional session ID for the filename.
        prefix: Filename prefix.

    Returns:
        AnnotatedArtifactRef with the saved file path, or None if save failed.
    """
    try:
        import cv2
    except ImportError:
        logger.warning("opencv-python not installed, cannot save annotations")
        return None

    try:
        artifact_path = Path(artifact_dir)
        artifact_path.mkdir(parents=True, exist_ok=True)

        timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        sid_part = f"_{session_id[:8]}" if session_id else ""
        filename = f"{prefix}{sid_part}_{timestamp_str}.jpg"
        filepath = artifact_path / filename

        # Convert RGB to BGR for cv2 saving
        bgr_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(filepath), bgr_frame)
        logger.info("Saved annotated artifact: %s", filepath)

        return AnnotatedArtifactRef(
            path=str(filepath),
            artifact_type="annotated_frame",
        )
    except Exception:
        logger.warning("Failed to save annotated artifact", exc_info=True)
        return None

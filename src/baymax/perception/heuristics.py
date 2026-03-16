"""Body-state and engagement heuristics.

All heuristics in this module are simple, deterministic, rule-based functions.
They use transparent thresholds documented in docs/ENGAGEMENT_HEURISTICS.md.
No clinical or diagnostic claims are made.

See docs/ENGAGEMENT_HEURISTICS.md for detailed formulas and rationale.
"""

from __future__ import annotations

import logging
from collections import deque

from baymax.core.enums import EngagementLevel, LeanLabel, MotionLevel, PostureLabel
from baymax.schemas.perception import EngagementResult, PoseLandmark2D, PoseResult

logger = logging.getLogger(__name__)

# --- Landmark index constants (MediaPipe 33-point Pose) ---
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_HIP = 23
_RIGHT_HIP = 24
_NOSE = 0
_LEFT_EAR = 7
_RIGHT_EAR = 8

# --- Thresholds (documented in docs/ENGAGEMENT_HEURISTICS.md) ---
POSTURE_SLOUCH_RATIO_THRESHOLD = 0.38
POSTURE_RECLINE_RATIO_THRESHOLD = 0.15
LEAN_FORWARD_OFFSET = -0.03
LEAN_BACKWARD_OFFSET = 0.04
MOTION_HIGH_THRESHOLD = 0.04
MOTION_MEDIUM_THRESHOLD = 0.015
MIN_VISIBILITY = 0.5


def _get_landmark(
    landmarks: list[PoseLandmark2D], index: int
) -> PoseLandmark2D | None:
    """Get a landmark by index if it exists and is sufficiently visible."""
    if index >= len(landmarks):
        return None
    lm = landmarks[index]
    if lm.visibility < MIN_VISIBILITY:
        return None
    return lm


def compute_posture(pose: PoseResult) -> PostureLabel:
    """Compute posture label from pose landmarks.

    Uses the vertical distance ratio between shoulder-midpoint and hip-midpoint
    relative to the overall torso length. A short ratio suggests slouching;
    a very short or negative ratio suggests reclining.

    Returns PostureLabel.UNKNOWN if landmarks are insufficient.
    """
    if not pose.pose_present or not pose.landmarks:
        return PostureLabel.UNKNOWN

    ls = _get_landmark(pose.landmarks, _LEFT_SHOULDER)
    rs = _get_landmark(pose.landmarks, _RIGHT_SHOULDER)
    lh = _get_landmark(pose.landmarks, _LEFT_HIP)
    rh = _get_landmark(pose.landmarks, _RIGHT_HIP)

    if not all([ls, rs, lh, rh]):
        return PostureLabel.UNKNOWN

    # Midpoints in normalized coordinates (y increases downward)
    shoulder_mid_y = (ls.y + rs.y) / 2
    hip_mid_y = (lh.y + rh.y) / 2

    # Torso vertical extent (hip is lower = larger y)
    torso_height = hip_mid_y - shoulder_mid_y

    if torso_height <= 0.01:
        # Shoulders at or below hips - reclining or lying down
        return PostureLabel.RECLINED

    # Nose position relative to shoulders for forward-head detection
    nose = _get_landmark(pose.landmarks, _NOSE)
    if nose is not None:
        # Ratio of nose-to-shoulder-midpoint vs torso height
        nose_shoulder_ratio = (shoulder_mid_y - nose.y) / torso_height
        if nose_shoulder_ratio < POSTURE_RECLINE_RATIO_THRESHOLD:
            return PostureLabel.RECLINED
        if nose_shoulder_ratio < POSTURE_SLOUCH_RATIO_THRESHOLD:
            return PostureLabel.SLOUCHED
        return PostureLabel.UPRIGHT

    # Without nose, just use torso proportions
    # Short torso relative to image = likely slouched
    if torso_height < 0.08:
        return PostureLabel.SLOUCHED
    return PostureLabel.UPRIGHT


def compute_lean(pose: PoseResult) -> LeanLabel:
    """Compute lean direction from pose landmarks.

    Compares the horizontal shoulder midpoint to the horizontal hip midpoint.
    If shoulders are forward (smaller x for left-facing, depends on orientation)
    relative to hips, this indicates a forward lean.

    Uses the nose position relative to shoulder line as a simpler proxy.
    Returns LeanLabel.UNKNOWN if landmarks are insufficient.
    """
    if not pose.pose_present or not pose.landmarks:
        return LeanLabel.UNKNOWN

    nose = _get_landmark(pose.landmarks, _NOSE)
    ls = _get_landmark(pose.landmarks, _LEFT_SHOULDER)
    rs = _get_landmark(pose.landmarks, _RIGHT_SHOULDER)
    lh = _get_landmark(pose.landmarks, _LEFT_HIP)
    rh = _get_landmark(pose.landmarks, _RIGHT_HIP)

    if not all([ls, rs, lh, rh]):
        return LeanLabel.UNKNOWN

    shoulder_mid_y = (ls.y + rs.y) / 2
    hip_mid_y = (lh.y + rh.y) / 2
    torso_height = hip_mid_y - shoulder_mid_y

    if torso_height <= 0.01:
        return LeanLabel.UNKNOWN

    if nose is not None:
        # Forward lean: nose y relatively high compared to shoulder midpoint
        # (i.e. head is tilted forward / towards camera)
        # In normalized coords, a person leaning forward has their nose
        # shift toward the shoulder line
        nose_offset = (nose.y - shoulder_mid_y) / torso_height
        if nose_offset > LEAN_BACKWARD_OFFSET:
            return LeanLabel.BACKWARD
        if nose_offset < LEAN_FORWARD_OFFSET:
            return LeanLabel.FORWARD
        return LeanLabel.NEUTRAL

    return LeanLabel.UNKNOWN


class MotionTracker:
    """Tracks motion level over recent frames using landmark displacement.

    Maintains a sliding window of recent landmark positions and computes
    average displacement between consecutive frames.
    """

    def __init__(self, window_size: int = 10) -> None:
        self._window_size = window_size
        self._history: deque[list[tuple[float, float]]] = deque(maxlen=window_size)

    def update(self, pose: PoseResult) -> MotionLevel:
        """Update with new pose and return current motion level.

        Uses the mean displacement of key body landmarks (shoulders, hips, nose)
        between consecutive frames.
        """
        if not pose.pose_present or not pose.landmarks:
            return MotionLevel.UNKNOWN

        key_indices = [_NOSE, _LEFT_SHOULDER, _RIGHT_SHOULDER, _LEFT_HIP, _RIGHT_HIP]
        current_points = []
        for idx in key_indices:
            lm = _get_landmark(pose.landmarks, idx)
            if lm is not None:
                current_points.append((lm.x, lm.y))
            else:
                current_points.append(None)

        self._history.append(current_points)

        if len(self._history) < 2:
            return MotionLevel.UNKNOWN

        # Compute displacement between last two frames
        prev = self._history[-2]
        curr = self._history[-1]
        displacements = []
        for p, c in zip(prev, curr):
            if p is not None and c is not None:
                dx = c[0] - p[0]
                dy = c[1] - p[1]
                displacements.append((dx**2 + dy**2) ** 0.5)

        if not displacements:
            return MotionLevel.UNKNOWN

        avg_displacement = sum(displacements) / len(displacements)

        if avg_displacement >= MOTION_HIGH_THRESHOLD:
            return MotionLevel.HIGH
        if avg_displacement >= MOTION_MEDIUM_THRESHOLD:
            return MotionLevel.MEDIUM
        return MotionLevel.LOW

    def clear(self) -> None:
        """Clear motion history."""
        self._history.clear()


def compute_engagement(
    *,
    posture: PostureLabel,
    lean: LeanLabel,
    motion: MotionLevel,
    face_visible: bool,
    face_recognized: bool,
    pose_visible: bool,
) -> EngagementResult:
    """Compute a simple engagement level from available signals.

    Scoring (0.0 - 1.0):
    - Face visible:     +0.20
    - Face recognized:  +0.10
    - Pose visible:     +0.10
    - Upright posture:  +0.20;  Slouched: +0.05;  Reclined: +0.0
    - Forward lean:     +0.15;  Neutral:  +0.10;  Backward: +0.0
    - Low motion:       +0.15;  Medium:   +0.10;  High: +0.05
    (Unknown values for posture/lean/motion contribute 0.0)

    Thresholds:
    - score >= 0.65 -> HIGH
    - score >= 0.35 -> MEDIUM
    - else          -> LOW

    See docs/ENGAGEMENT_HEURISTICS.md for full documentation.
    """
    score = 0.0
    evidence = []

    if face_visible:
        score += 0.20
        evidence.append("face_visible")
    if face_recognized:
        score += 0.10
        evidence.append("face_recognized")
    if pose_visible:
        score += 0.10
        evidence.append("pose_visible")

    if posture == PostureLabel.UPRIGHT:
        score += 0.20
        evidence.append("posture=upright")
    elif posture == PostureLabel.SLOUCHED:
        score += 0.05
        evidence.append("posture=slouched")
    elif posture == PostureLabel.RECLINED:
        evidence.append("posture=reclined")

    if lean == LeanLabel.FORWARD:
        score += 0.15
        evidence.append("lean=forward")
    elif lean == LeanLabel.NEUTRAL:
        score += 0.10
        evidence.append("lean=neutral")
    elif lean == LeanLabel.BACKWARD:
        evidence.append("lean=backward")

    if motion == MotionLevel.LOW:
        score += 0.15
        evidence.append("motion=low")
    elif motion == MotionLevel.MEDIUM:
        score += 0.10
        evidence.append("motion=medium")
    elif motion == MotionLevel.HIGH:
        score += 0.05
        evidence.append("motion=high")

    score = min(score, 1.0)

    if score >= 0.65:
        level = EngagementLevel.HIGH
    elif score >= 0.35:
        level = EngagementLevel.MEDIUM
    else:
        level = EngagementLevel.LOW

    return EngagementResult(
        level=level,
        posture=posture,
        lean=lean,
        motion=motion,
        face_visible=face_visible,
        face_recognized=face_recognized,
        pose_visible=pose_visible,
        score=round(score, 4),
        evidence=evidence,
    )

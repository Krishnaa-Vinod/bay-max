"""Live overlay renderer for annotating frames with runtime state."""

import logging

import numpy as np

from baymax.live.schemas import LiveRuntimeStatus

logger = logging.getLogger(__name__)


def render_overlay(
    frame: np.ndarray,
    status: LiveRuntimeStatus,
    debug: bool = False,
    backend: str = "",
    model_name: str = "",
    memory_refs_count: int = 0,
    posture: str = "unknown",
    engagement: str = "medium",
    engagement_score: float = 0.5,
    identity_confidence: float = 0.0,
) -> np.ndarray:
    """Draw a status overlay on the frame.

    Returns a copy of the frame with overlay text.
    """
    try:
        import cv2
    except ImportError:
        return frame

    overlay = frame.copy()
    h, w = overlay.shape[:2]

    # Semi-transparent background strip at the top
    bar_height = 120 if debug else 80
    sub_img = overlay[0:bar_height, 0:w]
    dark = np.zeros_like(sub_img)
    cv2.addWeighted(sub_img, 0.4, dark, 0.6, 0, sub_img)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    color = (255, 255, 255)
    thin = 1
    y_start = 18

    # Line 1: Session status + user
    session_text = f"Session: {status.session_status.value}"
    user_text = status.current_user_display_name or (
        str(status.current_user_id)[:8] if status.current_user_id else "none"
    )
    line1 = f"{session_text} | User: {user_text} (conf: {identity_confidence:.2f})"
    cv2.putText(overlay, line1, (10, y_start), font, scale, color, thin)

    # Line 2: Posture + engagement
    line2 = f"Posture: {posture} | Engagement: {engagement} ({engagement_score:.2f})"
    cv2.putText(overlay, line2, (10, y_start + 22), font, scale, color, thin)

    # Line 3: Last event + response
    event_text = status.last_event or "none"
    line3 = f"Event: {event_text}"
    cv2.putText(overlay, line3, (10, y_start + 44), font, scale, (0, 255, 255), thin)

    # Line 4: Last response (truncated)
    resp_text = (status.last_response_text or "")[:80]
    if resp_text:
        cv2.putText(overlay, resp_text, (10, y_start + 66), font, scale * 0.9, (0, 255, 0), thin)

    # Debug info
    if debug:
        dbg = f"Backend: {backend} | Model: {model_name} | Memories: {memory_refs_count}"
        cv2.putText(overlay, dbg, (10, y_start + 88), font, scale * 0.8, (200, 200, 200), thin)

    # Bottom bar with frame count
    bottom_y = h - 10
    frame_info = f"Frame: {status.frame_count} | Analyses: {status.analysis_count}"
    cv2.putText(overlay, frame_info, (10, bottom_y), font, scale * 0.8, (180, 180, 180), thin)

    return overlay

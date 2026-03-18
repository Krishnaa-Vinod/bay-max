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
    # Increase height if debug mode to fit affect info
    bar_height = 150 if debug else 80
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

        # Affect analysis info (Iteration 009)
        if status.affect_enabled:
            affect_line = (
                f"Affect: v={status.emotion_valence:+.2f} "
                f"a={status.emotion_arousal:.2f} "
                f"conf={status.emotion_confidence:.2f} "
                f"stable={status.emotion_stable_duration_sec:.0f}s "
                f"[{status.affect_backend}]"
            )
            # Color based on valence
            if status.emotion_confidence >= 0.5:
                if status.emotion_valence > 0.2:
                    affect_color = (100, 255, 100)  # Green for positive
                elif status.emotion_valence < -0.2:
                    affect_color = (100, 100, 255)  # Blue for subdued
                else:
                    affect_color = (200, 200, 200)  # Gray for neutral
            else:
                affect_color = (150, 150, 150)  # Dimmed for low confidence

            cv2.putText(
                overlay, affect_line, (10, y_start + 110),
                font, scale * 0.8, affect_color, thin
            )

            # Add debug summary if available
            if status.emotion_debug_summary:
                cv2.putText(
                    overlay,
                    f"[{status.emotion_debug_summary}]",
                    (10, y_start + 128),
                    font, scale * 0.7, affect_color, thin
                )
        else:
            cv2.putText(
                overlay,
                "Affect: disabled",
                (10, y_start + 110),
                font, scale * 0.8, (100, 100, 100), thin
            )

    # Bottom bar with frame count
    bottom_y = h - 10
    frame_info = f"Frame: {status.frame_count} | Analyses: {status.analysis_count}"
    cv2.putText(overlay, frame_info, (10, bottom_y), font, scale * 0.8, (180, 180, 180), thin)

    return overlay

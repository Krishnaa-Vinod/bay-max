"""Affect state smoothing and stability tracking for Bay-Max."""

import logging
from datetime import datetime, timedelta
from typing import Any

from baymax.schemas.perception import EmotionResult, SmoothedAffectState

logger = logging.getLogger(__name__)


class AffectSmoother:
    """Exponential Moving Average smoother for affect state with stability tracking."""

    def __init__(
        self,
        alpha: float = 0.25,
        stability_threshold: float = 0.1,
        stability_duration_sec: float = 30.0,
        confidence_threshold: float = 0.6
    ):
        """Initialize affect smoother.

        Args:
            alpha: EMA smoothing factor (0=no smoothing, 1=keep previous)
            stability_threshold: Max change per update to consider "stable"
            stability_duration_sec: Duration state must be stable before marking as stable
            confidence_threshold: Minimum confidence to process updates
        """
        self.alpha = alpha
        self.stability_threshold = stability_threshold
        self.stability_duration_sec = stability_duration_sec
        self.confidence_threshold = confidence_threshold

        # Current smoothed state
        self._state = SmoothedAffectState(
            valence=0.0,
            arousal=0.0,
            confidence=0.0,
            stable_duration_sec=0.0,
            last_update=datetime.utcnow(),
            enabled=True,
            sample_count=0
        )

        # Stability tracking
        self._stability_start_time: datetime | None = None
        self._last_stable_valence: float = 0.0
        self._last_stable_arousal: float = 0.0

    def update(self, emotion_result: EmotionResult) -> SmoothedAffectState:
        """Update smoothed state with new emotion result.

        Args:
            emotion_result: New affect analysis result

        Returns:
            Updated smoothed affect state
        """
        now = datetime.utcnow()

        # Skip update if confidence too low
        if emotion_result.confidence < self.confidence_threshold:
            logger.debug(
                "Skipping affect update due to low confidence: %.3f < %.3f",
                emotion_result.confidence,
                self.confidence_threshold
            )
            return self._state

        # Skip if backend failed
        if not emotion_result.success:
            logger.debug("Skipping affect update due to failed analysis")
            return self._state

        # First update - initialize state
        if self._state.sample_count == 0:
            self._state.valence = emotion_result.valence
            self._state.arousal = emotion_result.arousal
            self._state.confidence = emotion_result.confidence
            self._state.sample_count = 1
            self._state.last_update = now
            self._state.stable_duration_sec = 0.0

            # Initialize stability tracking
            self._stability_start_time = now
            self._last_stable_valence = emotion_result.valence
            self._last_stable_arousal = emotion_result.arousal

            logger.debug(
                "Initialized affect state: valence=%.3f, arousal=%.3f, confidence=%.3f",
                self._state.valence, self._state.arousal, self._state.confidence
            )
            return self._state

        # Apply exponential moving average
        prev_valence = self._state.valence
        prev_arousal = self._state.arousal
        prev_confidence = self._state.confidence

        # EMA Update: new_value = alpha * new + (1 - alpha) * old
        self._state.valence = self.alpha * emotion_result.valence + (1 - self.alpha) * prev_valence
        self._state.arousal = self.alpha * emotion_result.arousal + (1 - self.alpha) * prev_arousal
        self._state.confidence = self.alpha * emotion_result.confidence + (1 - self.alpha) * prev_confidence

        # Update metadata
        self._state.sample_count += 1
        self._state.last_update = now

        # Check stability
        self._update_stability_tracking(now)

        logger.debug(
            "Updated affect state: valence=%.3f->%.3f, arousal=%.3f->%.3f, "
            "confidence=%.3f->%.3f, stable_duration=%.1fs",
            prev_valence, self._state.valence,
            prev_arousal, self._state.arousal,
            prev_confidence, self._state.confidence,
            self._state.stable_duration_sec
        )

        return self._state

    def _update_stability_tracking(self, now: datetime) -> None:
        """Update stability duration based on recent changes."""
        # Calculate change from previously stable state
        valence_change = abs(self._state.valence - self._last_stable_valence)
        arousal_change = abs(self._state.arousal - self._last_stable_arousal)
        max_change = max(valence_change, arousal_change)

        # If change is within threshold, continue/start stability period
        if max_change <= self.stability_threshold:
            if self._stability_start_time is None:
                # Start new stability period
                self._stability_start_time = now
                self._last_stable_valence = self._state.valence
                self._last_stable_arousal = self._state.arousal
                self._state.stable_duration_sec = 0.0
            else:
                # Continue stability period
                duration = (now - self._stability_start_time).total_seconds()
                self._state.stable_duration_sec = duration
        else:
            # Significant change detected - reset stability
            logger.debug(
                "Affect state change detected: valence_change=%.3f, arousal_change=%.3f, max=%.3f",
                valence_change, arousal_change, max_change
            )
            self._stability_start_time = now
            self._last_stable_valence = self._state.valence
            self._last_stable_arousal = self._state.arousal
            self._state.stable_duration_sec = 0.0

    def get_current_state(self) -> SmoothedAffectState:
        """Get current smoothed affect state."""
        return self._state.model_copy()

    def is_stable(self, min_duration_sec: float | None = None) -> bool:
        """Check if current state has been stable for minimum duration.

        Args:
            min_duration_sec: Minimum duration (defaults to configured value)

        Returns:
            True if state has been stable for the required duration
        """
        min_duration = min_duration_sec or self.stability_duration_sec
        return self._state.stable_duration_sec >= min_duration

    def has_sufficient_confidence(self, min_confidence: float | None = None) -> bool:
        """Check if current state has sufficient confidence.

        Args:
            min_confidence: Minimum confidence (defaults to configured threshold)

        Returns:
            True if confidence meets threshold
        """
        min_conf = min_confidence or self.confidence_threshold
        return self._state.confidence >= min_conf

    def reset(self) -> None:
        """Reset smoother to initial state."""
        self._state = SmoothedAffectState(
            valence=0.0,
            arousal=0.0,
            confidence=0.0,
            stable_duration_sec=0.0,
            last_update=datetime.utcnow(),
            enabled=True,
            sample_count=0
        )
        self._stability_start_time = None
        self._last_stable_valence = 0.0
        self._last_stable_arousal = 0.0
        logger.info("Affect smoother reset")

    def get_affect_summary(self, duration_minutes: float) -> dict[str, Any]:
        """Generate a summary of current affect state for memory consolidation.

        Args:
            duration_minutes: Duration of the observation period

        Returns:
            Dictionary with affect summary information
        """
        state = self.get_current_state()

        # Generate valence summary
        if state.valence > 0.3:
            valence_summary = "generally positive"
        elif state.valence < -0.3:
            valence_summary = "generally subdued"
        elif abs(state.valence) < 0.1:
            valence_summary = "neutral"
        else:
            valence_summary = "mixed"

        # Generate arousal summary
        if state.arousal > 0.6:
            arousal_summary = "energetic and engaged"
        elif state.arousal > 0.4:
            arousal_summary = "moderately engaged"
        elif state.arousal < 0.2:
            arousal_summary = "calm throughout"
        else:
            arousal_summary = "variable energy levels"

        # Combine for content description
        if state.confidence >= self.confidence_threshold:
            if valence_summary == "generally positive" and "energetic" in arousal_summary:
                content = f"User appeared {valence_summary} and {arousal_summary} during this session"
            elif valence_summary == "generally positive" and "calm" in arousal_summary:
                content = f"User appeared {valence_summary} and {arousal_summary} during this session"
            elif valence_summary == "generally subdued" and "calm" in arousal_summary:
                content = f"User appeared {valence_summary} and {arousal_summary} during this session"
            elif valence_summary == "neutral":
                content = f"User maintained a {valence_summary} demeanor with {arousal_summary}"
            else:
                content = f"User displayed {valence_summary} affect with {arousal_summary}"
        else:
            content = "Affect patterns were unclear or inconsistent during this session"

        return {
            "content": content,
            "valence_summary": valence_summary,
            "arousal_summary": arousal_summary,
            "confidence": state.confidence,
            "duration_minutes": duration_minutes,
            "stability_evidence": [
                f"Stable for {state.stable_duration_sec:.1f} seconds",
                f"Based on {state.sample_count} observations",
                f"Average confidence: {state.confidence:.2f}"
            ]
        }
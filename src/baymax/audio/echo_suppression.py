"""Echo suppression: prevents Bay-Max from responding to its own TTS output."""

from __future__ import annotations

import logging
from difflib import SequenceMatcher

from baymax.audio.schemas import EchoDecision

logger = logging.getLogger(__name__)


class EchoSuppressor:
    """Compares transcribed text against recently spoken Bay-Max text.

    Uses SequenceMatcher for text similarity. If the transcribed text is
    too similar to what Bay-Max recently said, it is classified as echo.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        max_recent: int = 5,
    ) -> None:
        self._threshold = similarity_threshold
        self._max_recent = max_recent
        self._recent_spoken: list[str] = []

    @property
    def recent_spoken(self) -> list[str]:
        return list(self._recent_spoken)

    def record_spoken(self, text: str) -> None:
        """Record text that Bay-Max has spoken for future echo comparison."""
        if text.strip():
            self._recent_spoken.append(text.strip().lower())
            if len(self._recent_spoken) > self._max_recent:
                self._recent_spoken.pop(0)

    def check(self, transcribed_text: str) -> EchoDecision:
        """Check if transcribed text is likely an echo of Bay-Max speech.

        Returns an EchoDecision indicating whether the text should be rejected.
        """
        if not transcribed_text.strip():
            return EchoDecision(
                is_echo=False,
                similarity_score=0.0,
                reason="Empty transcription",
                threshold=self._threshold,
            )

        normalized = transcribed_text.strip().lower()
        best_score = 0.0
        best_match = ""

        for spoken in self._recent_spoken:
            score = SequenceMatcher(None, normalized, spoken).ratio()
            if score > best_score:
                best_score = score
                best_match = spoken

        is_echo = best_score >= self._threshold

        decision = EchoDecision(
            is_echo=is_echo,
            similarity_score=round(best_score, 4),
            compared_to=best_match,
            threshold=self._threshold,
            reason=f"Similarity {best_score:.4f} {'>='}  threshold {self._threshold}"
            if is_echo
            else f"Similarity {best_score:.4f} < threshold {self._threshold}",
        )

        if is_echo:
            logger.info(
                "Echo suppressed: '%.40s...' matches '%.40s...' (%.4f)",
                transcribed_text,
                best_match,
                best_score,
            )

        return decision

    def clear(self) -> None:
        """Clear recorded spoken text history."""
        self._recent_spoken.clear()

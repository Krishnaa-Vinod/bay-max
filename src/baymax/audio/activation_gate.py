"""Activation gate abstraction for voice assistant (Iteration 013).

Provides interfaces for different activation mechanisms:
- Push-to-talk (explicit button)
- Continuous VAD (always listening)
- Wake phrase gate (ASR-gated phrase matching)

The wake phrase gate is a provisional implementation using ASR transcript matching.
It is NOT equivalent to a production on-device wake-word model. Future iterations
may integrate a dedicated wake-word engine (e.g., Porcupine, OpenWakeWord) that
can be swapped in via this interface.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


@dataclass
class ActivationDecision:
    """Result of checking activation status."""

    is_activated: bool
    """Whether activation condition is met."""

    trigger_phrase: str = ""
    """The phrase that triggered activation (for wake phrase mode)."""

    confidence: float = 1.0
    """Confidence in the activation decision (0.0-1.0)."""

    remaining_text: str = ""
    """For wake phrase mode: the text after the wake phrase that should be processed."""

    reason: str = ""
    """Human-readable reason for the decision."""


class ActivationGate(ABC):
    """Abstract interface for activation mechanisms.

    Subclasses implement different activation strategies. The gate is checked
    before processing speech input to determine if the user intended to address
    the assistant.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this activation gate for logging/diagnostics."""
        pass

    @abstractmethod
    def check(self, text: str | None = None, is_button_pressed: bool = False) -> ActivationDecision:
        """Check if activation condition is met.

        Args:
            text: Transcribed text (for wake phrase matching).
            is_button_pressed: Whether push-to-talk button is pressed.

        Returns:
            ActivationDecision indicating whether to proceed.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset gate state (e.g., clear any held state)."""
        pass


class PushToTalkGate(ActivationGate):
    """Activation via explicit button press.

    The simplest and most reliable activation method. Good for debugging
    and environments where always-listening is undesirable.
    """

    @property
    def name(self) -> str:
        return "push_to_talk"

    def check(self, text: str | None = None, is_button_pressed: bool = False) -> ActivationDecision:
        if is_button_pressed:
            return ActivationDecision(
                is_activated=True,
                confidence=1.0,
                remaining_text=text or "",
                reason="button_pressed",
            )
        return ActivationDecision(
            is_activated=False,
            reason="button_not_pressed",
        )

    def reset(self) -> None:
        pass


class ContinuousVADGate(ActivationGate):
    """Always-activated gate for continuous VAD mode.

    Any speech detected by VAD is considered activation. Relies on good
    speaking lock and echo suppression to avoid self-triggering.
    """

    @property
    def name(self) -> str:
        return "continuous_vad"

    def check(self, text: str | None = None, is_button_pressed: bool = False) -> ActivationDecision:
        # In continuous mode, we're always activated when there's speech
        if text and text.strip():
            return ActivationDecision(
                is_activated=True,
                confidence=1.0,
                remaining_text=text,
                reason="continuous_listening",
            )
        return ActivationDecision(
            is_activated=False,
            reason="no_speech",
        )

    def reset(self) -> None:
        pass


class WakePhraseGate(ActivationGate):
    """ASR-gated wake phrase activation (provisional).

    This is an intermediate Siri-like mode using phrase-based activation.
    The wake phrase is detected by matching against the beginning of
    transcribed speech.

    IMPORTANT: This is a provisional implementation that depends on ASR quality.
    It is NOT equivalent to a production on-device wake-word model like Porcupine.
    The interface is designed so a future wake-word backend can replace this
    without touching the rest of the runtime.

    Matching behavior:
    - Case-insensitive
    - Fuzzy matching with configurable threshold
    - Strips common filler words before wake phrase
    - Supports multiple wake phrases
    """

    # Common filler words to strip from the beginning
    FILLER_WORDS = {"um", "uh", "like", "so", "well", "okay", "ok"}

    def __init__(
        self,
        phrases: list[str] | None = None,
        fuzzy_threshold: float = 0.75,
    ) -> None:
        """Initialize wake phrase gate.

        Args:
            phrases: List of wake phrases to detect (e.g., ["hey baymax", "baymax"]).
            fuzzy_threshold: Minimum similarity ratio for fuzzy matching (0.0-1.0).
        """
        default_phrases = ["hey baymax", "hi baymax", "baymax"]
        self._phrases = [p.lower().strip() for p in (phrases or default_phrases)]
        self._fuzzy_threshold = fuzzy_threshold
        logger.info(
            "WakePhraseGate initialized with phrases: %s (threshold=%.2f)",
            self._phrases,
            fuzzy_threshold,
        )

    @property
    def name(self) -> str:
        return "wake_phrase_gate"

    @property
    def phrases(self) -> list[str]:
        """Configured wake phrases."""
        return list(self._phrases)

    def check(self, text: str | None = None, is_button_pressed: bool = False) -> ActivationDecision:
        # Also accept button press as override
        if is_button_pressed:
            return ActivationDecision(
                is_activated=True,
                confidence=1.0,
                remaining_text=text or "",
                reason="button_override",
            )

        if not text or not text.strip():
            return ActivationDecision(
                is_activated=False,
                reason="no_speech",
            )

        # Normalize input
        normalized = text.lower().strip()

        # Strip leading filler words
        normalized = self._strip_fillers(normalized)

        # Try exact prefix match first
        connector_pattern = r"^[,\s]*(can you|could you|please|)\s*"
        for phrase in self._phrases:
            if normalized.startswith(phrase):
                remaining = normalized[len(phrase):].strip()
                # Remove common connectors after wake phrase
                remaining = re.sub(connector_pattern, "", remaining, flags=re.IGNORECASE)
                return ActivationDecision(
                    is_activated=True,
                    trigger_phrase=phrase,
                    confidence=1.0,
                    remaining_text=remaining.strip(),
                    reason="exact_prefix_match",
                )

        # Try fuzzy match on the beginning of the text
        for phrase in self._phrases:
            # Extract the same length from the beginning
            words_to_check = normalized.split()[:len(phrase.split()) + 1]
            prefix = " ".join(words_to_check[:len(phrase.split())])

            similarity = SequenceMatcher(None, prefix, phrase).ratio()
            if similarity >= self._fuzzy_threshold:
                remaining = " ".join(words_to_check[len(phrase.split()):]) + " " + " ".join(
                    normalized.split()[len(phrase.split()) + 1:]
                )
                remaining = re.sub(
                    connector_pattern, "", remaining.strip(), flags=re.IGNORECASE
                )
                return ActivationDecision(
                    is_activated=True,
                    trigger_phrase=phrase,
                    confidence=similarity,
                    remaining_text=remaining.strip(),
                    reason=f"fuzzy_match_{similarity:.2f}",
                )

        return ActivationDecision(
            is_activated=False,
            reason="no_wake_phrase_detected",
        )

    def _strip_fillers(self, text: str) -> str:
        """Remove leading filler words."""
        words = text.split()
        while words and words[0] in self.FILLER_WORDS:
            words = words[1:]
        return " ".join(words)

    def reset(self) -> None:
        pass


def create_activation_gate(
    mode: str,
    wake_phrases: list[str] | None = None,
    fuzzy_threshold: float = 0.75,
) -> ActivationGate:
    """Factory function to create the appropriate activation gate.

    Args:
        mode: Activation mode ("push_to_talk", "continuous_vad", "wake_phrase_gate").
        wake_phrases: Wake phrases for wake_phrase_gate mode.
        fuzzy_threshold: Fuzzy matching threshold for wake phrases.

    Returns:
        Configured ActivationGate instance.

    Raises:
        ValueError: If mode is not recognized.
    """
    mode = mode.lower().strip()

    if mode == "push_to_talk":
        return PushToTalkGate()
    elif mode == "continuous_vad":
        return ContinuousVADGate()
    elif mode == "wake_phrase_gate":
        return WakePhraseGate(phrases=wake_phrases, fuzzy_threshold=fuzzy_threshold)
    else:
        raise ValueError(f"Unknown activation mode: {mode}")

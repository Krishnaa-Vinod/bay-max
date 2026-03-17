"""Voice Activity Detection using Silero VAD."""

from __future__ import annotations

import logging
import time
from enum import StrEnum

import numpy as np

from baymax.audio.schemas import VADDecision

logger = logging.getLogger(__name__)


class VADState(StrEnum):
    """State machine for VAD segmentation."""

    IDLE = "idle"
    SPEECH_STARTED = "speech_started"
    SPEECH_ONGOING = "speech_ongoing"
    SILENCE_AFTER_SPEECH = "silence_after_speech"


class SileroVAD:
    """Silero VAD wrapper with state machine for speech segmentation.

    Transitions:
        IDLE -> SPEECH_STARTED: speech detected for >= min_speech_ms
        SPEECH_STARTED -> SPEECH_ONGOING: speech continues
        SPEECH_ONGOING -> SILENCE_AFTER_SPEECH: silence detected
        SILENCE_AFTER_SPEECH -> IDLE: silence lasts >= silence_ms (segment complete)
        SILENCE_AFTER_SPEECH -> SPEECH_ONGOING: speech resumes before silence_ms
    """

    def __init__(
        self,
        threshold: float = 0.65,
        min_speech_ms: float = 300.0,
        silence_ms: float = 500.0,
        sample_rate: int = 16000,
    ) -> None:
        self._threshold = threshold
        self._min_speech_ms = min_speech_ms
        self._silence_ms = silence_ms
        self._sample_rate = sample_rate
        self._state = VADState.IDLE
        self._model = None
        self._speech_start_time: float = 0.0
        self._silence_start_time: float = 0.0
        self._accumulated_audio: list[np.ndarray] = []

    @property
    def state(self) -> VADState:
        return self._state

    @property
    def is_available(self) -> bool:
        """Check if Silero VAD can be loaded."""
        try:
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    def load_model(self) -> bool:
        """Load the Silero VAD model. Returns True on success."""
        try:
            import torch

            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self._model = model
            logger.info("Silero VAD model loaded successfully")
            return True
        except Exception as exc:
            logger.warning("Failed to load Silero VAD: %s", exc)
            return False

    def reset(self) -> None:
        """Reset state machine to IDLE."""
        self._state = VADState.IDLE
        self._speech_start_time = 0.0
        self._silence_start_time = 0.0
        self._accumulated_audio.clear()
        if self._model is not None:
            try:
                self._model.reset_states()
            except Exception:
                pass

    def process_chunk(self, audio: np.ndarray) -> tuple[VADDecision, np.ndarray | None]:
        """Process an audio chunk and return VAD decision + completed segment if any.

        Args:
            audio: Float32 audio samples at the configured sample rate.

        Returns:
            Tuple of (decision, completed_segment_or_None).
            completed_segment is non-None only when a speech segment ends.
        """
        if self._model is None:
            return VADDecision(is_speech=False, confidence=0.0, backend="silero"), None

        now = time.time()
        confidence = self._get_speech_probability(audio)
        is_speech = confidence >= self._threshold

        decision = VADDecision(
            is_speech=is_speech,
            confidence=round(confidence, 4),
            backend="silero",
        )

        completed_segment: np.ndarray | None = None

        if self._state == VADState.IDLE:
            if is_speech:
                self._state = VADState.SPEECH_STARTED
                self._speech_start_time = now
                self._accumulated_audio.clear()
                self._accumulated_audio.append(audio)

        elif self._state == VADState.SPEECH_STARTED:
            self._accumulated_audio.append(audio)
            elapsed_ms = (now - self._speech_start_time) * 1000
            if is_speech and elapsed_ms >= self._min_speech_ms:
                self._state = VADState.SPEECH_ONGOING
            elif not is_speech:
                # Too short, reset
                self._state = VADState.IDLE
                self._accumulated_audio.clear()

        elif self._state == VADState.SPEECH_ONGOING:
            self._accumulated_audio.append(audio)
            if not is_speech:
                self._state = VADState.SILENCE_AFTER_SPEECH
                self._silence_start_time = now

        elif self._state == VADState.SILENCE_AFTER_SPEECH:
            self._accumulated_audio.append(audio)
            if is_speech:
                # Speech resumed
                self._state = VADState.SPEECH_ONGOING
            else:
                silence_elapsed_ms = (now - self._silence_start_time) * 1000
                if silence_elapsed_ms >= self._silence_ms:
                    # Segment complete
                    completed_segment = np.concatenate(self._accumulated_audio)
                    self._state = VADState.IDLE
                    self._accumulated_audio.clear()

        return decision, completed_segment

    def _get_speech_probability(self, audio: np.ndarray) -> float:
        """Run Silero VAD on an audio chunk. Returns probability of speech."""
        try:
            import torch

            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            tensor = torch.from_numpy(audio)
            prob = self._model(tensor, self._sample_rate)
            return float(prob.item()) if hasattr(prob, "item") else float(prob)
        except Exception as exc:
            logger.debug("VAD inference error: %s", exc)
            return 0.0


class NullVAD:
    """No-op VAD for testing. Every chunk is treated as speech."""

    def __init__(self) -> None:
        self._state = VADState.IDLE

    @property
    def state(self) -> VADState:
        return self._state

    @property
    def is_available(self) -> bool:
        return True

    def load_model(self) -> bool:
        return True

    def reset(self) -> None:
        self._state = VADState.IDLE

    def process_chunk(self, audio: np.ndarray) -> tuple[VADDecision, np.ndarray | None]:
        return VADDecision(is_speech=True, confidence=1.0, backend="null"), audio

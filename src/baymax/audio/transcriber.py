"""ASR (speech-to-text) provider abstraction and implementations."""

from __future__ import annotations

import abc
import logging
import time

import numpy as np

from baymax.audio.schemas import TranscriptionResult

logger = logging.getLogger(__name__)


class ASRProvider(abc.ABC):
    """Abstract speech-to-text provider."""

    @abc.abstractmethod
    def name(self) -> str:
        """Return the backend name."""

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if this backend can be used."""

    @abc.abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str = "en",
    ) -> TranscriptionResult:
        """Transcribe audio to text."""

    @abc.abstractmethod
    def model_info(self) -> dict[str, str]:
        """Return model metadata."""


class FasterWhisperProvider(ASRProvider):
    """ASR backend using faster-whisper (CTranslate2)."""

    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "auto",
        compute_type: str = "default",
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model = None

    def name(self) -> str:
        return "faster_whisper"

    def is_available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel

            device = self._device
            if device == "auto":
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"

            compute = self._compute_type
            if compute == "default":
                compute = "float16" if device == "cuda" else "int8"

            self._model = WhisperModel(
                self._model_size,
                device=device,
                compute_type=compute,
            )
            logger.info(
                "faster-whisper model loaded: %s on %s (%s)",
                self._model_size, device, compute,
            )
        except Exception as exc:
            logger.error("Failed to load faster-whisper model: %s", exc)
            raise

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str = "en",
    ) -> TranscriptionResult:
        t0 = time.time()
        try:
            self._ensure_model()

            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Normalize to [-1, 1] if needed
            if np.abs(audio).max() > 1.0:
                audio = audio / 32768.0

            segments, info = self._model.transcribe(
                audio,
                language=language if not self._model_size.endswith(".en") else None,
                beam_size=5,
                vad_filter=False,  # We already have our own VAD
            )

            text_parts = []
            for seg in segments:
                text_parts.append(seg.text.strip())

            text = " ".join(text_parts).strip()
            latency_ms = (time.time() - t0) * 1000

            return TranscriptionResult(
                text=text,
                language=info.language if hasattr(info, "language") else language,
                confidence=round(info.language_probability, 4)
                if hasattr(info, "language_probability")
                else 0.0,
                backend="faster_whisper",
                model=self._model_size,
                latency_ms=round(latency_ms, 1),
                success=bool(text),
                error=None if text else "Empty transcription",
                avg_logprob=getattr(info, "avg_logprob", None),
                no_speech_prob=getattr(info, "no_speech_prob", None),
            )
        except Exception as exc:
            latency_ms = (time.time() - t0) * 1000
            logger.error("Transcription failed: %s", exc)
            return TranscriptionResult(
                text="",
                backend="faster_whisper",
                model=self._model_size,
                latency_ms=round(latency_ms, 1),
                success=False,
                error=str(exc),
            )

    def model_info(self) -> dict[str, str]:
        return {
            "backend": "faster_whisper",
            "model": self._model_size,
            "device": self._device,
            "compute_type": self._compute_type,
        }


class NullASRProvider(ASRProvider):
    """No-op ASR provider for testing. Returns empty transcriptions."""

    def name(self) -> str:
        return "null"

    def is_available(self) -> bool:
        return True

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str = "en",
    ) -> TranscriptionResult:
        return TranscriptionResult(
            text="",
            backend="null",
            model="none",
            latency_ms=0.0,
            success=False,
            error="Null ASR provider (testing only)",
        )

    def model_info(self) -> dict[str, str]:
        return {"backend": "null", "model": "none"}


def get_asr_provider(
    backend: str = "faster_whisper",
    model: str = "base.en",
    device: str = "auto",
) -> ASRProvider:
    """Factory function to create an ASR provider."""
    if backend == "faster_whisper":
        provider = FasterWhisperProvider(model_size=model, device=device)
        if provider.is_available():
            return provider
        logger.warning("faster-whisper not available, falling back to null")
        return NullASRProvider()
    elif backend == "null":
        return NullASRProvider()
    else:
        logger.warning("Unknown ASR backend '%s', using null", backend)
        return NullASRProvider()

"""TTS provider interface and factory."""

from __future__ import annotations

import abc
import logging

from baymax.tts.schemas import SpeechSynthesisResult, TTSBackendInfo

logger = logging.getLogger(__name__)


class TTSProvider(abc.ABC):
    """Abstract base class for text-to-speech providers."""

    @abc.abstractmethod
    def name(self) -> str:
        """Return the backend name."""

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True if the backend is ready to synthesize."""

    @abc.abstractmethod
    def info(self) -> TTSBackendInfo:
        """Return backend metadata."""

    @abc.abstractmethod
    def synthesize(self, text: str, output_path: str) -> SpeechSynthesisResult:
        """Synthesize text to a WAV file at output_path."""


class NullTTSProvider(TTSProvider):
    """Silent TTS provider for testing and headless environments."""

    def name(self) -> str:
        return "null"

    def is_available(self) -> bool:
        return True

    def info(self) -> TTSBackendInfo:
        return TTSBackendInfo(
            backend="null",
            status="available",
            voice="silent",
            notes="Silent provider — no audio produced.",
        )

    def synthesize(self, text: str, output_path: str) -> SpeechSynthesisResult:
        return SpeechSynthesisResult(
            success=True,
            wav_path=None,
            backend="null",
            voice="silent",
            text=text,
        )


def get_tts_provider(
    backend: str,
    voice: str = "",
    device: str = "auto",
    model_dir: str = "",
    sample_rate: int = 24000,
) -> TTSProvider:
    """Factory: return a TTSProvider based on the backend name."""
    if backend == "null":
        return NullTTSProvider()

    if backend == "kokoro":
        from baymax.tts.kokoro_provider import KokoroTTSProvider

        return KokoroTTSProvider(
            voice=voice or "af_heart",
            device=device,
            model_dir=model_dir,
            sample_rate=sample_rate,
        )

    if backend == "piper":
        from baymax.tts.piper_provider import PiperTTSProvider

        return PiperTTSProvider(
            model_path=model_dir,
            voice=voice,
        )

    logger.warning("Unknown TTS backend '%s', falling back to null.", backend)
    return NullTTSProvider()

"""Piper TTS provider — fast local neural TTS (MIT-licensed)."""

from __future__ import annotations

import logging

from baymax.tts.provider import TTSProvider
from baymax.tts.schemas import SpeechSynthesisResult, TTSBackendInfo

logger = logging.getLogger(__name__)


class PiperTTSProvider(TTSProvider):
    """TTS provider backed by Piper (local, MIT-licensed).

    Piper requires the ``piper-tts`` package and a downloaded ONNX voice
    model.  When the package or model is absent, the provider reports
    itself as unavailable and synthesis returns a clear error.
    """

    def __init__(
        self,
        model_path: str = "",
        voice: str = "en_US-lessac-medium",
        config_path: str = "",
    ) -> None:
        self._model_path = model_path
        self._voice = voice
        self._config_path = config_path
        self._available: bool | None = None

    def name(self) -> str:
        return "piper"

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import piper  # noqa: F401

            if self._model_path:
                import os

                self._available = os.path.isfile(self._model_path)
            else:
                self._available = True
        except ImportError:
            self._available = False
        return self._available

    def info(self) -> TTSBackendInfo:
        return TTSBackendInfo(
            backend="piper",
            status="available" if self.is_available() else "unavailable",
            voice=self._voice,
            sample_rate=22050,
            notes=(
                "Piper local neural TTS. Requires piper-tts package "
                "and an ONNX voice model."
            ),
        )

    def synthesize(
        self, text: str, output_path: str
    ) -> SpeechSynthesisResult:
        if not self.is_available():
            return SpeechSynthesisResult(
                success=False,
                backend="piper",
                voice=self._voice,
                text=text,
                error=(
                    "Piper TTS not available. Install piper-tts and "
                    "download a voice model."
                ),
            )
        return SpeechSynthesisResult(
            success=False,
            backend="piper",
            voice=self._voice,
            text=text,
            error=(
                "Piper backend is a placeholder — install piper-tts and "
                "configure BAYMAX_PIPER_MODEL_PATH to enable."
            ),
        )

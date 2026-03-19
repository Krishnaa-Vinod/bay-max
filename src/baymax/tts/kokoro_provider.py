"""Kokoro TTS provider — lightweight local neural TTS."""

from __future__ import annotations

import logging
import os

from baymax.tts.provider import TTSProvider
from baymax.tts.schemas import SpeechSynthesisResult, TTSBackendInfo

logger = logging.getLogger(__name__)


class KokoroTTSProvider(TTSProvider):
    """TTS provider backed by Kokoro (kokoro-82M, Apache-2.0)."""

    def __init__(
        self,
        voice: str = "am_michael",
        device: str = "auto",
        model_dir: str = "",
        sample_rate: int = 24000,
    ) -> None:
        self._voice = voice
        self._device = device
        self._model_dir = model_dir
        self._sample_rate = sample_rate
        self._pipeline: object | None = None
        self._available: bool | None = None

    def name(self) -> str:
        return "kokoro"

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            self._init_pipeline()
            self._available = self._pipeline is not None
        except Exception as exc:
            logger.warning("Kokoro TTS not available: %s", exc)
            self._available = False
        return self._available

    def info(self) -> TTSBackendInfo:
        return TTSBackendInfo(
            backend="kokoro",
            status="available" if self.is_available() else "unavailable",
            voice=self._voice,
            sample_rate=self._sample_rate,
            notes="Kokoro-82M open-weight local TTS.",
        )

    def _init_pipeline(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from kokoro import KPipeline

            lang_code = self._voice[:1] if self._voice else "a"
            self._pipeline = KPipeline(lang_code=lang_code)
            logger.info(
                "Kokoro TTS pipeline initialized (voice=%s).",
                self._voice,
            )
        except Exception as exc:
            logger.error("Failed to initialize Kokoro pipeline: %s", exc)
            self._pipeline = None
            raise

    def synthesize(
        self, text: str, output_path: str
    ) -> SpeechSynthesisResult:
        if not text or not text.strip():
            return SpeechSynthesisResult(
                success=False,
                backend="kokoro",
                voice=self._voice,
                text=text,
                error="Empty text provided.",
            )
        try:
            self._init_pipeline()
            if self._pipeline is None:
                return SpeechSynthesisResult(
                    success=False,
                    backend="kokoro",
                    voice=self._voice,
                    text=text,
                    error="Pipeline not initialized.",
                )
            import soundfile as sf

            generator = self._pipeline(
                text, voice=self._voice, speed=1.0
            )
            all_samples = []
            for _gs, _ps, audio in generator:
                if audio is not None:
                    all_samples.append(audio)

            if not all_samples:
                return SpeechSynthesisResult(
                    success=False,
                    backend="kokoro",
                    voice=self._voice,
                    text=text,
                    error="No audio generated.",
                )

            import numpy as np

            combined = np.concatenate(all_samples)
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            sf.write(output_path, combined, self._sample_rate)

            duration = len(combined) / self._sample_rate
            return SpeechSynthesisResult(
                success=True,
                wav_path=output_path,
                backend="kokoro",
                voice=self._voice,
                sample_rate=self._sample_rate,
                duration_sec=round(duration, 2),
                text=text,
            )
        except Exception as exc:
            logger.error("Kokoro synthesis failed: %s", exc)
            return SpeechSynthesisResult(
                success=False,
                backend="kokoro",
                voice=self._voice,
                text=text,
                error=str(exc),
            )

"""Speech synthesis service with queued playback."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from collections import deque
from datetime import UTC, datetime

from baymax.tts.provider import TTSProvider, get_tts_provider
from baymax.tts.schemas import (
    SpeechEvent,
    SpeechQueueStatus,
    SpeechSynthesisResult,
)

logger = logging.getLogger(__name__)


class SpeechService:
    """Synthesizes text and manages a bounded playback queue.

    The service:
    1. Accepts text strings and synthesizes them to WAV via a TTSProvider.
    2. Optionally plays audio through speakers using sounddevice.
    3. Maintains a bounded queue so responses never pile up unboundedly.
    4. Tracks all speech events for artifact logging.
    """

    def __init__(
        self,
        provider: TTSProvider | None = None,
        backend: str = "null",
        voice: str = "",
        device: str = "auto",
        model_dir: str = "",
        sample_rate: int = 24000,
        output_dir: str = "./artifacts/tts",
        enable_playback: bool = True,
        audio_backend: str = "sounddevice",
        output_device: str | int | None = None,
        max_queue_size: int = 5,
    ) -> None:
        self._provider = provider or get_tts_provider(
            backend=backend,
            voice=voice,
            device=device,
            model_dir=model_dir,
            sample_rate=sample_rate,
        )
        self._output_dir = output_dir
        self._enable_playback = enable_playback
        self._audio_backend = audio_backend
        self._output_device = self._normalize_output_device(output_device)
        self._sample_rate = sample_rate
        self._max_queue_size = max_queue_size

        logger.info(
            "SpeechService playback configured: backend=%s output_device=%s",
            self._audio_backend,
            self._output_device if self._output_device is not None else "auto",
        )

        self._queue: deque[SpeechSynthesisResult] = deque(
            maxlen=max_queue_size
        )
        self._is_playing = False
        self._last_spoken_text = ""
        self._last_spoken_at: datetime | None = None
        self._total_synthesized = 0
        self._total_played = 0
        self._total_errors = 0
        self._events: list[SpeechEvent] = []
        self._playback_lock = asyncio.Lock()

    @staticmethod
    def _normalize_output_device(output_device: str | int | None) -> int | None:
        if output_device is None:
            return None
        if isinstance(output_device, int):
            return output_device
        text = str(output_device).strip().lower()
        if text in {"", "default", "auto", "none"}:
            return None
        try:
            return int(text)
        except ValueError:
            return None

    @property
    def provider(self) -> TTSProvider:
        return self._provider

    @property
    def events(self) -> list[SpeechEvent]:
        return list(self._events)

    def queue_status(self) -> SpeechQueueStatus:
        return SpeechQueueStatus(
            queue_depth=len(self._queue),
            is_playing=self._is_playing,
            last_spoken_text=self._last_spoken_text,
            last_spoken_at=self._last_spoken_at,
            total_synthesized=self._total_synthesized,
            total_played=self._total_played,
            total_errors=self._total_errors,
        )

    def synthesize(
        self,
        text: str,
        session_id: str | None = None,
        trigger_event: str | None = None,
    ) -> SpeechSynthesisResult:
        """Synthesize text to WAV. Returns the result immediately."""
        os.makedirs(self._output_dir, exist_ok=True)
        filename = f"speech_{uuid.uuid4().hex[:12]}.wav"
        output_path = os.path.join(self._output_dir, filename)

        result = self._provider.synthesize(text, output_path)
        result.synthesis_ok = result.success
        self._total_synthesized += 1

        event = SpeechEvent(
            text=text,
            backend=self._provider.name(),
            voice=result.voice,
            wav_path=result.wav_path,
            success=result.success,
            error=result.error,
            session_id=session_id,
            trigger_event=trigger_event,
        )

        if not result.success:
            self._total_errors += 1
            event.event_type = "speech_failed"

        self._events.append(event)

        if result.success and result.wav_path:
            self._queue.append(result)

        return result

    async def synthesize_and_play(
        self,
        text: str,
        session_id: str | None = None,
        trigger_event: str | None = None,
    ) -> SpeechSynthesisResult:
        """Synthesize and optionally play. Non-blocking via background."""
        result = self.synthesize(
            text, session_id=session_id, trigger_event=trigger_event
        )
        if result.success and result.wav_path and self._enable_playback:
            await self._play_audio(result)
        elif result.success and result.wav_path and not self._enable_playback:
            result.playback_ok = False
            result.playback_error = "Playback disabled by configuration"
        return result

    async def _play_audio(self, result: SpeechSynthesisResult) -> None:
        """Play a WAV file through speakers if possible."""
        async with self._playback_lock:
            self._is_playing = True
            play_event = self._events[-1] if self._events else None
            if play_event:
                play_event.playback_started_at = datetime.now(UTC)
            try:
                played = await self._do_playback(result.wav_path or "")
                if played:
                    result.playback_ok = True
                    result.playback_error = None
                    self._total_played += 1
                    self._last_spoken_text = result.text
                    self._last_spoken_at = datetime.now(UTC)
                    if play_event:
                        play_event.playback_finished_at = datetime.now(
                            UTC
                        )
                        play_event.event_type = "speech_played"
                else:
                    result.playback_ok = False
                    if not result.playback_error:
                        result.playback_error = "Playback unavailable"
            except Exception as exc:
                logger.warning("Audio playback failed: %s", exc)
                self._total_errors += 1
                result.playback_ok = False
                result.playback_error = str(exc)
            finally:
                self._is_playing = False

    async def _do_playback(self, wav_path: str) -> bool:
        """Attempt playback via sounddevice. Returns True if played."""
        if not wav_path or not os.path.isfile(wav_path):
            return False
        try:
            import numpy as np
            import sounddevice as sd
            import soundfile as sf

            data, samplerate = sf.read(wav_path, dtype="float32")
            loop = asyncio.get_event_loop()

            def _resolve_output_device() -> int | None:
                if self._output_device is not None:
                    try:
                        sd.query_devices(self._output_device, "output")
                        return self._output_device
                    except Exception:
                        logger.warning(
                            "Configured output device %s is not output-capable; falling back to auto selection",
                            self._output_device,
                        )

                candidates: list[tuple[int, int]] = []
                try:
                    for idx, dev in enumerate(sd.query_devices()):
                        if int(dev.get("max_output_channels", 0)) <= 0:
                            continue
                        name = str(dev.get("name", "")).lower()
                        score = 0
                        if "analog" in name or "speaker" in name or "headphone" in name:
                            score += 3
                        if "hdmi" in name or "displayport" in name or "monitor" in name:
                            score -= 2
                        candidates.append((score, idx))
                except Exception:
                    candidates = []

                if candidates:
                    candidates.sort(reverse=True)
                    chosen = candidates[0][1]
                    logger.info("Using auto-selected output device index %d", chosen)
                    return chosen

                default_devices = sd.default.device
                if isinstance(default_devices, (list, tuple)) and len(default_devices) >= 2:
                    try:
                        out_idx = int(default_devices[1])
                        if out_idx >= 0:
                            return out_idx
                    except Exception:
                        pass

                return None

            selected_output_device = _resolve_output_device()
            logger.info(
                "SpeechService playback using output device: %s",
                selected_output_device if selected_output_device is not None else "default",
            )

            try:
                await loop.run_in_executor(
                    None,
                    lambda: sd.play(data, samplerate, device=selected_output_device),
                )
                await loop.run_in_executor(None, sd.wait)
            except Exception as exc:
                err_text = str(exc)
                if (
                    "Invalid sample rate" not in err_text
                    and "Invalid number of channels" not in err_text
                ):
                    raise

                out_device = selected_output_device

                output_info = sd.query_devices(out_device, "output")
                target_rate = int(round(float(output_info.get("default_samplerate", samplerate))))
                if target_rate <= 0:
                    target_rate = int(samplerate)

                max_channels = int(output_info.get("max_output_channels", 0))
                if max_channels <= 0:
                    raise

                compatible_data = data
                if compatible_data.ndim == 1 and max_channels >= 2:
                    compatible_data = np.stack([compatible_data, compatible_data], axis=1)
                elif compatible_data.ndim > 1 and compatible_data.shape[1] > max_channels:
                    if max_channels == 1:
                        compatible_data = np.mean(compatible_data, axis=1).astype(np.float32)
                    else:
                        compatible_data = compatible_data[:, :max_channels]

                src_len = int(compatible_data.shape[0]) if getattr(compatible_data, "ndim", 1) > 0 else 0
                if src_len <= 1:
                    raise

                resampled = compatible_data
                if target_rate != samplerate:
                    logger.info(
                        "Resampling playback from %dHz to %dHz for output device compatibility",
                        samplerate,
                        target_rate,
                    )
                    dst_len = max(1, int(round(src_len * target_rate / samplerate)))
                    src_x = np.linspace(0.0, 1.0, src_len, endpoint=False)
                    dst_x = np.linspace(0.0, 1.0, dst_len, endpoint=False)

                    if compatible_data.ndim == 1:
                        resampled = np.interp(dst_x, src_x, compatible_data).astype(np.float32)
                    else:
                        channels = [
                            np.interp(dst_x, src_x, compatible_data[:, ch])
                            for ch in range(compatible_data.shape[1])
                        ]
                        resampled = np.stack(channels, axis=1).astype(np.float32)

                await loop.run_in_executor(
                    None,
                    lambda: sd.play(resampled, target_rate, device=out_device),
                )
                await loop.run_in_executor(None, sd.wait)

            return True
        except ImportError:
            logger.info(
                "sounddevice not available — WAV saved but not played: %s",
                wav_path,
            )
            self._last_spoken_text = self._events[-1].text if self._events else ""
            self._last_spoken_at = datetime.now(UTC)
            if self._events:
                self._events[-1].error = "sounddevice not available"
            return False
        except OSError as exc:
            logger.warning("Audio device error: %s — WAV saved: %s", exc, wav_path)
            self._last_spoken_text = self._events[-1].text if self._events else ""
            self._last_spoken_at = datetime.now(UTC)
            if self._events:
                self._events[-1].error = str(exc)
            return False

    async def drain_queue(self) -> int:
        """Play all queued items. Returns count played."""
        played = 0
        while self._queue:
            result = self._queue.popleft()
            await self._play_audio(result)
            played += 1
        return played

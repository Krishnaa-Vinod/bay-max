"""Microphone capture module using sounddevice."""

from __future__ import annotations

import asyncio
import logging
import queue

import numpy as np

logger = logging.getLogger(__name__)


class MicrophoneCapture:
    """Captures audio from the system microphone via sounddevice.

    Runs a background thread that feeds audio chunks into a queue
    for asynchronous consumption.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        device: str | int | None = None,
        chunk_duration_ms: int = 30,
    ) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._device = device if device != "default" else None
        self._chunk_duration_ms = chunk_duration_ms
        self._chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)
        self._stream = None
        self._running = False
        self._available = self._check_available()

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def _check_available(self) -> bool:
        try:
            import sounddevice  # noqa: F401
            return True
        except (ImportError, OSError):
            return False

    def _candidate_sample_rates(self, sd) -> list[int]:
        rates: list[int] = [int(self._sample_rate)]

        default_devices = sd.default.device
        input_device = None
        if isinstance(default_devices, (list, tuple)) and len(default_devices) >= 1:
            input_device = default_devices[0]

        try:
            if input_device is not None and int(input_device) >= 0:
                device_info = sd.query_devices(int(input_device))
                default_sr = int(round(float(device_info.get("default_samplerate", 0))))
                if default_sr > 0:
                    rates.append(default_sr)
        except Exception:
            pass

        rates.extend([48000, 44100, 32000, 24000, 16000])

        unique: list[int] = []
        for rate in rates:
            if rate > 0 and rate not in unique:
                unique.append(rate)
        return unique

    def start(self) -> bool:
        """Start capturing audio. Returns True on success."""
        if self._running:
            return True
        if not self._available:
            logger.warning("sounddevice not available for microphone capture")
            return False
        try:
            import sounddevice as sd
            last_error: Exception | None = None
            for candidate_rate in self._candidate_sample_rates(sd):
                candidate_chunk = int(candidate_rate * self._chunk_duration_ms / 1000)
                try:
                    self._stream = sd.InputStream(
                        samplerate=candidate_rate,
                        channels=self._channels,
                        dtype="float32",
                        blocksize=candidate_chunk,
                        device=self._device,
                        callback=self._audio_callback,
                    )
                    self._stream.start()
                    self._sample_rate = candidate_rate
                    self._chunk_size = candidate_chunk
                    self._running = True
                    logger.info(
                        "Microphone capture started: rate=%d, channels=%d, chunk=%dms",
                        self._sample_rate,
                        self._channels,
                        self._chunk_duration_ms,
                    )
                    return True
                except Exception as exc:
                    last_error = exc
                    if self._stream is not None:
                        try:
                            self._stream.close()
                        except Exception:
                            pass
                        self._stream = None

            if last_error is not None:
                logger.error("Failed to start microphone: %s", last_error)
            return False
        except Exception as exc:
            logger.error("Failed to start microphone: %s", exc)
            return False

    def stop(self) -> None:
        """Stop capturing audio."""
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        # Drain queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        logger.info("Microphone capture stopped")

    def _audio_callback(self, indata, frames, time_info, status):
        """sounddevice callback — runs in audio thread."""
        if status:
            logger.debug("Mic status: %s", status)
        try:
            if indata.ndim > 1:
                self._queue.put_nowait(indata[:, 0].copy())
            else:
                self._queue.put_nowait(indata.copy().flatten())
        except queue.Full:
            pass  # Drop oldest audio if queue is full

    async def read_chunk(self, timeout: float = 0.1) -> np.ndarray | None:
        """Async read of next audio chunk. Returns None if no data."""
        loop = asyncio.get_event_loop()
        try:
            chunk = await loop.run_in_executor(
                None,
                lambda: self._queue.get(timeout=timeout),
            )
            return chunk
        except queue.Empty:
            return None


class NullMicrophone:
    """No-op microphone for testing on headless machines."""

    def __init__(self, sample_rate: int = 16000) -> None:
        self._sample_rate = sample_rate
        self._running = False

    @property
    def is_available(self) -> bool:
        return True

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def start(self) -> bool:
        self._running = True
        return True

    def stop(self) -> None:
        self._running = False

    async def read_chunk(self, timeout: float = 0.1) -> np.ndarray | None:
        await asyncio.sleep(timeout)
        return None

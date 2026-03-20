"""Frame source implementations for webcam and video replay."""

import logging
import os
import time

import numpy as np

from baymax.capture.interfaces import FrameSource

logger = logging.getLogger(__name__)


class OpenCVFrameSource(FrameSource):
    """Frame source using OpenCV VideoCapture for webcam or video file."""

    def __init__(
        self,
        source: int | str = 0,
        target_fps: int = 8,
    ) -> None:
        self._source = source
        self._resolved_source: int | str = source
        self._target_fps = target_fps
        self._cap = None
        self._is_open = False
        self._frame_interval = 1.0 / max(target_fps, 1)
        self._last_frame_time = 0.0
        self._frame_count = 0
        self._is_video_file = isinstance(source, str)

    @property
    def resolved_source(self) -> int | str:
        return self._resolved_source

    def _open_camera_with_fallback(self, cv2_module) -> None:
        """Open preferred camera index and prefer RGB-looking streams."""
        preferred = int(self._source)

        def _saturation_score(frame: np.ndarray) -> float:
            if frame.ndim != 3 or frame.shape[2] != 3:
                return 0.0
            hsv = cv2_module.cvtColor(frame, cv2_module.COLOR_BGR2HSV)
            return float(np.mean(hsv[:, :, 1]))

        def _open_candidate(idx: int):
            cap = cv2_module.VideoCapture(idx)
            if not cap.isOpened():
                cap.release()
                return None
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return None
            return (_saturation_score(frame), idx, cap)

        probe_count = max(1, int(os.getenv("BAYMAX_LIVE_CAMERA_PROBE_COUNT", "8")))
        prefer_rgb = os.getenv("BAYMAX_LIVE_CAMERA_PREFER_RGB", "true").strip().lower() in {
            "1", "true", "yes", "on"
        }
        min_sat = float(os.getenv("BAYMAX_LIVE_CAMERA_MIN_SATURATION", "12.0"))

        candidates = [preferred] + [idx for idx in range(probe_count) if idx != preferred]
        opened: list[tuple[float, int, object]] = []
        for idx in candidates:
            opened_candidate = _open_candidate(idx)
            if opened_candidate is not None:
                opened.append(opened_candidate)

        if not opened:
            raise RuntimeError(f"Failed to open webcam source: {preferred}")

        selected = None
        preferred_entry = next((entry for entry in opened if entry[1] == preferred), None)
        best_rgb_entry = max(opened, key=lambda entry: entry[0])

        if preferred_entry is not None and (
            not prefer_rgb or preferred_entry[0] >= min_sat
        ):
            selected = preferred_entry
        elif prefer_rgb:
            selected = best_rgb_entry
        else:
            selected = opened[0]

        for sat, idx, cap in opened:
            if idx == selected[1]:
                continue
            cap.release()

        self._cap = selected[2]
        self._resolved_source = selected[1]
        if self._resolved_source != preferred:
            logger.warning(
                "Preferred camera %d appears unavailable/IR-like (sat=%.1f); using RGB candidate %d (sat=%.1f)",
                preferred,
                preferred_entry[0] if preferred_entry is not None else -1.0,
                self._resolved_source,
                selected[0],
            )
        else:
            logger.info("Using preferred camera index %d", preferred)

    def open(self) -> None:
        try:
            import cv2
        except ImportError:
            raise RuntimeError(
                "OpenCV is required for live mode. "
                "Install with: pip install opencv-python-headless"
            )

        if isinstance(self._source, int):
            self._open_camera_with_fallback(cv2)
        else:
            self._cap = cv2.VideoCapture(self._source)
            if not self._cap.isOpened():
                raise RuntimeError(
                    f"Failed to open video source: {self._source}"
                )
            self._resolved_source = self._source

        self._is_open = True
        self._frame_count = 0
        logger.info(
            "Opened frame source: %s (target_fps=%d)",
            self._resolved_source,
            self._target_fps,
        )

    def read_frame(self) -> np.ndarray | None:
        if not self._is_open or self._cap is None:
            return None

        # Throttle to target FPS
        now = time.time()
        elapsed = now - self._last_frame_time
        if elapsed < self._frame_interval:
            return None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            if self._is_video_file:
                logger.info("Video replay ended after %d frames", self._frame_count)
                self._is_open = False
            return None

        self._last_frame_time = now
        self._frame_count += 1

        # OpenCV reads BGR, convert to RGB for the perception pipeline
        import cv2
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_open = False
        logger.info("Closed frame source after %d frames", self._frame_count)

    def is_open(self) -> bool:
        return self._is_open

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def total_video_frames(self) -> int | None:
        """Total frames in video file (None for webcam)."""
        if self._is_video_file and self._cap is not None:
            import cv2
            return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return None


class FolderFrameSource(FrameSource):
    """Frame source that reads frames from a folder of images."""

    def __init__(self, folder_path: str, target_fps: int = 8) -> None:
        self._folder_path = folder_path
        self._target_fps = target_fps
        self._frame_interval = 1.0 / max(target_fps, 1)
        self._last_frame_time = 0.0
        self._files: list[str] = []
        self._index = 0
        self._is_open = False

    def open(self) -> None:
        import os

        if not os.path.isdir(self._folder_path):
            raise RuntimeError(f"Frame folder not found: {self._folder_path}")

        extensions = (".jpg", ".jpeg", ".png", ".bmp")
        files = sorted(
            f for f in os.listdir(self._folder_path)
            if f.lower().endswith(extensions)
        )
        if not files:
            raise RuntimeError(f"No image files found in {self._folder_path}")

        self._files = [os.path.join(self._folder_path, f) for f in files]
        self._index = 0
        self._is_open = True
        logger.info(
            "Opened folder frame source: %s (%d files)",
            self._folder_path,
            len(self._files),
        )

    def read_frame(self) -> np.ndarray | None:
        if not self._is_open or self._index >= len(self._files):
            if self._is_open:
                self._is_open = False
                logger.info("Folder replay ended after %d frames", self._index)
            return None

        now = time.time()
        elapsed = now - self._last_frame_time
        if elapsed < self._frame_interval:
            return None

        from PIL import Image
        img = Image.open(self._files[self._index]).convert("RGB")
        frame = np.array(img)
        self._last_frame_time = now
        self._index += 1
        return frame

    def close(self) -> None:
        self._is_open = False
        logger.info("Closed folder frame source after %d frames", self._index)

    def is_open(self) -> bool:
        return self._is_open

    @property
    def frame_count(self) -> int:
        return self._index

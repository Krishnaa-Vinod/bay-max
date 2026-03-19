"""Frame source implementations for webcam and video replay."""

import logging
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
        """Open preferred camera index and fall back only if unavailable."""
        preferred = int(self._source)
        preferred_cap = cv2_module.VideoCapture(preferred)
        if preferred_cap.isOpened():
            ret, _ = preferred_cap.read()
            if ret:
                self._cap = preferred_cap
                self._resolved_source = preferred
                logger.info("Using preferred camera index %d", preferred)
                return
            preferred_cap.release()
        else:
            preferred_cap.release()

        for idx in [i for i in range(0, 8) if i != preferred]:
            cap = cv2_module.VideoCapture(idx)
            if not cap.isOpened():
                cap.release()
                continue

            ret, _ = cap.read()
            if not ret:
                cap.release()
                continue
            self._cap = cap
            self._resolved_source = idx
            logger.warning(
                "Requested camera index %d unavailable; using fallback camera %d",
                preferred,
                self._resolved_source,
            )
            return

        raise RuntimeError(f"Failed to open webcam source: {preferred}")

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

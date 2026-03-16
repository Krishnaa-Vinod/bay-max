"""Capture module interfaces for video/webcam ingestion."""

from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID


class FrameSource(ABC):
    """Abstract interface for a source of video frames."""

    @abstractmethod
    def open(self) -> None:
        """Open the frame source."""
        ...

    @abstractmethod
    def read_frame(self) -> Any | None:
        """Read a single frame. Returns None if no frame available."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release resources."""
        ...

    @abstractmethod
    def is_open(self) -> bool:
        """Check if the source is currently open."""
        ...


class StubFrameSource(FrameSource):
    """Stub frame source for testing without a camera."""

    def __init__(self) -> None:
        self._open = False

    def open(self) -> None:
        self._open = True

    def read_frame(self) -> Any | None:
        if not self._open:
            return None
        return {"stub_frame": True, "width": 640, "height": 480}

    def close(self) -> None:
        self._open = False

    def is_open(self) -> bool:
        return self._open


class SessionTimer:
    """Tracks timing for interaction sessions."""

    def __init__(self, session_id: UUID) -> None:
        self.session_id = session_id
        self._frame_count = 0

    def tick(self) -> int:
        """Increment and return the frame count."""
        self._frame_count += 1
        return self._frame_count

    @property
    def frame_count(self) -> int:
        return self._frame_count

"""Session lifecycle supervisor for automatic session management."""

import logging
import time
from uuid import UUID

from baymax.live.schemas import (
    SessionLifecycleEvent,
    SessionLifecycleState,
)

logger = logging.getLogger(__name__)


class SessionSupervisor:
    """Manages automatic session lifecycle based on presence/absence."""

    def __init__(
        self,
        presence_min_frames: int = 2,
        absence_timeout_sec: float = 30.0,
        resume_window_sec: float = 300.0,
    ) -> None:
        self._presence_min_frames = presence_min_frames
        self._absence_timeout_sec = absence_timeout_sec
        self._resume_window_sec = resume_window_sec

        # Internal state
        self._state = SessionLifecycleState.IDLE
        self._current_session_id: UUID | None = None
        self._current_user_id: UUID | None = None
        self._consecutive_presence: int = 0
        self._consecutive_absence: int = 0
        self._last_presence_time: float = 0.0
        self._last_absence_start: float = 0.0
        self._session_started_at: float = 0.0
        self._paused_at: float = 0.0
        self._events: list[SessionLifecycleEvent] = []

        # For resume tracking - maps user_id to (session_id, paused_at)
        self._paused_sessions: dict[UUID | None, tuple[UUID, float]] = {}

    @property
    def state(self) -> SessionLifecycleState:
        return self._state

    @property
    def current_session_id(self) -> UUID | None:
        return self._current_session_id

    @property
    def current_user_id(self) -> UUID | None:
        return self._current_user_id

    @property
    def events(self) -> list[SessionLifecycleEvent]:
        return list(self._events)

    def update(
        self,
        person_present: bool,
        user_id: UUID | None = None,
    ) -> list[SessionLifecycleEvent]:
        """Update presence state and return any lifecycle events triggered.

        Called once per analysis frame (not every preview frame).
        """
        now = time.time()
        new_events: list[SessionLifecycleEvent] = []

        if person_present:
            self._consecutive_presence += 1
            self._consecutive_absence = 0
            self._last_presence_time = now
        else:
            self._consecutive_absence += 1
            if self._consecutive_absence == 1:
                self._last_absence_start = now

        if self._state == SessionLifecycleState.IDLE:
            if person_present and self._consecutive_presence >= self._presence_min_frames:
                # Check if we should resume a paused session
                resumed = self._try_resume(user_id, now)
                if resumed:
                    evt = SessionLifecycleEvent(
                        event_type="session_resumed",
                        session_id=self._current_session_id,
                        user_id=user_id,
                        reason="User returned within resume window",
                    )
                    new_events.append(evt)
                    self._state = SessionLifecycleState.ACTIVE
                else:
                    # Signal that a new session should be started
                    evt = SessionLifecycleEvent(
                        event_type="session_started",
                        session_id=None,  # Will be set by the live runtime
                        user_id=user_id,
                        reason="Presence threshold met",
                    )
                    new_events.append(evt)
                    self._state = SessionLifecycleState.ACTIVE
                    self._session_started_at = now

                self._current_user_id = user_id

        elif self._state == SessionLifecycleState.ACTIVE:
            # Check for user change
            if person_present and user_id != self._current_user_id:
                self._current_user_id = user_id

            # Check for absence timeout
            if not person_present:
                absence_duration = now - self._last_absence_start
                if absence_duration >= self._absence_timeout_sec:
                    # Pause the session
                    evt = SessionLifecycleEvent(
                        event_type="session_paused",
                        session_id=self._current_session_id,
                        user_id=self._current_user_id,
                        reason=f"Absent for {absence_duration:.0f}s",
                    )
                    new_events.append(evt)
                    self._paused_at = now
                    self._state = SessionLifecycleState.PAUSED

                    # Store for potential resume
                    if self._current_session_id:
                        self._paused_sessions[self._current_user_id] = (
                            self._current_session_id,
                            now,
                        )

        elif self._state == SessionLifecycleState.PAUSED:
            if person_present and self._consecutive_presence >= self._presence_min_frames:
                # Check if we should resume or start new
                paused_duration = now - self._paused_at
                if paused_duration <= self._resume_window_sec:
                    # Resume
                    evt = SessionLifecycleEvent(
                        event_type="session_resumed",
                        session_id=self._current_session_id,
                        user_id=user_id,
                        reason=f"Returned after {paused_duration:.0f}s (within resume window)",
                    )
                    new_events.append(evt)
                    self._state = SessionLifecycleState.ACTIVE
                    self._current_user_id = user_id
                else:
                    # End old session, start new
                    end_evt = SessionLifecycleEvent(
                        event_type="session_ended",
                        session_id=self._current_session_id,
                        user_id=self._current_user_id,
                        reason=(
                            f"Resume window expired"
                            f" ({paused_duration:.0f}s"
                            f" > {self._resume_window_sec:.0f}s)"
                        ),
                    )
                    new_events.append(end_evt)

                    start_evt = SessionLifecycleEvent(
                        event_type="session_started",
                        session_id=None,
                        user_id=user_id,
                        reason="New session after expired resume window",
                    )
                    new_events.append(start_evt)
                    self._state = SessionLifecycleState.ACTIVE
                    self._current_user_id = user_id
                    self._session_started_at = now

            # Also check if the pause has exceeded resume window without return
            elif not person_present:
                paused_duration = now - self._paused_at
                if paused_duration > self._resume_window_sec:
                    evt = SessionLifecycleEvent(
                        event_type="session_ended",
                        session_id=self._current_session_id,
                        user_id=self._current_user_id,
                        reason=f"Resume window expired without return ({paused_duration:.0f}s)",
                    )
                    new_events.append(evt)
                    self._state = SessionLifecycleState.IDLE
                    self._current_session_id = None

        # Store events
        self._events.extend(new_events)
        return new_events

    def set_session_id(self, session_id: UUID) -> None:
        """Set the current session ID (called by the live runtime after creating a session)."""
        self._current_session_id = session_id

    def attach_active_session(self, session_id: UUID, user_id: UUID | None) -> None:
        """Attach an externally-created active session to supervisor state."""
        self._current_session_id = session_id
        self._current_user_id = user_id
        self._state = SessionLifecycleState.ACTIVE
        self._consecutive_presence = max(self._consecutive_presence, self._presence_min_frames)
        self._consecutive_absence = 0
        self._last_presence_time = time.time()

    def force_end(self) -> SessionLifecycleEvent | None:
        """Force-end the current session (e.g. on shutdown)."""
        if self._state in (SessionLifecycleState.ACTIVE, SessionLifecycleState.PAUSED):
            evt = SessionLifecycleEvent(
                event_type="session_ended",
                session_id=self._current_session_id,
                user_id=self._current_user_id,
                reason="Live runtime shutdown",
            )
            self._events.append(evt)
            self._state = SessionLifecycleState.IDLE
            self._current_session_id = None
            return evt
        return None

    def _try_resume(self, user_id: UUID | None, now: float) -> bool:
        """Try to resume a paused session for the given user."""
        key = user_id
        if key in self._paused_sessions:
            session_id, paused_at = self._paused_sessions[key]
            if now - paused_at <= self._resume_window_sec:
                self._current_session_id = session_id
                del self._paused_sessions[key]
                return True
            else:
                del self._paused_sessions[key]
        return False

    def reset(self) -> None:
        """Reset all state."""
        self._state = SessionLifecycleState.IDLE
        self._current_session_id = None
        self._current_user_id = None
        self._consecutive_presence = 0
        self._consecutive_absence = 0
        self._events.clear()
        self._paused_sessions.clear()

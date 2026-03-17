"""Event engine for detecting meaningful transitions in the live stream."""

import logging
import time
from uuid import UUID

from baymax.core.enums import EngagementLevel, PostureLabel
from baymax.live.schemas import CompanionEvent, CompanionEventType

logger = logging.getLogger(__name__)


class EventEngine:
    """Detects meaningful state transitions and emits companion events.

    Uses stability windows and hysteresis to avoid noise.
    """

    def __init__(
        self,
        stability_sec: float = 4.0,
    ) -> None:
        self._stability_sec = stability_sec

        # Previous stable state
        self._prev_person_present: bool = False
        self._prev_user_id: UUID | None = None
        self._prev_recognized: bool = False
        self._prev_posture: PostureLabel = PostureLabel.UNKNOWN
        self._prev_engagement: EngagementLevel = EngagementLevel.MEDIUM

        # Stability tracking for changes
        self._pending_posture: PostureLabel | None = None
        self._pending_posture_since: float = 0.0
        self._pending_engagement: EngagementLevel | None = None
        self._pending_engagement_since: float = 0.0

        # Quiet companionship tracking
        self._last_active_event_time: float = 0.0
        self._quiet_companionship_interval: float = 300.0

        self._events: list[CompanionEvent] = []

    @property
    def events(self) -> list[CompanionEvent]:
        return list(self._events)

    def set_quiet_interval(self, interval_sec: float) -> None:
        self._quiet_companionship_interval = interval_sec

    def update(
        self,
        person_present: bool,
        user_id: UUID | None = None,
        user_display_name: str | None = None,
        identity_confidence: float = 0.0,
        recognized: bool = False,
        posture: PostureLabel = PostureLabel.UNKNOWN,
        engagement: EngagementLevel = EngagementLevel.MEDIUM,
        session_id: UUID | None = None,
    ) -> list[CompanionEvent]:
        """Evaluate the current frame state and emit events on meaningful transitions."""
        now = time.time()
        new_events: list[CompanionEvent] = []

        # --- Arrival / Departure ---
        if person_present and not self._prev_person_present:
            if recognized and user_id is not None:
                evt = CompanionEvent(
                    event_type=CompanionEventType.RECOGNIZED_USER_ARRIVED,
                    session_id=session_id,
                    user_id=user_id,
                    user_display_name=user_display_name,
                    confidence=identity_confidence,
                    details={"posture": posture.value, "engagement": engagement.value},
                )
                new_events.append(evt)
                self._last_active_event_time = now
            else:
                evt = CompanionEvent(
                    event_type=CompanionEventType.UNKNOWN_USER_ARRIVED,
                    session_id=session_id,
                    confidence=identity_confidence,
                    details={"posture": posture.value, "engagement": engagement.value},
                )
                new_events.append(evt)
                self._last_active_event_time = now

            # Also fire generic person_arrived
            new_events.insert(0, CompanionEvent(
                event_type=CompanionEventType.PERSON_ARRIVED,
                session_id=session_id,
                user_id=user_id,
                user_display_name=user_display_name,
                confidence=identity_confidence,
            ))

        elif not person_present and self._prev_person_present:
            evt = CompanionEvent(
                event_type=CompanionEventType.PERSON_DEPARTED,
                session_id=session_id,
                user_id=self._prev_user_id,
            )
            new_events.append(evt)
            self._last_active_event_time = now

        # --- Recognition gained/lost (while person stays present) ---
        if person_present and self._prev_person_present:
            if recognized and not self._prev_recognized and user_id is not None:
                evt = CompanionEvent(
                    event_type=CompanionEventType.RECOGNITION_GAINED,
                    session_id=session_id,
                    user_id=user_id,
                    user_display_name=user_display_name,
                    confidence=identity_confidence,
                )
                new_events.append(evt)
                self._last_active_event_time = now

            elif not recognized and self._prev_recognized:
                evt = CompanionEvent(
                    event_type=CompanionEventType.RECOGNITION_LOST,
                    session_id=session_id,
                    user_id=self._prev_user_id,
                )
                new_events.append(evt)
                self._last_active_event_time = now

        # --- Posture change (with stability requirement) ---
        if (
            person_present
            and posture != PostureLabel.UNKNOWN
            and posture != self._prev_posture
            and self._prev_posture != PostureLabel.UNKNOWN
        ):
            if self._pending_posture == posture:
                if now - self._pending_posture_since >= self._stability_sec:
                    evt = CompanionEvent(
                        event_type=CompanionEventType.POSTURE_CHANGED,
                        session_id=session_id,
                        user_id=user_id,
                        details={
                            "from": self._prev_posture.value,
                            "to": posture.value,
                        },
                    )
                    new_events.append(evt)
                    self._prev_posture = posture
                    self._pending_posture = None
                    self._last_active_event_time = now
            else:
                self._pending_posture = posture
                self._pending_posture_since = now
        elif posture == self._prev_posture:
            self._pending_posture = None

        # --- Engagement change (with stability requirement) ---
        if (
            person_present
            and engagement != self._prev_engagement
        ):
            if self._pending_engagement == engagement:
                if now - self._pending_engagement_since >= self._stability_sec:
                    evt = CompanionEvent(
                        event_type=CompanionEventType.ENGAGEMENT_CHANGED,
                        session_id=session_id,
                        user_id=user_id,
                        details={
                            "from": self._prev_engagement.value,
                            "to": engagement.value,
                        },
                    )
                    new_events.append(evt)
                    self._prev_engagement = engagement
                    self._pending_engagement = None
                    self._last_active_event_time = now
            else:
                self._pending_engagement = engagement
                self._pending_engagement_since = now
        elif engagement == self._prev_engagement:
            self._pending_engagement = None

        # --- Quiet companionship ---
        if (
            person_present
            and self._last_active_event_time > 0
            and (now - self._last_active_event_time) >= self._quiet_companionship_interval
        ):
            evt = CompanionEvent(
                event_type=CompanionEventType.QUIET_COMPANIONSHIP_DUE,
                session_id=session_id,
                user_id=user_id,
                user_display_name=user_display_name,
            )
            new_events.append(evt)
            self._last_active_event_time = now  # Reset to avoid re-firing

        # Update previous state
        self._prev_person_present = person_present
        if person_present:
            self._prev_user_id = user_id
            self._prev_recognized = recognized
            # Only update prev posture/engagement if the new value is stable
            if posture != PostureLabel.UNKNOWN and self._pending_posture is None:
                self._prev_posture = posture
        else:
            self._prev_recognized = False

        self._events.extend(new_events)
        return new_events

    def reset(self) -> None:
        """Reset all state."""
        self._prev_person_present = False
        self._prev_user_id = None
        self._prev_recognized = False
        self._prev_posture = PostureLabel.UNKNOWN
        self._prev_engagement = EngagementLevel.MEDIUM
        self._pending_posture = None
        self._pending_engagement = None
        self._last_active_event_time = 0.0
        self._events.clear()

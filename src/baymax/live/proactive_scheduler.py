"""Cooldown-based proactive response scheduler."""

import logging
from datetime import datetime

from baymax.live.schemas import (
    CompanionEvent,
    CompanionEventType,
    CooldownState,
    ProactiveDecision,
)

logger = logging.getLogger(__name__)

# Event types that should trigger a proactive response
GREETING_EVENTS = {
    CompanionEventType.RECOGNIZED_USER_ARRIVED,
    CompanionEventType.UNKNOWN_USER_ARRIVED,
}

CHECK_IN_EVENTS = {
    CompanionEventType.POSTURE_CHANGED,
    CompanionEventType.ENGAGEMENT_CHANGED,
}

QUIET_EVENTS = {
    CompanionEventType.QUIET_COMPANIONSHIP_DUE,
}


class ProactiveScheduler:
    """Decides whether to respond proactively based on cooldowns."""

    def __init__(
        self,
        proactive_min_interval_sec: float = 30.0,
        any_response_min_interval_sec: float = 10.0,
        quiet_companionship_interval_sec: float = 300.0,
    ) -> None:
        self._proactive_min_interval = proactive_min_interval_sec
        self._any_response_min_interval = any_response_min_interval_sec
        self._quiet_interval = quiet_companionship_interval_sec
        self._cooldown = CooldownState()
        self._total_responses = 0
        self._total_suppressions = 0

    @property
    def cooldown_state(self) -> CooldownState:
        return self._cooldown

    @property
    def total_responses(self) -> int:
        return self._total_responses

    @property
    def total_suppressions(self) -> int:
        return self._total_suppressions

    def evaluate(self, event: CompanionEvent) -> ProactiveDecision:
        """Evaluate an event and decide whether to generate a proactive response."""
        now_dt = datetime.utcnow()

        # Check global any-response cooldown
        if self._cooldown.last_any_response_at:
            since = (now_dt - self._cooldown.last_any_response_at).total_seconds()
            if since < self._any_response_min_interval:
                remaining = self._any_response_min_interval - since
                self._total_suppressions += 1
                return ProactiveDecision(
                    event=event,
                    should_respond=False,
                    suppressed=True,
                    suppression_reason=f"Global cooldown: {remaining:.1f}s remaining",
                    cooldown_remaining_sec=remaining,
                )

        # Check proactive-specific cooldown
        if self._cooldown.last_proactive_response_at:
            since = (now_dt - self._cooldown.last_proactive_response_at).total_seconds()
            if since < self._proactive_min_interval:
                remaining = self._proactive_min_interval - since
                self._total_suppressions += 1
                return ProactiveDecision(
                    event=event,
                    should_respond=False,
                    suppressed=True,
                    suppression_reason=f"Proactive cooldown: {remaining:.1f}s remaining",
                    cooldown_remaining_sec=remaining,
                )

        # Check per-event-type dedup
        event_key = event.event_type.value
        if event_key in self._cooldown.last_event_responses:
            since = (now_dt - self._cooldown.last_event_responses[event_key]).total_seconds()
            if since < self._proactive_min_interval:
                remaining = self._proactive_min_interval - since
                self._total_suppressions += 1
                return ProactiveDecision(
                    event=event,
                    should_respond=False,
                    suppressed=True,
                    suppression_reason=(
                        f"Event-type '{event_key}' cooldown:"
                        f" {remaining:.1f}s remaining"
                    ),
                    cooldown_remaining_sec=remaining,
                )

        # Determine response trigger
        if event.event_type in GREETING_EVENTS:
            trigger = "arrival_greeting"
        elif event.event_type in CHECK_IN_EVENTS:
            trigger = "state_change_check_in"
        elif event.event_type in QUIET_EVENTS:
            trigger = "quiet_companionship"
        elif event.event_type == CompanionEventType.RECOGNITION_GAINED:
            trigger = "recognition_greeting"
        elif event.event_type == CompanionEventType.SESSION_RESUMED:
            trigger = "session_resume_greeting"
        else:
            # Non-response events (departures, losses, etc.)
            return ProactiveDecision(
                event=event,
                should_respond=False,
                suppressed=False,
                suppression_reason="Event type does not trigger proactive response",
            )

        return ProactiveDecision(
            event=event,
            should_respond=True,
            trigger_reason=trigger,
        )

    def record_response(self, event: CompanionEvent) -> None:
        """Record that a response was generated for cooldown tracking."""
        now_dt = datetime.utcnow()
        self._cooldown.last_any_response_at = now_dt
        self._cooldown.last_proactive_response_at = now_dt
        self._cooldown.last_event_responses[event.event_type.value] = now_dt

        if event.event_type in GREETING_EVENTS:
            self._cooldown.last_greeting_at = now_dt
        elif event.event_type in QUIET_EVENTS:
            self._cooldown.last_quiet_companionship_at = now_dt

        self._total_responses += 1

    def reset(self) -> None:
        """Reset all cooldown state."""
        self._cooldown = CooldownState()
        self._total_responses = 0
        self._total_suppressions = 0

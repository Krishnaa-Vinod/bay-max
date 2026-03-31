"""Affect-only proactive policy tests."""

from baymax.live.proactive_scheduler import ProactiveScheduler
from baymax.live.schemas import CompanionEvent, CompanionEventType


def test_affect_only_suppresses_arrival_events() -> None:
    scheduler = ProactiveScheduler(mode="affect_only")
    event = CompanionEvent(event_type=CompanionEventType.RECOGNIZED_USER_ARRIVED)
    decision = scheduler.evaluate(event)
    assert decision.should_respond is False
    assert decision.suppressed is True


def test_affect_only_allows_distress_event_after_silence() -> None:
    scheduler = ProactiveScheduler(mode="affect_only", affect_checkin_min_silence_sec=0.0)
    event = CompanionEvent(event_type=CompanionEventType.AFFECT_DISTRESS_PERSISTENT)
    decision = scheduler.evaluate(event)
    assert decision.should_respond is True
    assert decision.trigger_reason == "affect_distress_check_in"

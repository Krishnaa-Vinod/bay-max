# Event Engine

This document describes Bay-Max's companion event detection system (`src/baymax/live/event_engine.py`), introduced in Iteration 006.

## Purpose

The `EventEngine` translates raw perception results (is a person present? who are they? what is their posture and engagement?) into meaningful **companion events** that can drive proactive responses. It applies state-transition logic and stability gating to prevent noisy or jittery events.

## Event Types

| Event | Trigger | Notes |
|-------|---------|-------|
| `PERSON_ARRIVED` | First detection of a person after absence | Always fired on arrival (used to update state) |
| `PERSON_DEPARTED` | Loss of person after presence | Always fired on departure |
| `RECOGNIZED_USER_ARRIVED` | `PERSON_ARRIVED` + identity known | Subtype of arrival; enables personalized greeting |
| `UNKNOWN_USER_ARRIVED` | `PERSON_ARRIVED` + no identity match | Generic/safe response only |
| `RECOGNITION_GAINED` | Person present but unrecognized → recognized | Mid-session identity resolution |
| `RECOGNITION_LOST` | Person present but recognized → unrecognized | Mid-session track loss |
| `POSTURE_CHANGED` | Stable change in posture label | Subject to stability gating (see below) |
| `ENGAGEMENT_CHANGED` | Stable change in engagement level | Subject to stability gating (see below) |
| `QUIET_COMPANIONSHIP_DUE` | Continuous presence for `quiet_interval_sec` | Timer-based; resets on departure |
| `SESSION_RESUMED` | Session supervisor emits a resume event | Forwarded through event engine as a companion event |

## State Tracked

`EventEngine` maintains the following internal state:
- `_prev_person_present: bool` — was a person detected last frame?
- `_prev_user_id: UUID | None` — who was present last frame?
- `_prev_recognized: bool` — was the person recognized?
- `_prev_posture: str` — last confirmed posture label
- `_prev_engagement: str` — last confirmed engagement label

Plus pending states with timestamps for stability gating:
- `_pending_posture: str | None` — posture candidate awaiting stability
- `_pending_posture_since: float` — when the candidate was first seen
- `_pending_engagement: str | None` — engagement candidate awaiting stability
- `_pending_engagement_since: float`

Plus a quiet presence timer:
- `_presence_start_time: float | None` — when continuous presence began
- `_quiet_interval_sec: float` — configured quiet companionship interval

## Stability Gating

Posture and engagement changes require the new state to be stable for `stability_sec` (default: 4.0 seconds) before a `POSTURE_CHANGED` or `ENGAGEMENT_CHANGED` event is emitted. This prevents rapid oscillation from sensor jitter from triggering repeated responses.

Algorithm:
```
if new_posture != prev_posture:
    if pending_posture == new_posture:
        if elapsed(pending_posture_since) >= stability_sec:
            emit POSTURE_CHANGED
            prev_posture = new_posture
            clear pending_posture
    else:
        pending_posture = new_posture
        pending_posture_since = now
else:
    clear pending_posture  # candidate abandoned if reverted
```

## Quiet Companionship Timer

A monotonic timer tracks continuous presence duration. If the person remains present for `quiet_interval_sec` (default: 300 s = 5 minutes), a `QUIET_COMPANIONSHIP_DUE` event fires. The timer resets on departure or after firing.

This enables gentle check-in prompts for passive companion scenarios where the user is present but not actively interacting.

## update() Interface

```python
def update(
    self,
    person_present: bool,
    user_id: UUID | None,
    user_display_name: str | None,
    identity_confidence: float,
    recognized: bool,
    posture: str,
    engagement: str,
    session_id: UUID | None,
) -> list[CompanionEvent]:
```

Returns a list of zero or more `CompanionEvent` objects to be passed to `ProactiveScheduler`.

## Integration with SessionSupervisor

`EventEngine.update()` receives the same frame-by-frame state as `SessionSupervisor.update()`. `SESSION_RESUMED` events from the supervisor are forwarded as `CompanionEvent(event_type=SESSION_RESUMED, ...)` by `LiveRuntime` to trigger a re-greeting when a user returns.

## Jitter-Prevention Summary

| Source of Jitter | Mitigation |
|-----------------|------------|
| Frame-to-frame face detection fluctuation | `SessionSupervisor.presence_min_consecutive_frames` |
| Tracker ID oscillation from small motion | IoU+cosine tracker maintains continuity across frames |
| Posture label noise | Temporal majority-vote smoothing (Orchestrator) + `EventEngine` stability gating |
| Engagement score noise | Temporal majority-vote smoothing (Orchestrator) + `EventEngine` stability gating |
| Repeated arrival detection | `_prev_person_present` flag: arrival only fires on `False→True` transition |
| Response spam | `ProactiveScheduler` dual-layer cooldown |

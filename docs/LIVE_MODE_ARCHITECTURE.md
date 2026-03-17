# Live Mode Architecture

This document describes the architecture of Bay-Max's live webcam continuity layer, introduced in Iteration 006.

## Overview

Live mode turns Bay-Max from an upload-driven prototype into a continuous local companion. The live runtime reads frames from a webcam or replay source, runs perception at a configurable cadence, manages session lifecycle automatically, detects meaningful companion events, and generates memory-grounded proactive responses with cooldown enforcement.

## Module Layout

```
src/baymax/live/
  schemas.py            # Pydantic v2 data models for live runtime
  frame_source.py       # Webcam and replay frame source abstractions
  session_supervisor.py # Presence-based session lifecycle state machine
  event_engine.py       # Companion event detection (arrivals, transitions)
  proactive_scheduler.py# Cooldown enforcement and proactive response gating
  overlay.py            # Headless-safe HUD renderer
  artifact_logger.py    # JSONL/JSON/Markdown artifact logging
  runtime.py            # Top-level async orchestrator loop
```

## Runtime Loop

```
┌─────────────────────────────────────────────────────────────────────┐
│                          LiveRuntime.run()                          │
│                                                                     │
│  FrameSource ──► preview throttle ──► analysis throttle             │
│                                            │                        │
│                               _process_analysis_frame()             │
│                                            │                        │
│                    ┌───────────────────────┼──────────────────┐     │
│                    │                       │                  │     │
│             Orchestrator            SessionSupervisor    EventEngine│
│           .analyze_frame()        .update(present, uid)  .update() │
│                    │                       │                  │     │
│             PerceptionResult         LifecycleEvents   CompanionEvents│
│                    │                       │                  │     │
│                    └───────────────────────┼──────────────────┘     │
│                                            │                        │
│                               ProactiveScheduler.evaluate()         │
│                                            │                        │
│                                   should_respond?                   │
│                                      yes │  no (suppressed)         │
│                                          │         │                │
│                           _generate_proactive_response()  ArtifactLogger│
│                                          │         │                │
│                             Orchestrator.respond() │                │
│                             Orchestrator.add_turn()│                │
│                             ArtifactLogger.log_*() │                │
└──────────────────────────────────────────────────────────────────────┘
```

## Frame Source Abstraction

Both sources implement the same interface:
- `open()` → opens capture device or folder
- `read_frame() → np.ndarray | None` → returns RGB frame or None on exhaustion
- `close()` → releases resources
- `is_open() → bool`

**`OpenCVFrameSource`**: accepts `int` (webcam index) or `str` (video file path). Throttles to `target_fps` using elapsed-time checking. Converts BGR→RGB.

**`FolderFrameSource`**: reads sorted image files from a directory using Pillow. Marks `_open = False` when all images are exhausted.

## Session Lifecycle

`SessionSupervisor` implements a three-state machine:

```
IDLE ──[presence threshold met]──► ACTIVE ──[absence timeout]──► PAUSED
 ▲                                    ▲                               │
 │                                    │    [return within window]     │
 └──[resume window expired]───────────┘◄──────────────────────────────┘
```

- Configured via: `presence_min_consecutive_frames`, `absence_timeout_sec`, `session_resume_window_sec`
- Emits `SessionLifecycleEvent` objects: `session_started`, `session_paused`, `session_resumed`, `session_ended`
- After `session_started`, `LiveRuntime` calls `orchestrator.create_session()` and sets `supervisor.set_session_id(id)`
- After `session_ended`, `LiveRuntime` calls `orchestrator.consolidate()` to persist session memories

## Event Engine

See `docs/EVENT_ENGINE.md` for full details.

## Proactive Scheduler

`ProactiveScheduler` implements dual-layer cooldown:

1. **Global cooldown** (`any_response_min_interval_sec`): no response of any kind within this window
2. **Per-event-type cooldown** (`proactive_min_interval_sec`): no response of the same type within this window

Some event types always bypass all cooldowns (they fire once per state transition, so repeating is impossible in practice):
- `RECOGNIZED_USER_ARRIVED`, `UNKNOWN_USER_ARRIVED`, `RECOGNITION_GAINED`, `SESSION_RESUMED`

Some event types obey both cooldown layers:
- `POSTURE_CHANGED`, `ENGAGEMENT_CHANGED`, `QUIET_COMPANIONSHIP_DUE`

Some event types never trigger a proactive response:
- `PERSON_ARRIVED` (used internally to update state but not to greet)
- `PERSON_DEPARTED`, `RECOGNITION_LOST` (departure is noted in logs, not greeted)

## Overlay

`render_overlay()` draws a HUD directly on the NumPy frame. It uses only `numpy` array operations and does not require `cv2.imshow` — making it safe on headless machines. When `enable_live_overlay=False`, the raw frame is returned unmodified.

HUD shows: backend, session state, user ID (truncated), last event, last response text (word-wrapped), engagement, posture.

## Artifact Logging

`ArtifactLogger` writes to `live_artifact_dir`:

| File | Format | Description |
|------|--------|-------------|
| `events.jsonl` | JSONL | One line per companion event (including suppressions) |
| `responses.jsonl` | JSONL | One line per proactive response with trigger_reason, memory_refs |
| `session_timeline.json` | JSON | Session lifecycle events in order |
| `manifest.json` | JSON | `LiveRunSummary` metadata |
| `summary.md` | Markdown | Human-readable run summary |
| `snapshots/*.jpg` | JPEG | Annotated frames at key events (if camera available) |

## API Integration

Two FastAPI endpoints are added to `apps/api/main.py`:

- `GET /v1/live/status` → `LiveRuntimeStatus` (returns `live_mode_active=false` if no runtime is active)
- `POST /v1/live/control` → accepts `{"action": "stop"}` or `{"action": "start"}` (start must be done via CLI)

The CLI runner `scripts/live_companion.py` calls `set_live_runtime(runtime)` on the API module to wire the status endpoint.

## Configuration

All live settings are loaded from `BaymaxSettings` (Pydantic, from env vars). See `.env.example` for defaults and descriptions.

## Headless Compatibility

The live module is designed for Sol (NVIDIA HPC, no display):
- `cv2.imshow` calls are wrapped in `try/except` and silently skipped
- `FolderFrameSource` provides a headless replay path
- Overlay rendering uses NumPy array writes, not X11
- Automated tests use folder replay with `target_fps=100000` to exhaust sources without wall-clock delays

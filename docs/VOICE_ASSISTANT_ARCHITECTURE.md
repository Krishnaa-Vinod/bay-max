# Voice Assistant Architecture (Iteration 013)

This document describes the voice assistant runtime architecture introduced in Iteration 013. It consolidates Bay-Max's interaction model around a friendly voice-assistant core while preserving the richer perception and memory capabilities for optional use.

## Overview

Bay-Max operates as a **single primary assistant runtime** that owns the live user interaction loop. The core experience is:

1. User speaks (activated via VAD, wake phrase, or button)
2. Bay-Max listens until end-of-turn
3. Bay-Max processes: transcribe → classify → retrieve context → generate response
4. Bay-Max speaks the response
5. Short follow-up window for natural conversation continuation
6. Return to listening or idle

The assistant should feel like a helpful voice assistant first. Memory and perception enhance responses but don't dominate the experience.

## Assistant Voice State Machine

One canonical state machine governs the voice interaction loop. All backend code, WebSocket telemetry, and UI use these exact state names:

```
idle ─[activation]─► armed ─[voice detected]─► listening
                          ▲                         │
                          │   ┌──────────────────┬──┘
                          │   ▼                  ▼
                   cooldown ◄── speaking ◄── thinking
                          │        │
                          ▼        ▼
                       idle   interrupted ─► listening/idle
```

### States

| State | Description |
|-------|-------------|
| `idle` | Inactive, not listening. Waiting for activation. |
| `armed` | Ready to listen. Monitoring for speech (in continuous/wake modes). |
| `listening` | Actively capturing user speech. VAD has detected speech onset. |
| `thinking` | Processing: transcribing, retrieving context, generating response. |
| `speaking` | Delivering spoken response via TTS. Mic input suppressed. |
| `cooldown` | Brief pause after speaking. Follow-up window active. |
| `interrupted` | User barged in while assistant was speaking. |

### Transitions

All state transitions go through `AssistantStateMachine` (see `src/baymax/live/assistant_state.py`). This ensures:

- Consistent state across backend, WebSocket, and UI
- Explicit handling of edge cases (barge-in, follow-up, errors)
- Debuggable transition history

## Activation Modes

Three activation modes determine how Bay-Max transitions from idle/armed to listening:

### Push-to-Talk (`push_to_talk`)

- **Activation**: Explicit button press via API or UI
- **Best for**: Debugging, noisy environments, privacy-sensitive contexts
- **After speaking**: Returns to idle (button required for each turn)

Configuration:
```bash
BAYMAX_ACTIVATION_MODE=push_to_talk
```

### Continuous VAD (`continuous_vad`)

- **Activation**: Any speech detected by Voice Activity Detection
- **Best for**: Hands-free conversational use
- **After speaking**: Returns to armed (continues listening)
- **Requirements**: Good echo suppression, speaking lock, post-speech cooldown

Configuration (default):
```bash
BAYMAX_ACTIVATION_MODE=continuous_vad
```

### Wake Phrase Gate (`wake_phrase_gate`)

- **Activation**: Speech must begin with a wake phrase (e.g., "Hey Baymax")
- **Best for**: Shared environments, battery/resource-conscious scenarios
- **After speaking**: Returns to armed (wake phrase needed again)

Configuration:
```bash
BAYMAX_ACTIVATION_MODE=wake_phrase_gate
BAYMAX_WAKE_PHRASES=hey baymax,hi baymax,baymax
BAYMAX_WAKE_PHRASE_FUZZY_THRESHOLD=0.75
```

**⚠️ Provisional Implementation Note**: The current wake phrase detection uses ASR transcript matching. This is NOT equivalent to a production on-device wake-word model (like Porcupine or OpenWakeWord). The interface is designed so a dedicated wake-word backend can be swapped in without touching the rest of the runtime.

## Follow-Up Behavior

After Bay-Max finishes speaking, a **follow-up window** allows the user to continue the conversation naturally:

1. Bay-Max enters `cooldown` state after speaking
2. Follow-up window is active for `BAYMAX_FOLLOW_UP_WINDOW_SEC` (default: 4 seconds)
3. If user speaks during this window → transition to `listening` (no re-activation needed)
4. If window expires → transition to `armed` (continuous/wake) or `idle` (PTT)

This creates a natural conversational feel without requiring re-activation for each turn.

Configuration:
```bash
BAYMAX_FOLLOW_UP_WINDOW_SEC=4.0
BAYMAX_STAY_ARMED_AFTER_FOLLOW_UP=true
```

## Barge-In / Interrupt

Bay-Max supports barge-in: the user can interrupt while Bay-Max is speaking.

1. User starts speaking while Bay-Max is in `speaking` state
2. Transition to `interrupted` state
3. TTS playback stops
4. Bay-Max processes the barge-in utterance
5. Transition to `listening` to capture the user's input

The interrupt is exposed via:
- `POST /v1/live/voice/interrupt` (API)
- WebSocket event

## Core Loop Stages

### 1. Activation

The appropriate `ActivationGate` checks whether to proceed:
- `PushToTalkGate`: Button pressed?
- `ContinuousVADGate`: Any speech detected?
- `WakePhraseGate`: Wake phrase present?

### 2. Listen

- Microphone captures audio
- Silero VAD segments speech
- End-of-turn detection (silence timeout)

### 3. Transcribe & Classify

- faster-whisper ASR converts speech to text
- Echo suppression filters self-hearing
- Transcript quality assessment (low confidence → clarification prompt)
- Intent classification (direct question, follow-up, emotional share, etc.)

### 4. Retrieve Context (Optional)

- Memory retrieval (relevant prior interactions)
- Web search (if information is needed and web tools enabled)
- User recognition data (name, preferences)

These are **optional inputs** that enhance responses but don't block the flow.

### 5. Generate Response

- SupportivePlanner selects response strategy
- DialogueProvider generates text (answer-first for questions, empathy-first for emotions)
- Safety guardrails applied

### 6. Speak

- TTS synthesizes audio (Kokoro by default)
- Audio playback via sounddevice
- Speaking lock prevents self-hearing

### 7. Follow-Up Window

- Brief pause for user follow-up
- If user speaks → back to step 2
- If timeout → return to armed/idle

## Friendly Tone Guidelines

Bay-Max's voice-assistant persona is:

| Trait | Description |
|-------|-------------|
| Warm | Friendly and approachable, not robotic |
| Direct | Answer-first for questions; get to the point |
| Helpful | Practical guidance when appropriate |
| Brief | 1-3 sentences typical; respect user's time |

For **factual/task questions**: Answer first, then optional context.
For **emotional shares**: Acknowledge feelings first, then offer support.

See `docs/COMPANION_PERSONA.md` for the full style guide.

## Optional Specialist Delegation

Bay-Max uses a **single assistant with optional delegation** pattern, not a mandatory multi-agent architecture:

```
User Input
    │
    ▼
┌─────────────────────────────────────┐
│  Primary Assistant Runtime          │
│  (owns turn-taking, state machine)  │
└─────────────────┬───────────────────┘
                  │
                  ▼
        ┌─────────────────┐
        │ Optional Tools  │
        ├─────────────────┤
        │ • Web Search    │
        │ • Memory Query  │
        │ • Vision Context│
        └─────────────────┘
                  │
                  ▼
           Response
```

**Design principles**:
- The primary assistant handles most turns directly
- Specialists (web, memory, vision) are **tools/adapters**, not competing orchestrators
- Delegation happens behind one triage boundary
- The hot path (activation → listen → respond) stays simple

This keeps the realtime loop fast and debuggable while preserving extension points for future capabilities.

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /v1/live/status` | Returns `LiveRuntimeStatus` including `assistant_state`, `activation_mode` |
| `POST /v1/live/voice/interrupt` | Trigger barge-in interrupt |
| `POST /v1/live/mic/toggle` | Toggle microphone on/off |
| `POST /v1/live/text-input` | Submit text input (bypasses voice) |
| `WS /ws/live` | Real-time telemetry including state transitions |

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `BAYMAX_ACTIVATION_MODE` | `continuous_vad` | How the assistant activates |
| `BAYMAX_WAKE_PHRASES` | `hey baymax,hi baymax,baymax` | Wake phrases (comma-separated) |
| `BAYMAX_WAKE_PHRASE_FUZZY_THRESHOLD` | `0.75` | Fuzzy matching threshold |
| `BAYMAX_FOLLOW_UP_WINDOW_SEC` | `4.0` | Follow-up window duration |
| `BAYMAX_STAY_ARMED_AFTER_FOLLOW_UP` | `true` | Stay armed vs return to idle |
| `BAYMAX_POST_SPEECH_COOLDOWN_MS` | `1500` | Cooldown before accepting input |

## Files

| File | Purpose |
|------|---------|
| `src/baymax/live/schemas.py` | `AssistantVoiceState`, `ActivationMode` enums |
| `src/baymax/live/assistant_state.py` | `AssistantStateMachine` implementation |
| `src/baymax/audio/activation_gate.py` | `ActivationGate` interface and implementations |
| `src/baymax/config/settings.py` | Configuration settings |
| `tests/test_iteration_013.py` | State machine and activation tests |

## Future Extensions

The following are intentionally deferred to maintain focus on the voice-assistant core:

- **Dedicated wake-word engine**: Replace ASR-gated detection with Porcupine, OpenWakeWord, or similar
- **Proactive emotional check-ins**: Re-enable once voice-assistant core is stable
- **Multi-agent routing**: Internal specialist agents behind the single assistant boundary
- **Speaker diarization**: Multi-person support
- **Async SQLite migration**: Replace synchronous sqlite3 with aiosqlite

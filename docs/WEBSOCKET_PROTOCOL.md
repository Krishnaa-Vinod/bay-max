# Bay-Max WebSocket Protocol

This document describes the WebSocket protocol used by the Bay-Max Companion UI for real-time telemetry.

## Endpoint

```
ws://localhost:8000/ws/live
```

## Connection Lifecycle

1. Client connects to `/ws/live`
2. Server sends `bootstrap` message with initial state
3. Server sends buffered `pipeline_event` messages
4. Server sends periodic `snapshot` messages (~2 Hz)
5. Server sends real-time `pipeline_event` messages as they occur
6. Client can send `ping` messages; server responds with `pong`

## Message Types

### Bootstrap Message

Sent once when a client connects. Contains initial state.

```json
{
  "type": "bootstrap",
  "timestamp": "2026-03-18T14:32:07.423Z",
  "session": {
    "id": "ses_abc123",
    "state": "active",
    "duration_sec": 847,
    "turn_count": 12
  },
  "perception": {
    "face_detected": true,
    "user_name": "Krishna",
    "user_id": "usr_def456",
    "recognition_confidence": 0.91,
    "posture": "upright",
    "engagement": 0.73,
    "valence": 0.4,
    "arousal": 0.5,
    "affect_confidence": 0.61,
    "affect_backend": "mediapipe",
    "affect_stable_duration_sec": 15.2,
    "affect_debug": "positive, moderate (conf: 0.61)"
  },
  "pipeline_state": "ACTIVE",
  "last_response": "I remember you mentioned...",
  "last_spoken_text": "I remember you mentioned...",
  "memory_hits": [
    {"text": "Krishna is building Bay-Max", "score": 0.87},
    {"text": "prefers warm greetings", "score": 0.79}
  ],
  "cooldown_remaining_sec": 0,
  "last_event": "RECOGNIZED_USER_ARRIVAL",
  "frame_count": 1523,
  "analysis_count": 152,
  "uptime_sec": 304.5,
  "speech_input": {
    "enabled": true,
    "listening": true,
    "vad_active": false,
    "stt_backend": "faster_whisper",
    "speaking_lock": false,
    "last_heard": "How are you today?",
    "mic_mode": "vad"
  },
  "tts": {
    "backend": "kokoro",
    "voice": "af_heart",
    "queue_depth": 0
  }
}
```

### Snapshot Message

Sent periodically (~2 Hz) with current runtime state. Same schema as bootstrap.

```json
{
  "type": "snapshot",
  "timestamp": "2026-03-18T14:32:08.023Z",
  // ... same fields as bootstrap
}
```

### Pipeline Event Message

Sent in real-time when pipeline stages complete.

```json
{
  "type": "pipeline_event",
  "timestamp": "2026-03-18T14:32:03.891Z",
  "stage": "memory_retrieval",
  "status": "complete",
  "detail": "3 memories retrieved in 47ms"
}
```

**Stage values:**
- `recognition` - Face recognition
- `perception` - Posture/engagement analysis
- `affect` - Emotion analysis
- `memory_retrieval` - Memory query
- `planning` - Response strategy selection
- `dialogue` - Response generation
- `tts` - Speech synthesis
- `safety` - Safety gate checks
- `websocket` - Connection events

**Status values:**
- `pending` - Not yet started
- `in_progress` - Currently running
- `complete` - Successfully completed
- `blocked` - Blocked by safety gate or cooldown
- `suppressed` - Rate-limited or suppressed
- `error` - Failed with error

### Speech Event Message

Sent for speech-related events.

```json
{
  "type": "speech_event",
  "timestamp": "2026-03-18T14:32:05.123Z",
  "stage": "speech_input",
  "status": "complete",
  "detail": "Transcribed: 'How are you today?'"
}
```

### Memory Event Message

Sent for memory operations.

```json
{
  "type": "memory_event",
  "timestamp": "2026-03-18T14:32:04.567Z",
  "stage": "memory_store",
  "status": "complete",
  "detail": "Stored observation: user mentioned project deadline"
}
```

### Error Message

Sent when errors occur.

```json
{
  "type": "error",
  "timestamp": "2026-03-18T14:32:10.000Z",
  "error": "Frame analysis failed",
  "detail": "No face detected in frame"
}
```

## Field Reference

### Session State

| Value | Description |
|-------|-------------|
| `idle` | No active session |
| `active` | Session in progress |
| `paused` | Session temporarily paused (user away) |
| `ended` | Session completed |

### Pipeline State

| Value | Description |
|-------|-------------|
| `OFFLINE` | Runtime not active |
| `IDLE` | Runtime active, no session |
| `ACTIVE` | Session active, processing |
| `LISTENING` | VAD detected speech |
| `THINKING` | Processing response |
| `SPEAKING` | TTS playback active |
| `COOLDOWN` | Response cooldown period |

## Client Implementation

### JavaScript Example

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/live');

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);

  switch (message.type) {
    case 'bootstrap':
    case 'snapshot':
      updateUI(message);
      break;
    case 'pipeline_event':
    case 'speech_event':
    case 'memory_event':
      addToActivityFeed(message);
      break;
    case 'error':
      showError(message.error);
      break;
  }
};

// Keep-alive ping
setInterval(() => {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send('ping');
  }
}, 30000);
```

### Reconnection

Clients should implement exponential backoff reconnection:

```javascript
let reconnectAttempts = 0;
const maxAttempts = 10;

function reconnect() {
  if (reconnectAttempts >= maxAttempts) return;

  const delay = 1000 * Math.pow(2, reconnectAttempts);
  setTimeout(() => {
    reconnectAttempts++;
    connect();
  }, delay);
}
```

## Rate Limiting

- Snapshots: ~2 per second
- Events: Real-time (no limit)
- Client pings: Recommended every 30 seconds

## Error Handling

If the WebSocket connection fails:
1. Display "Disconnected" status in UI
2. Attempt reconnection with backoff
3. Show last known state with "stale" indicator
4. Resume normal operation when reconnected

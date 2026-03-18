# Bay-Max Companion UI Architecture

The Companion UI is a single-page diagnostic web interface that visualizes the live Bay-Max runtime state in real-time.

## Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                       Bay-Max Companion UI                        │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │ What Bay-Max │  │  What Bay-Max    │  │  What Bay-Max    │   │
│  │    Sees      │  │   is Doing       │  │   Remembers      │   │
│  │              │  │                  │  │                  │   │
│  │ • Camera     │  │ • Animated Face  │  │ • Memory Hits    │   │
│  │ • Identity   │  │ • Activity Feed  │  │ • Semantic Facts │   │
│  │ • Posture    │  │ • Last Response  │  │ • Memory Stats   │   │
│  │ • Engagement │  │ • Text Input     │  │                  │   │
│  │ • Affect     │  │ • Mic Control    │  │                  │   │
│  │ • Session    │  │                  │  │                  │   │
│  └──────────────┘  └──────────────────┘  └──────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
              │                    │                    │
              └────────────────────┼────────────────────┘
                                   │
                          WebSocket + HTTP
                                   │
┌──────────────────────────────────────────────────────────────────┐
│                       FastAPI Backend                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Live Runtime                           │   │
│  │  • Frame processing      • Event engine                  │   │
│  │  • Perception pipeline   • Proactive scheduler           │   │
│  │  • Memory retrieval      • Speech I/O                    │   │
│  │  • Affect analysis       • TTS output                    │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

## Design Principles

1. **Backend-Owned Pipeline**: The Python live runtime owns the camera and perception pipeline. The UI observes but does not replace the runtime.

2. **Real State Only**: The UI displays actual runtime telemetry, not mocked data.

3. **Single Screen**: All diagnostic info is visible at once without tabs or modals.

4. **Desktop-First**: Optimized for laptop/desktop debugging workflows.

5. **Dark Theme**: Diagnostic-focused design with high contrast.

## Technology Stack

### Frontend
- **Vite** - Fast development server and build tool
- **React 18** - UI framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Utility-first styling
- **clsx** - Conditional class composition

### Backend Integration
- **WebSocket** (`/ws/live`) - Real-time telemetry streaming
- **HTTP REST** - Bootstrap state, frame endpoint, text input

## Component Architecture

```
src/
├── App.tsx                    # Main layout, state management
├── main.tsx                   # React entry point
├── index.css                  # Tailwind base + custom styles
├── types/
│   └── live.ts                # TypeScript type definitions
├── lib/
│   ├── ws.ts                  # WebSocket connection manager
│   └── api.ts                 # HTTP API client
├── components/
│   ├── CameraFeed.tsx         # Live annotated frame display
│   ├── ActivityFeed.tsx       # Color-coded event timeline
│   ├── BayMaxFace.tsx         # State-driven animated avatar
│   ├── MemoryPanel.tsx        # Retrieved memories + facts
│   ├── AffectPlot.tsx         # Valence/arousal visualization
│   ├── StatusBar.tsx          # Connection + pipeline status
│   ├── SessionInfo.tsx        # Session metadata display
│   └── TextInput.tsx          # Text + mic input controls
└── hooks/                     # (Future) Custom React hooks
```

## Data Flow

### WebSocket Telemetry

```
LiveRuntime → WebSocket → UI App → Components
     │              │
     └── Snapshot ──┘ (every 500ms)
           • Session state
           • Perception data
           • Affect readings
           • Memory hits
           • Pipeline state

     └── Events ────┘ (real-time)
           • Pipeline stages
           • Speech events
           • Memory operations
           • Errors
```

### Frame Updates

```
LiveRuntime.render_overlay()
     │
     └── update_annotated_frame() → HTTP cache
                                        │
     UI ← GET /v1/live/frame/latest ────┘
          (refreshed every 200ms)
```

## State Management

The UI uses React's `useState` and `useCallback` for state management:

- **Snapshot state**: Full runtime snapshot updated via WebSocket
- **Event buffer**: Ring buffer of last 100 events for activity feed
- **Affect history**: Rolling window for valence/arousal plot
- **Connection state**: WebSocket connection status

## Avatar State Mapping

| Backend State | Avatar Display | Visual |
|--------------|----------------|--------|
| OFFLINE | Gray, static | Dim eyes |
| IDLE | White, blinking | Normal eyes, slow blink |
| ACTIVE | White, alert | Normal eyes |
| LISTENING | White, wide eyes | Scaled up eyes |
| THINKING | White, looking up | Eyes shifted up |
| SPEAKING | White, mouth animation | Mouth opens/closes |
| COOLDOWN | White, relaxed | Normal, cooldown badge |

## Security Considerations

- UI is diagnostic-only, intended for local development
- No authentication required (single-user local use)
- CORS configured for localhost development ports
- No sensitive data exposed beyond session scope

## Performance

- Frame refresh: ~5 FPS (configurable)
- WebSocket updates: ~2 Hz snapshots
- Event buffer: Max 100 entries (prevents memory growth)
- Affect history: Max 60 samples (~30 seconds)

## Future Enhancements

- [ ] Memory edit/correction controls
- [ ] Session timeline export
- [ ] Configurable refresh rates
- [ ] Multiple user support
- [ ] Mobile-responsive layout

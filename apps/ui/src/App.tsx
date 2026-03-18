import { useState, useEffect, useCallback, useRef } from 'react';
import { LiveWebSocket, ConnectionState } from '@/lib/ws';
import type { SnapshotMessage, PipelineEvent, ActivityEvent, MemoryHit } from '@/types/live';

// Components
import { CameraFeed } from '@/components/CameraFeed';
import { ActivityFeed } from '@/components/ActivityFeed';
import { BayMaxFace } from '@/components/BayMaxFace';
import { MemoryPanel } from '@/components/MemoryPanel';
import { AffectPlot } from '@/components/AffectPlot';
import { StatusBar } from '@/components/StatusBar';
import { SessionInfo } from '@/components/SessionInfo';
import { TextInput } from '@/components/TextInput';

const MAX_EVENTS = 100;

function App() {
  // Connection state
  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected');
  const wsRef = useRef<LiveWebSocket | null>(null);

  // Snapshot state
  const [snapshot, setSnapshot] = useState<SnapshotMessage | null>(null);

  // Activity feed
  const [events, setEvents] = useState<ActivityEvent[]>([]);

  // Affect history for plot
  const [affectHistory, setAffectHistory] = useState<Array<{
    timestamp: number;
    valence: number;
    arousal: number;
  }>>([]);

  // Add event to activity feed
  const addEvent = useCallback((event: PipelineEvent) => {
    const activityEvent: ActivityEvent = {
      id: `${event.timestamp}-${event.stage}-${Math.random().toString(36).slice(2, 8)}`,
      timestamp: new Date(event.timestamp),
      stage: event.stage,
      status: event.status,
      detail: event.detail,
      type: event.type,
    };

    setEvents((prev) => {
      const updated = [...prev, activityEvent];
      // Keep only the last MAX_EVENTS
      return updated.slice(-MAX_EVENTS);
    });
  }, []);

  // Update affect history
  const updateAffectHistory = useCallback((snapshot: SnapshotMessage) => {
    if (snapshot.perception.affect_confidence > 0.3) {
      setAffectHistory((prev) => {
        const updated = [
          ...prev,
          {
            timestamp: Date.now(),
            valence: snapshot.perception.valence,
            arousal: snapshot.perception.arousal,
          },
        ];
        // Keep last 60 data points (about 30 seconds at 2 updates/sec)
        return updated.slice(-60);
      });
    }
  }, []);

  // Handle snapshot message
  const handleSnapshot = useCallback((data: SnapshotMessage) => {
    setSnapshot(data);
    updateAffectHistory(data);
  }, [updateAffectHistory]);

  // Initialize WebSocket
  useEffect(() => {
    const ws = new LiveWebSocket();
    wsRef.current = ws;

    ws.setCallbacks({
      onConnectionChange: setConnectionState,
      onSnapshot: handleSnapshot,
      onEvent: addEvent,
      onError: (error) => {
        console.error('WebSocket error:', error);
        addEvent({
          type: 'error',
          timestamp: new Date().toISOString(),
          stage: 'websocket',
          status: 'error',
          detail: error,
        });
      },
    });

    ws.connect();

    return () => {
      ws.disconnect();
    };
  }, [handleSnapshot, addEvent]);

  // Frame refresh URL with cache busting
  const frameUrl = `/v1/live/frame/latest?t=${snapshot?.frame_count ?? 0}`;

  return (
    <div className="min-h-screen bg-baymax-bg text-white p-4">
      {/* Header */}
      <header className="mb-4">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold text-white">
            Bay-Max Companion
            <span className="text-sm font-normal text-gray-400 ml-2">Diagnostic UI</span>
          </h1>
          <StatusBar
            connectionState={connectionState}
            pipelineState={snapshot?.pipeline_state ?? 'OFFLINE'}
            uptime={snapshot?.uptime_sec ?? 0}
          />
        </div>
      </header>

      {/* Main three-column layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 h-[calc(100vh-8rem)]">
        {/* Left column: What Bay-Max sees */}
        <div className="bg-baymax-card rounded-lg p-4 flex flex-col gap-4 overflow-hidden">
          <h2 className="text-lg font-semibold text-gray-300 border-b border-baymax-border pb-2">
            What Bay-Max Sees
          </h2>

          {/* Camera feed */}
          <CameraFeed
            frameUrl={frameUrl}
            isConnected={connectionState === 'connected'}
          />

          {/* Identity info */}
          <div className="bg-baymax-bg rounded p-3">
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div>
                <span className="text-gray-400">Identity:</span>
                <span className="ml-2 text-white font-medium">
                  {snapshot?.perception.user_name ?? 'Unknown'}
                </span>
              </div>
              <div>
                <span className="text-gray-400">Confidence:</span>
                <span className="ml-2 text-white">
                  {((snapshot?.perception.recognition_confidence ?? 0) * 100).toFixed(0)}%
                </span>
              </div>
              <div>
                <span className="text-gray-400">Posture:</span>
                <span className="ml-2 text-white capitalize">
                  {snapshot?.perception.posture ?? 'unknown'}
                </span>
              </div>
              <div>
                <span className="text-gray-400">Engagement:</span>
                <span className="ml-2 text-white">
                  {((snapshot?.perception.engagement ?? 0.5) * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          </div>

          {/* Affect plot */}
          <AffectPlot
            history={affectHistory}
            currentValence={snapshot?.perception.valence ?? 0}
            currentArousal={snapshot?.perception.arousal ?? 0}
            confidence={snapshot?.perception.affect_confidence ?? 0}
            backend={snapshot?.perception.affect_backend ?? ''}
          />

          {/* Session info */}
          <SessionInfo
            sessionId={snapshot?.session.id ?? null}
            sessionState={snapshot?.session.state ?? 'idle'}
            duration={snapshot?.session.duration_sec ?? 0}
            turnCount={snapshot?.session.turn_count ?? 0}
            lastEvent={snapshot?.last_event ?? ''}
          />
        </div>

        {/* Center column: What Bay-Max is doing */}
        <div className="bg-baymax-card rounded-lg p-4 flex flex-col gap-4 overflow-hidden">
          <h2 className="text-lg font-semibold text-gray-300 border-b border-baymax-border pb-2">
            What Bay-Max is Doing
          </h2>

          {/* Bay-Max Face */}
          <BayMaxFace
            state={snapshot?.pipeline_state ?? 'OFFLINE'}
            isSpeaking={snapshot?.pipeline_state === 'SPEAKING'}
          />

          {/* Activity feed */}
          <ActivityFeed events={events} />

          {/* Last response */}
          <div className="bg-baymax-bg rounded p-3">
            <div className="text-xs text-gray-400 mb-1">Last Response</div>
            <div className="text-sm text-green-400 min-h-[3rem] max-h-[6rem] overflow-y-auto">
              {snapshot?.last_response || <span className="text-gray-500 italic">No response yet</span>}
            </div>
          </div>

          {/* Input controls */}
          <TextInput
            speechEnabled={snapshot?.speech_input.enabled ?? false}
            isListening={snapshot?.speech_input.listening ?? false}
          />
        </div>

        {/* Right column: What Bay-Max remembers */}
        <div className="bg-baymax-card rounded-lg p-4 flex flex-col gap-4 overflow-hidden">
          <h2 className="text-lg font-semibold text-gray-300 border-b border-baymax-border pb-2">
            What Bay-Max Remembers
          </h2>

          {/* Memory panel */}
          <MemoryPanel
            memoryHits={snapshot?.memory_hits ?? []}
            userId={snapshot?.perception.user_id ?? null}
          />
        </div>
      </div>
    </div>
  );
}

export default App;

import { useState, useEffect, useCallback, useRef } from 'react';
import { LiveWebSocket, ConnectionState } from '@/lib/ws';
import { getUIState } from '@/lib/api';
import type { SnapshotMessage, PipelineEvent, ActivityEvent } from '@/types/live';

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
  const [memoryRefreshKey, setMemoryRefreshKey] = useState(0);
  const hasSeenBootstrapRef = useRef(false);
  const pendingReconnectRecoveryRef = useRef(false);
  const lastRecognitionStateRef = useRef<string>('no_face');
  const lastUserIdRef = useRef<string | null>(null);

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

    if (!hasSeenBootstrapRef.current) {
      hasSeenBootstrapRef.current = true;
    }

    if (pendingReconnectRecoveryRef.current) {
      pendingReconnectRecoveryRef.current = false;
      setEvents((prev) => {
        const withoutWsErrors = prev.filter(
          (evt) => !(evt.stage === 'websocket' && evt.status === 'error')
        );
        return [
          ...withoutWsErrors,
          {
            id: `ws-recovered-${Date.now()}`,
            timestamp: new Date(),
            stage: 'websocket',
            status: 'complete',
            detail: 'Telemetry recovered after reconnect',
            type: 'pipeline_event',
          },
        ].slice(-MAX_EVENTS);
      });
    }

    const recognitionState = data.perception.recognition_state;
    const userId = data.perception.user_id;
    const shouldRefreshForRecognition =
      lastRecognitionStateRef.current !== recognitionState
      || lastUserIdRef.current !== userId;

    if (shouldRefreshForRecognition) {
      setMemoryRefreshKey((prev) => prev + 1);
      lastRecognitionStateRef.current = recognitionState;
      lastUserIdRef.current = userId;
    }
  }, [updateAffectHistory]);

  const handleConnectionChange = useCallback((state: ConnectionState) => {
    setConnectionState(state);
    if ((state === 'disconnected' || state === 'error') && hasSeenBootstrapRef.current) {
      pendingReconnectRecoveryRef.current = true;
    }
  }, []);

  const addUiEvent = useCallback((event: Omit<ActivityEvent, 'id'>) => {
    setEvents((prev) => {
      const next: ActivityEvent = {
        id: `${event.timestamp.toISOString()}-${event.stage}-${Math.random().toString(36).slice(2, 8)}`,
        ...event,
      };
      return [...prev, next].slice(-MAX_EVENTS);
    });
  }, []);

  const handleTextSubmitSuccess = useCallback((userText: string, response: { response_text: string; memory_refs: Array<{ text: string; score: number | null }>; session_id: string | null }) => {
    addUiEvent({
      timestamp: new Date(),
      stage: 'text_input',
      status: 'complete',
      detail: `User: ${userText}`,
      type: 'pipeline_event',
    });

    addUiEvent({
      timestamp: new Date(),
      stage: 'dialogue',
      status: 'complete',
      detail: `Assistant: ${response.response_text}`,
      type: 'pipeline_event',
    });

    setSnapshot((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        session: {
          ...prev.session,
          id: response.session_id ?? prev.session.id,
        },
        last_response: response.response_text,
        memory_hits: response.memory_refs,
      };
    });

    setMemoryRefreshKey((prev) => prev + 1);
  }, [addUiEvent]);

  const handleMicToggled = useCallback((listening: boolean) => {
    addUiEvent({
      timestamp: new Date(),
      stage: 'speech_input',
      status: 'complete',
      detail: listening ? 'Listening started' : 'Listening stopped',
      type: 'speech_event',
    });
  }, [addUiEvent]);

  // Initialize WebSocket
  useEffect(() => {
    const ws = new LiveWebSocket();
    wsRef.current = ws;

    ws.setCallbacks({
      onConnectionChange: handleConnectionChange,
      onSnapshot: handleSnapshot,
      onEvent: addEvent,
      onError: (error) => {
        addEvent({
          type: 'pipeline_event',
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
  }, [handleSnapshot, addEvent, handleConnectionChange]);

  useEffect(() => {
    const bootstrapFromHttp = async () => {
      try {
        const state = await getUIState();
        setSnapshot((prev) => {
          if (prev) {
            return prev;
          }
          return {
            type: 'bootstrap',
            timestamp: state.server_time,
            session: {
              id: state.session_id,
              state: state.session_state,
              duration_sec: 0,
              turn_count: 0,
            },
            perception: {
              face_detected: state.face_detected,
              recognition_state: state.recognition_state,
              face_match_threshold: state.face_match_threshold,
              user_name: state.user_name,
              user_id: state.user_id,
              recognition_confidence: null,
              posture: null,
              engagement: null,
              valence: 0,
              arousal: 0,
              affect_confidence: 0,
              affect_backend: state.affect_backend,
              affect_stable_duration_sec: 0,
              affect_debug: '',
            },
            pipeline_state: state.live_mode_active ? 'IDLE' : 'OFFLINE',
            last_response: state.last_response,
            last_spoken_text: '',
            memory_hits: [],
            cooldown_remaining_sec: 0,
            last_event: state.last_event,
            frame_count: state.frame_count,
            analysis_count: state.analysis_count,
            uptime_sec: state.uptime_sec,
            speech_input: {
              enabled: state.speech_input_enabled,
              disabled_reason: state.speech_disabled_reason,
              listening: false,
              vad_active: false,
              stt_backend: state.stt_backend,
              speaking_lock: false,
              last_heard: '',
              mic_mode: 'vad',
            },
            session_binding: state.session_binding,
            tts: {
              backend: state.tts_backend,
              voice: state.tts_voice,
              queue_depth: 0,
            },
          };
        });
      } catch {
        // WebSocket bootstrap remains primary source of truth.
      }
    };

    void bootstrapFromHttp();
  }, []);

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
              <div className="col-span-2">
                <span className="text-gray-400">Recognition:</span>
                <span className="ml-2 text-white">
                  {snapshot?.perception.recognition_state === 'no_face' && 'no face detected'}
                  {snapshot?.perception.recognition_state === 'unknown_user' && 'face detected, unknown user'}
                  {snapshot?.perception.recognition_state === 'below_threshold' && `below threshold (< ${(snapshot?.perception.face_match_threshold ?? 0).toFixed(2)})`}
                  {snapshot?.perception.recognition_state === 'recognized_enrolled' && 'recognized enrolled user'}
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
            sessionBinding={snapshot?.session_binding ?? 'anonymous'}
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
            speechDisabledReason={snapshot?.speech_input.disabled_reason ?? ''}
            isListening={snapshot?.speech_input.listening ?? false}
            onSubmitSuccess={handleTextSubmitSuccess}
            onMicToggled={handleMicToggled}
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
            refreshKey={memoryRefreshKey}
          />
        </div>
      </div>
    </div>
  );
}

export default App;

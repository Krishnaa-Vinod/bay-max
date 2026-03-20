import { useState, useEffect, useCallback, useRef } from 'react';
import { LiveWebSocket, ConnectionState } from '@/lib/ws';
import { getUIState } from '@/lib/api';
import type { SnapshotMessage, PipelineEvent, ActivityEvent, TextInputResponse } from '@/types/live';

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
    if (snapshot.perception.affect_confidence > 0.1) {
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

  const handleTextSubmitSuccess = useCallback((userText: string, response: TextInputResponse) => {
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

    addUiEvent({
      timestamp: new Date(),
      stage: 'dialogue_backend',
      status: response.fallback_used ? 'blocked' : 'complete',
      detail: response.fallback_used
        ? `LLM unavailable - using rule-based fallback (${response.backend})`
        : `Backend: ${response.backend}${response.model_name ? ` (${response.model_name})` : ''}`,
      type: 'pipeline_event',
    });

    if (response.tts_result) {
      addUiEvent({
        timestamp: new Date(),
        stage: 'tts',
        status: response.tts_error ? 'error' : 'complete',
        detail: response.tts_error
          ? `${response.tts_result}: ${response.tts_error}${response.tts_wav_path ? `; wav: ${response.tts_wav_path}` : ''}`
          : response.tts_result,
        type: 'speech_event',
      });
    }

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
        dialogue: {
          ...prev.dialogue,
          active_backend: response.backend || prev.dialogue.active_backend,
          active_model: response.model_name || prev.dialogue.active_model,
          fallback_warning: response.fallback_used
            ? 'LLM unavailable — using rule-based fallback'
            : '',
        },
        tools: {
          ...prev.tools,
          last_tools_used: response.tool_usage || [],
          last_web_sources: response.source_refs || [],
        },
        tts: {
          ...prev.tts,
          last_result: response.tts_result || prev.tts.last_result,
          last_error: response.tts_error || '',
          last_wav_path: response.tts_wav_path || prev.tts.last_wav_path,
        },
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
            mode: {
              voice_mode: state.voice_mode,
              requested: state.voice_mode_requested,
              fallback_reason: state.voice_fallback_reason,
              speech_loop_state: state.speech_loop_state,
            },
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
            dialogue: {
              active_backend: state.dialogue_backend,
              active_model: state.dialogue_model,
              requested_backend: state.dialogue_requested_backend,
              requested_model: state.dialogue_requested_model,
              fallback_warning: state.dialogue_fallback_warning,
            },
            memory_hits: [],
            tools: {
              web_tools_enabled: state.web_tools_enabled,
              web_tools_available: state.web_tools_available,
              web_tools_disabled_reason: state.web_tools_disabled_reason,
              last_tools_used: [],
              last_web_sources: [],
            },
            latency: {
              session_start_to_ready_ms: state.session_start_to_ready_ms,
              end_of_speech_to_first_audio_ms: state.end_of_speech_to_first_audio_ms,
              interrupt_to_audio_stop_ms: state.interrupt_to_audio_stop_ms,
            },
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
              voice_preset: state.tts_voice_preset,
              queue_depth: 0,
              last_result: state.tts_last_result,
              last_error: state.tts_last_error,
              last_wav_path: state.tts_last_wav_path,
              last_playback_ok: state.tts_last_playback_ok,
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
  const micStateLabel = !snapshot?.speech_input.enabled
    ? 'disabled'
    : snapshot.mode?.speech_loop_state || 'idle';

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
                  {snapshot?.perception.recognition_state === 'face_seen_unknown' && 'face detected, unknown user'}
                  {snapshot?.perception.recognition_state === 'known_low_confidence' && `known user, low confidence (< ${(snapshot?.perception.face_match_threshold ?? 0).toFixed(2)})`}
                  {snapshot?.perception.recognition_state === 'known_attached' && 'known user attached'}
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

          <div className="bg-baymax-bg rounded p-3 text-xs text-gray-300 space-y-1">
            <div className="flex justify-between gap-2">
              <span className="text-gray-400">Dialogue:</span>
              <span className="text-cyan-300 text-right">
                {snapshot?.dialogue.active_backend || 'unknown'}
                {snapshot?.dialogue.active_model ? ` / ${snapshot.dialogue.active_model}` : ''}
              </span>
            </div>
            {!!snapshot?.dialogue.fallback_warning && (
              <div className="text-amber-300 border border-amber-700/50 bg-amber-900/20 rounded px-2 py-1">
                {snapshot.dialogue.fallback_warning}
              </div>
            )}
            <div className="flex justify-between gap-2">
              <span className="text-gray-400">Mic state:</span>
              <span className="text-white">{micStateLabel}</span>
            </div>
            <div className="flex justify-between gap-2">
              <span className="text-gray-400">Voice mode:</span>
              <span className="text-white">{snapshot?.mode.voice_mode || 'unknown'}</span>
            </div>
            {!!snapshot?.mode.fallback_reason && (
              <div className="text-amber-300 border border-amber-700/50 bg-amber-900/20 rounded px-2 py-1">
                {snapshot.mode.fallback_reason}
              </div>
            )}
            {!snapshot?.speech_input.enabled && (
              <div className="text-red-300">Speech unavailable: {snapshot?.speech_input.disabled_reason || 'unavailable'}</div>
            )}
            <div className="flex justify-between gap-2">
              <span className="text-gray-400">Web tools:</span>
              <span className="text-white">{snapshot?.tools.web_tools_available ? 'available' : 'unavailable'}</span>
            </div>
            {!snapshot?.tools.web_tools_available && !!snapshot?.tools.web_tools_disabled_reason && (
              <div className="text-red-300">Web tools unavailable: {snapshot.tools.web_tools_disabled_reason}</div>
            )}
            {snapshot?.tools.last_tools_used?.length ? (
              <div className="text-gray-200">Last tools: {snapshot.tools.last_tools_used.join(', ')}</div>
            ) : null}
            <div className="flex justify-between gap-2">
              <span className="text-gray-400">TTS:</span>
              <span className="text-white">{snapshot?.tts.backend || 'none'} / {snapshot?.tts.voice || 'n/a'}</span>
            </div>
            <div className="text-gray-300">Latency (speech->audio): {(snapshot?.latency.end_of_speech_to_first_audio_ms ?? 0).toFixed(1)} ms</div>
            <div className="text-gray-300">Latency (interrupt): {(snapshot?.latency.interrupt_to_audio_stop_ms ?? 0).toFixed(1)} ms</div>
            <div className="flex justify-between gap-2">
              <span className="text-gray-400">Voice preset:</span>
              <span className="text-white">{snapshot?.tts.voice_preset || 'default'}</span>
            </div>
            {!!snapshot?.tts.last_result && (
              <div className="text-gray-200">TTS status: {snapshot.tts.last_result}</div>
            )}
            {!!snapshot?.tts.last_error && (
              <div className="text-red-300">Playback or synth error: {snapshot.tts.last_error}</div>
            )}
            {!!snapshot?.tts.last_wav_path && (
              <div className="text-blue-300 break-all">WAV: {snapshot.tts.last_wav_path}</div>
            )}
          </div>

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
            speakingLockActive={snapshot?.speech_input.speaking_lock ?? false}
            vadActive={snapshot?.speech_input.vad_active ?? false}
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
            recognitionState={snapshot?.perception.recognition_state ?? 'no_face'}
            sessionBinding={snapshot?.session_binding ?? 'anonymous'}
          />
        </div>
      </div>
    </div>
  );
}

export default App;

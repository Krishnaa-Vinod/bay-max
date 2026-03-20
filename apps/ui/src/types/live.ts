/**
 * TypeScript types for Bay-Max Live UI telemetry
 */

// Session state
export type SessionState = 'idle' | 'active' | 'paused' | 'ended';

// Pipeline state for avatar
export type PipelineState = 'OFFLINE' | 'IDLE' | 'ACTIVE' | 'LISTENING' | 'THINKING' | 'SPEAKING' | 'COOLDOWN';

// WebSocket message types
export type MessageType = 'bootstrap' | 'snapshot' | 'pipeline_event' | 'speech_event' | 'memory_event' | 'error';

// Session info in snapshot
export interface SessionInfo {
  id: string | null;
  state: SessionState;
  duration_sec: number;
  turn_count: number;
}

// Perception data in snapshot
export interface PerceptionData {
  face_detected: boolean;
  recognition_state: 'no_face' | 'face_seen_unknown' | 'known_low_confidence' | 'known_attached';
  face_match_threshold: number;
  user_name: string | null;
  user_id: string | null;
  recognition_confidence: number | null;
  posture: string | null;
  engagement: number | null;
  valence: number;
  arousal: number;
  affect_confidence: number;
  affect_backend: string;
  affect_stable_duration_sec: number;
  affect_debug: string;
}

// Speech input state
export interface SpeechInputState {
  enabled: boolean;
  disabled_reason: string;
  listening: boolean;
  vad_active: boolean;
  stt_backend: string;
  speaking_lock: boolean;
  last_heard: string;
  transcript_quality_score?: number;
  transcript_quality_reason?: string;
  mic_mode: string;
}

// TTS state
export interface TTSState {
  backend: string;
  voice: string;
  voice_preset: string;
  queue_depth: number;
  last_result: string;
  last_error: string;
  last_wav_path: string;
  last_playback_ok: boolean | null;
}

export interface DialogueState {
  active_backend: string;
  active_model: string;
  requested_backend: string;
  requested_model: string;
  fallback_warning: string;
}

// Memory hit
export interface MemoryHit {
  text: string;
  score: number | null;
  type?: string;
}

// Snapshot message from WebSocket
export interface SnapshotMessage {
  type: 'snapshot' | 'bootstrap';
  timestamp: string;
  mode: {
    voice_mode: 'realtime_voice' | 'local_chained_voice' | 'text_only';
    requested: string;
    fallback_reason: string;
    speech_loop_state: string;
    proactive_mode?: string;
  };
  session: SessionInfo;
  perception: PerceptionData;
  pipeline_state: PipelineState;
  last_response: string;
  last_spoken_text: string;
  dialogue: DialogueState;
  memory_hits: MemoryHit[];
  tools: {
    web_tools_enabled: boolean;
    web_tools_available: boolean;
    web_tools_disabled_reason: string;
    last_tools_used: string[];
    last_web_sources: Array<Record<string, unknown>>;
  };
  latency: {
    session_start_to_ready_ms: number;
    end_of_speech_to_first_audio_ms: number;
    interrupt_to_audio_stop_ms: number;
  };
  cooldown_remaining_sec: number;
  last_event: string;
  frame_count: number;
  analysis_count: number;
  uptime_sec: number;
  speech_input: SpeechInputState;
  session_binding: 'anonymous' | 'identified_user';
  tts: TTSState;
}

// Pipeline event message
export interface PipelineEvent {
  type: 'pipeline_event' | 'speech_event' | 'memory_event';
  timestamp: string;
  stage: string;
  status: 'pending' | 'in_progress' | 'complete' | 'blocked' | 'suppressed' | 'error';
  detail: string;
}

// Error message
export interface ErrorMessage {
  type: 'error';
  timestamp: string;
  error: string;
  detail?: string;
}

// Union type for all WebSocket messages
export type WebSocketMessage = SnapshotMessage | PipelineEvent | ErrorMessage;

// UI Bootstrap state from HTTP endpoint
export interface UIBootstrapState {
  live_mode_active: boolean;
  source: string;
  voice_mode: 'realtime_voice' | 'local_chained_voice' | 'text_only';
  voice_mode_requested: string;
  voice_fallback_reason: string;
  speech_loop_state: string;
  session_id: string | null;
  session_state: SessionState;
  user_id: string | null;
  user_name: string | null;
  dialogue_backend: string;
  dialogue_model: string;
  dialogue_requested_backend: string;
  dialogue_requested_model: string;
  dialogue_fallback_warning: string;
  tts_backend: string;
  tts_voice: string;
  tts_voice_preset: string;
  tts_last_result: string;
  tts_last_error: string;
  tts_last_wav_path: string;
  tts_last_playback_ok: boolean | null;
  stt_backend: string;
  affect_backend: string;
  speech_input_enabled: boolean;
  speech_disabled_reason: string;
  web_tools_enabled: boolean;
  web_tools_available: boolean;
  web_tools_disabled_reason: string;
  affect_enabled: boolean;
  tts_enabled: boolean;
  face_detected: boolean;
  recognition_state: 'no_face' | 'face_seen_unknown' | 'known_low_confidence' | 'known_attached';
  face_match_threshold: number;
  session_binding: 'anonymous' | 'identified_user';
  frame_count: number;
  analysis_count: number;
  uptime_sec: number;
  session_start_to_ready_ms: number;
  end_of_speech_to_first_audio_ms: number;
  interrupt_to_audio_stop_ms: number;
  last_response: string;
  last_event: string;
  server_time: string;
}

// Text input response
export interface TextInputResponse {
  success: boolean;
  session_id: string | null;
  response_text: string;
  memory_refs: MemoryHit[];
  strategy: string;
  backend: string;
  model_name: string;
  fallback_used: boolean;
  tts_result: string;
  tts_error: string;
  tts_wav_path: string;
  modality: string;
  tool_usage: string[];
  source_refs: Array<Record<string, unknown>>;
  error: string | null;
}

// Mic toggle response
export interface MicToggleResponse {
  success: boolean;
  listening: boolean;
  mic_mode: string;
  disabled_reason: string;
  speech_loop_state: string;
  error: string | null;
}

// Memory data from HTTP endpoint
export interface MemoryData {
  memories: Array<{
    text: string;
    type: string;
    created_at: string;
  }>;
  facts: Array<{
    key: string;
    value: string;
    confidence: number;
  }>;
  stats: {
    total_memories: number;
    total_facts: number;
    total_sessions: number;
  };
  empty_reason: 'no active user' | 'no retrieved memories yet' | 'memory service error' | '';
  recognition_reason?: string;
  memory_unavailable_hint?: string;
  error?: string | null;
}

// Activity feed event (derived from PipelineEvent for UI)
export interface ActivityEvent {
  id: string;
  timestamp: Date;
  stage: string;
  status: string;
  detail: string;
  type: string;
}

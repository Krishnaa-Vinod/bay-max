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
  recognition_state: 'no_face' | 'unknown_user' | 'below_threshold' | 'recognized_enrolled';
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
  mic_mode: string;
}

// TTS state
export interface TTSState {
  backend: string;
  voice: string;
  queue_depth: number;
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
  session: SessionInfo;
  perception: PerceptionData;
  pipeline_state: PipelineState;
  last_response: string;
  last_spoken_text: string;
  memory_hits: MemoryHit[];
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
  session_id: string | null;
  session_state: SessionState;
  user_id: string | null;
  user_name: string | null;
  dialogue_backend: string;
  dialogue_model: string;
  tts_backend: string;
  tts_voice: string;
  stt_backend: string;
  affect_backend: string;
  speech_input_enabled: boolean;
  speech_disabled_reason: string;
  affect_enabled: boolean;
  tts_enabled: boolean;
  face_detected: boolean;
  recognition_state: 'no_face' | 'unknown_user' | 'below_threshold' | 'recognized_enrolled';
  face_match_threshold: number;
  session_binding: 'anonymous' | 'identified_user';
  frame_count: number;
  analysis_count: number;
  uptime_sec: number;
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
  error: string | null;
}

// Mic toggle response
export interface MicToggleResponse {
  success: boolean;
  listening: boolean;
  mic_mode: string;
  disabled_reason: string;
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

"""Bay-Max configuration settings from environment variables."""

import os

from pydantic_settings import BaseSettings


class BaymaxSettings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = {"env_prefix": "BAYMAX_", "env_file": ".env", "extra": "ignore"}

    # Data directories
    data_dir: str = os.path.expandvars("/scratch/$USER/bay-max/data")
    cache_dir: str = os.path.expandvars("/scratch/$USER/bay-max/cache")
    model_dir: str = os.path.expandvars("/scratch/$USER/bay-max/models")

    # Database
    db_url: str = "sqlite+aiosqlite:///./baymax.db"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Logging
    log_level: str = "INFO"

    # Vision / perception
    device: str = "cuda_if_available_else_cpu"
    face_backend: str = "facenet_pytorch"
    face_match_threshold: float = 0.75
    max_faces_per_frame: int = 5

    # Pose estimation
    pose_backend: str = "mediapipe"
    pose_min_confidence: float = 0.5

    # Annotation and debug artifacts
    enable_annotations: bool = True
    artifact_dir: str = "./artifacts"

    # Vector store
    vector_backend: str = "lancedb"

    # Text embedding (for memory retrieval)
    text_embedding_model: str = "all-MiniLM-L6-v2"
    text_embedding_dim: int = 384

    # Memory retrieval
    memory_top_k: int = 5
    memory_similarity_threshold: float = 0.4

    # Session smoothing
    session_smoothing_window: int = 5

    # Consolidation
    enable_memory_consolidation: bool = True

    # --- Iteration 005: Dialogue backend settings ---

    # Backend selector: "rule_based" | "ollama" | "transformers"
    dialogue_backend: str = "rule_based"

    # Enable structured debug fields in responses
    enable_dialogue_debug: bool = False

    # Safe-health mode: refuse to make diagnostic claims
    enable_safe_health_mode: bool = True

    # Max recent conversation turns to include in grounded prompt
    dialogue_max_history_turns: int = 8

    # Max memories fed into the dialogue prompt
    dialogue_top_k_memories: int = 5

    # Generation temperature (applies to Ollama and Transformers backends)
    dialogue_temperature: float = 0.4

    # Max new tokens for the LLM response
    dialogue_max_new_tokens: int = 220

    # Ollama backend settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = ""  # e.g. "qwen2.5:1.5b"

    # HuggingFace Transformers backend settings
    hf_chat_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    hf_dtype: str = "auto"  # "auto" | "float16" | "bfloat16" | "float32"

    # Enable rule-based fallback when the selected backend fails
    enable_rule_based_fallback: bool = True

    # --- Iteration 006: Live webcam continuity settings ---

    # Source type: "webcam" | "replay"
    live_source: str = "webcam"
    live_camera_index: int = 0

    # Frame rate / analysis cadence
    preview_fps: int = 8
    analysis_interval_sec: float = 2.0

    # Presence detection thresholds
    presence_min_consecutive_frames: int = 2
    absence_timeout_sec: float = 30.0
    session_resume_window_sec: float = 300.0

    # Response cooldowns
    proactive_min_interval_sec: float = 30.0
    any_response_min_interval_sec: float = 10.0
    quiet_companionship_interval_sec: float = 300.0
    event_min_stability_sec: float = 4.0

    # Live overlay and artifacts
    enable_live_overlay: bool = True
    enable_live_debug_hud: bool = True
    enable_artifact_logging: bool = True
    live_artifact_dir: str = "./artifacts/live_run_007"
    live_max_run_sec: float = 0.0  # 0 = unlimited
    live_video_replay_path: str = ""

    # --- Iteration 007: TTS / Spoken Output settings ---

    # TTS backend: "null" | "kokoro" | "piper"
    tts_backend: str = "kokoro"
    tts_voice: str = "af_heart"
    tts_rate: int = 24000
    tts_enabled: bool = True
    tts_device: str = "auto"
    tts_output_dir: str = "./artifacts/tts"

    # Audio playback
    enable_audio_playback: bool = True
    audio_backend: str = "sounddevice"

    # Kokoro-specific
    kokoro_voice: str = "af_heart"

    # Piper-specific
    piper_model_path: str = ""
    piper_config_path: str = ""

    # --- Iteration 008: Speech Input / Bidirectional Speech ---

    # Master enable for speech input
    enable_speech_input: bool = True

    # Microphone settings
    mic_backend: str = "sounddevice"
    mic_sample_rate: int = 16000
    mic_channels: int = 1
    mic_device: str = "default"

    # VAD settings
    vad_backend: str = "silero"
    vad_threshold: float = 0.65
    vad_min_speech_ms: float = 300.0
    vad_silence_ms: float = 500.0

    # STT settings
    stt_backend: str = "faster_whisper"
    whisper_model: str = "base.en"
    whisper_device: str = "auto"

    # Echo suppression
    echo_suppression_enabled: bool = True
    echo_similarity_threshold: float = 0.85

    # Microphone mode: "vad" or "push_to_talk"
    mic_mode: str = "vad"

    # Post-speech cooldown before accepting new mic input (ms)
    post_speech_cooldown_ms: float = 1500.0

    # Listening indicator
    enable_listening_indicator: bool = True

    # Audio artifact directory
    audio_artifact_dir: str = "./artifacts/audio"

    # --- Iteration 009: Affect Analysis / Facial Emotion Recognition ---

    # Master enable for affect analysis
    enable_affect: bool = True

    # Affect backend: "null" | "mediapipe"
    # Note: py-feat attempted but incompatible with Python 3.11 environment
    affect_backend: str = "mediapipe"

    # Device for affect analysis
    affect_device: str = "auto"

    # Sample every N frames for affect analysis (to control computational load)
    affect_sample_every_n_frames: int = 10

    # Minimum confidence to accept an affect analysis result
    affect_confidence_threshold: float = 0.60

    # Smoothing parameter for EMA (0=no smoothing, 1=keep previous)
    affect_smoothing_alpha: float = 0.25

    # Duration (seconds) state must be stable before considering for memory
    affect_stability_duration_sec: float = 30.0

    # Minimum confidence to write affect observations to memory
    affect_memory_confidence_threshold: float = 0.65

    # Enable affect-based planner bias
    affect_bias_enabled: bool = True

    # Minimum confidence to apply affect bias to response strategy
    affect_bias_confidence_threshold: float = 0.60

    # Valence thresholds for strategy adjustment
    affect_negative_valence_threshold: float = -0.30
    affect_positive_valence_threshold: float = 0.40

    # Arousal thresholds for strategy adjustment
    affect_low_arousal_threshold: float = 0.40
    affect_high_arousal_threshold: float = 0.55

    # Enable debug output for affect analysis
    affect_debug: bool = False

    # Directory for affect artifacts and timelines
    affect_artifact_dir: str = "./artifacts/affect"


def get_settings() -> BaymaxSettings:
    """Return a settings instance."""
    return BaymaxSettings()

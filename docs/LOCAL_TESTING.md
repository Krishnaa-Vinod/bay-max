# Local Testing Guide

This document explains how to test Bay-Max locally on a headless machine (e.g., HPC cluster nodes) without a display.

## Prerequisites

Install all dependencies including vision extras:

```bash
pip install -e ".[all]"
```

This installs `torch`, `facenet-pytorch`, `opencv-python`, and `mediapipe` alongside dev dependencies.

## Recommended Local Workflow (Full Experience)

Use a single backend process so API and live runtime share state in one Python process.

```bash
# Terminal 1
make run-local-backend-full

# Terminal 2
make run-ui
```

Open `http://localhost:3000`.

Alternative one-command workflow:

```bash
make run-local-full
```

Explicit local backend modes:

```bash
# basic: no speech
make run-local-backend-basic

# voice: TTS only
make run-local-backend-voice

# full: TTS + speech input
make run-local-backend-full
```

Full mode dialogue preference order:
1. Ollama (if reachable)
2. Transformers (if explicitly configured)
3. Rule-based fallback

If fallback is active, the UI shows a warning badge:
LLM unavailable - using rule-based fallback

Advanced/debug split-process workflow (not recommended for daily local usage):

```bash
make run-api
make live-webcam
make run-ui
```

## Local Stabilization Smoke Checklist

1. UI opens without bootstrap errors and telemetry transitions to connected after snapshot/bootstrap.
2. Recognition panel distinguishes: no face, unknown user, below threshold, recognized enrolled user.
3. Text submit shows inline success or inline backend error (no silent console-only failure).
4. Successful text submit appends user + assistant activity entries and updates last response immediately.
5. Memory panel refreshes immediately after text response and user recognition changes.
6. Memory panel empty state explains one of: `no active user`, `no retrieved memories yet`, `memory service error`.
7. Speech disabled reason is shown exactly (for example dependency/startup/backend disabled reasons).
8. When speech is enabled, mic toggle shows listening started/stopped and emits activity feed event.
9. Reconnect clears stale websocket error rows and adds one recovery event.

## Environment Variables

Copy `.env.example` and configure:

```bash
cp .env.example .env
```

Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `BAYMAX_DEVICE` | `cuda_if_available_else_cpu` | Compute device for models |
| `BAYMAX_POSE_BACKEND` | `mediapipe` | Pose estimation backend (`mediapipe` or `stub`) |
| `BAYMAX_POSE_MIN_CONFIDENCE` | `0.5` | Minimum pose detection confidence |
| `BAYMAX_ENABLE_ANNOTATIONS` | `true` | Save annotated debug frames |
| `BAYMAX_ARTIFACT_DIR` | `./artifacts` | Directory for annotated outputs |
| `BAYMAX_FACE_MATCH_THRESHOLD` | `0.75` | Cosine similarity threshold for face matching |

## CLI Test Script

The `scripts/local_test.py` script provides a command-line interface for local testing.

### Enroll a face

```bash
python scripts/local_test.py --enroll path/to/face.jpg --name "Alice"
```

Output: JSON with user_id, enrollment status, and message.

### Analyze a single image

```bash
python scripts/local_test.py --image path/to/image.jpg
```

Output: JSON with face detection/recognition results, pose, engagement, and path to annotated artifact.

### Analyze a folder of images (frame replay)

```bash
python scripts/local_test.py --folder path/to/frames/
```

Processes all `.jpg`, `.jpeg`, `.png`, `.bmp` files in sorted order, simulating a video replay.

### Full workflow (enroll then analyze)

```bash
python scripts/local_test.py --enroll face.jpg --name "Alice" --image test.jpg
```

## Annotated Outputs

When `BAYMAX_ENABLE_ANNOTATIONS` is true, the test script saves annotated images to `BAYMAX_ARTIFACT_DIR` with:

- **Green rectangles**: Recognized faces with name and confidence
- **Red rectangles**: Unknown/unrecognized faces
- **Yellow circles and lines**: Pose landmarks and skeleton connections
- **Text overlay**: Engagement level, posture, lean, and motion labels

Files are named: `frame_{session_id_prefix}_{timestamp}.jpg`

## Running Tests

```bash
# Run all tests
make test

# Run only iteration 005 tests
pytest tests/test_iteration_005.py -v

# Run with lint
make lint
```

## Gradio Demo

```bash
make run-demo
```

Opens on `http://0.0.0.0:7860`. The Interact tab now shows backend, model_name, fallback_used, safety_flags, and memory_refs. The Backends tab shows active backend configuration.

## Smoke Test Checklist (Iteration 005)

1. `pip install -e ".[all]"` completes without errors
2. `make lint` passes (0 errors)
3. `make test` passes (194 tests)
4. `curl http://localhost:8000/v1/dialogue/backends` returns JSON with `available_backends`, `active_backend`
5. `POST /v1/respond` returns `backend`, `model_name`, `fallback_used`, `safety_flags` fields
6. Safety test: context `"Can you diagnose me from my face?"` returns non-empty `safety_flags`
7. (Ollama) Two-session recall returns non-empty `memory_refs` after consolidation

---

## Dialogue Backend Testing (Iteration 005)

### Rule-Based Mode (no model required)

```bash
# Default — rule_based requires nothing extra
BAYMAX_DIALOGUE_BACKEND=rule_based make run-api
```

### Ollama Mode

```bash
# Install Ollama and pull model
ollama pull qwen2.5:1.5b

# Run Bay-Max with Ollama
BAYMAX_DIALOGUE_BACKEND=ollama BAYMAX_OLLAMA_MODEL=qwen2.5:1.5b make run-api
```

### Transformers Mode

```bash
pip install -e ".[dialogue]"
BAYMAX_DIALOGUE_BACKEND=transformers \
  BAYMAX_HF_CHAT_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
  make run-api
```

### Smoke Test Script

```bash
# Rule-based (no model needed)
python scripts/dialogue_smoke.py

# Ollama
BAYMAX_DIALOGUE_BACKEND=ollama BAYMAX_OLLAMA_MODEL=qwen2.5:1.5b \
  python scripts/dialogue_smoke.py

# Transformers
BAYMAX_DIALOGUE_BACKEND=transformers \
  BAYMAX_HF_CHAT_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
  python scripts/dialogue_smoke.py
```

Outputs: `./artifacts/smoke_test_005.json`

## Live Mode Testing (Iteration 006)

### Environment Variables for Live Mode

| Variable | Default | Description |
|----------|---------|-------------|
| `BAYMAX_LIVE_SOURCE` | `webcam` | Frame source: `webcam` or `replay` |
| `BAYMAX_LIVE_CAMERA_INDEX` | `2` | Webcam device index (host-dependent; adjust if needed) |
| `BAYMAX_PREVIEW_FPS` | `8` | Frame read rate (preview loop) |
| `BAYMAX_ANALYSIS_INTERVAL_SEC` | `2.0` | Seconds between full analyses |
| `BAYMAX_PRESENCE_MIN_CONSECUTIVE_FRAMES` | `2` | Frames needed to confirm presence |
| `BAYMAX_ABSENCE_TIMEOUT_SEC` | `30.0` | Seconds before session pauses on absence |
| `BAYMAX_SESSION_RESUME_WINDOW_SEC` | `300.0` | Seconds within which same user resumes |
| `BAYMAX_PROACTIVE_MIN_INTERVAL_SEC` | `30.0` | Per-event-type cooldown |
| `BAYMAX_ANY_RESPONSE_MIN_INTERVAL_SEC` | `10.0` | Global any-response cooldown |
| `BAYMAX_QUIET_COMPANIONSHIP_INTERVAL_SEC` | `300.0` | Quiet presence interval before prompt |
| `BAYMAX_EVENT_MIN_STABILITY_SEC` | `4.0` | Stability window for posture/engagement events |
| `BAYMAX_ENABLE_LIVE_OVERLAY` | `true` | Draw HUD on frame (disable for headless) |
| `BAYMAX_ENABLE_ARTIFACT_LOGGING` | `true` | Write JSONL event/response logs |
| `BAYMAX_LIVE_ARTIFACT_DIR` | `./artifacts/live_run_006` | Artifact output directory |
| `BAYMAX_LIVE_MAX_RUN_SEC` | `0` | Max run duration (0 = unlimited) |
| `BAYMAX_LIVE_VIDEO_REPLAY_PATH` | `` | Path to video file or frame folder for replay |

### Webcam Live Mode

```bash
make live-webcam
# or
python scripts/live_companion.py
```

This starts the continuous webcam loop. Press Ctrl-C to stop cleanly. Artifacts are written to `./artifacts/live_run_006/`.

### Replay Mode (Headless / No Webcam)

```bash
# Replay from a folder of images (sorted alphabetically)
REPLAY_PATH=/path/to/frames make live-replay
# or
python scripts/live_companion.py --replay /path/to/frames --no-overlay

# Replay from a video file
python scripts/live_companion.py --replay /path/to/video.mp4 --no-overlay
```

### Limiting Run Duration

```bash
# Stop after 60 seconds
python scripts/live_companion.py --max-run-sec 60

# Stop after 100 frames
python scripts/live_companion.py --max-frames 100 --no-overlay
```

### Inspecting Live State via API

While the live runner is active, query the API:

```bash
# Start API in background first:
make run-api &

# Then in another terminal:
curl http://localhost:8000/v1/live/status
```

### Artifact Inspection

After any live or replay run, inspect artifacts in `./artifacts/live_run_006/`:

```
events.jsonl          - All companion events (arrivals, departures, etc.)
responses.jsonl       - All proactive responses with trigger_reason and memory_refs
session_timeline.json - Session lifecycle transitions
manifest.json         - Run summary metadata
summary.md            - Human-readable run summary
snapshots/            - Annotated JPEG frames at key events (if camera available)
```

### Smoke Test Checklist (Iteration 006)

1. `make test` passes (256 tests)
2. `make lint` passes (0 errors)
3. `python scripts/live_companion.py --replay path/to/frames --max-frames 50 --no-overlay` completes
4. `artifacts/live_run_006/events.jsonl` exists and is readable after replay
5. `artifacts/live_run_006/manifest.json` has valid JSON
6. `curl http://localhost:8000/v1/live/status` returns `{"live_mode_active": false}` (no active runtime)
7. `POST /v1/live/control {"action": "stop"}` returns `{"status": "ok", "message": "Live mode not active"}`

---

## TTS / Spoken Output Testing (Iteration 007)

### Environment Variables for TTS

| Variable | Default | Description |
|----------|---------|-------------|
| `BAYMAX_TTS_BACKEND` | `kokoro` | TTS backend: `kokoro`, `piper`, or `null` |
| `BAYMAX_TTS_VOICE` | `am_michael` | Voice ID for TTS synthesis |
| `BAYMAX_TTS_SAMPLE_RATE` | `24000` | Audio sample rate in Hz |
| `BAYMAX_TTS_OUTPUT_DIR` | `./artifacts/tts_output` | Directory for generated WAV files |
| `BAYMAX_TTS_ENABLE_PLAYBACK` | `true` | Play audio through speakers |
| `BAYMAX_TTS_QUEUE_MAX_SIZE` | `10` | Maximum queued utterances |
| `BAYMAX_TTS_SPEED` | `1.0` | Speech speed multiplier |
| `BAYMAX_ENABLE_TTS` | `true` | Enable/disable TTS globally |

### Installing TTS Dependencies

```bash
pip install -e ".[tts]"
```

This installs `kokoro`, `sounddevice`, and `soundfile` for speech synthesis and audio playback.

### Testing TTS Standalone

```bash
# Test TTS synthesis only (no webcam)
python scripts/verify_007.py --tts-only
```

### Testing Live Mode with TTS

```bash
# Webcam + TTS (spoken proactive responses)
make live-webcam-tts
# or
python scripts/live_companion.py --tts

# Replay mode with TTS
python scripts/live_companion.py --replay /path/to/frames --tts --no-overlay
```

### Makefile Targets

```bash
make live-webcam-tts   # Webcam + TTS enabled
make verify-007        # Run iteration 007 verification script
```

### Smoke Test Checklist (Iteration 007)

1. `make test` passes (296 tests)
2. `make lint` passes (0 errors)
3. `pip install -e ".[tts]"` completes without errors
4. `python scripts/verify_007.py --tts-only` generates a WAV file
5. `curl http://localhost:8000/v1/tts/backends` returns JSON with `available_backends`, `active_backend`
6. `BAYMAX_TTS_BACKEND=null make run-api` starts without TTS errors
7. `python scripts/verify_007.py` runs full webcam + TTS verification

### TTS Troubleshooting

- **PortAudio not found**: Install PortAudio system package (`apt install portaudio19-dev` on Debian/Ubuntu, `yum install portaudio-devel` on RHEL/CentOS)
- **No audio output on headless machine**: Set `BAYMAX_TTS_ENABLE_PLAYBACK=false` to skip playback; WAV files are still generated
- **Kokoro import error**: Ensure `pip install -e ".[tts]"` was run; Kokoro requires Python 3.11+
- **Piper not available**: Piper backend is a placeholder; use `kokoro` or `null` instead

---

## Speech Input Testing (Iteration 008)

### Environment Variables for Speech Input

| Variable | Default | Description |
|----------|---------|-------------|
| `BAYMAX_ENABLE_SPEECH_INPUT` | `true` | Enable/disable speech input globally |
| `BAYMAX_MIC_BACKEND` | `sounddevice` | Microphone backend |
| `BAYMAX_MIC_SAMPLE_RATE` | `48000` | Audio sample rate in Hz |
| `BAYMAX_MIC_CHANNELS` | `1` | Audio channels (mono) |
| `BAYMAX_MIC_DEVICE` | `default` | Microphone device ID |
| `BAYMAX_VAD_BACKEND` | `silero` | VAD backend: `silero` or `null` |
| `BAYMAX_VAD_THRESHOLD` | `0.65` | Speech detection threshold (0.0-1.0) |
| `BAYMAX_VAD_MIN_SPEECH_MS` | `300` | Min speech duration in ms |
| `BAYMAX_VAD_SILENCE_MS` | `500` | Silence timeout to end segment in ms |
| `BAYMAX_STT_BACKEND` | `faster_whisper` | ASR backend: `faster_whisper` or `null` |
| `BAYMAX_WHISPER_MODEL` | `base.en` | Whisper model name |
| `BAYMAX_WHISPER_DEVICE` | `auto` | Compute device for ASR |
| `BAYMAX_ECHO_SUPPRESSION_ENABLED` | `true` | Enable echo suppression |
| `BAYMAX_ECHO_SIMILARITY_THRESHOLD` | `0.85` | Echo text similarity threshold |
| `BAYMAX_MIC_MODE` | `vad` | Mic mode: `vad` or `push_to_talk` |
| `BAYMAX_POST_SPEECH_COOLDOWN_MS` | `1500` | Cooldown after TTS finishes in ms |
| `BAYMAX_ENABLE_LISTENING_INDICATOR` | `true` | Show listening state indicator |
| `BAYMAX_AUDIO_ARTIFACT_DIR` | `./artifacts/audio` | Directory for audio artifacts |

### Installing Speech Dependencies

```bash
pip install -e ".[speech]"

# Or install everything
pip install -e ".[all]"
```

This installs `faster-whisper`, `sounddevice`, and `soundfile` for speech input.

### Testing Full Bidirectional Speech

```bash
# Webcam + TTS + Speech Input (full bidirectional)
make live-webcam-speech
# or
python scripts/live_companion.py --stt-backend faster_whisper --tts-backend kokoro

# Push-to-talk mode (manual trigger instead of continuous VAD)
python scripts/live_companion.py --stt-backend faster_whisper --mic-mode push_to_talk

# Speech input without TTS
python scripts/live_companion.py --stt-backend faster_whisper --no-tts

# Webcam only (no speech input)
python scripts/live_companion.py --no-speech-input
```

### Checking Speech Status via API

```bash
# Check speech input/output status
curl http://localhost:8000/v1/audio/status

# Check STT backends
curl http://localhost:8000/v1/stt/backends
```

### Smoke Test Checklist (Iteration 008)

1. `make test` passes (356 tests)
2. `make lint` passes (0 errors)
3. `pip install -e ".[speech]"` completes without errors
4. `curl http://localhost:8000/v1/audio/status` returns JSON with `speech_input_enabled`, `listening`, `stt_backend`
5. `curl http://localhost:8000/v1/stt/backends` returns JSON with `available_backends`, `active_backend`
6. `python scripts/live_companion.py --stt-backend faster_whisper` starts without crash (requires microphone)
7. Speaking a phrase produces a transcription and a spoken response (requires mic + speakers)

### Speech Input Troubleshooting

- **PortAudio not found**: Install PortAudio system package (`apt install portaudio19-dev` on Debian/Ubuntu)
- **No microphone on headless machine**: Set `BAYMAX_ENABLE_SPEECH_INPUT=false` or use `--no-speech-input` flag
- **Silero VAD model download fails**: Check internet connectivity; model is cached after first download via `torch.hub`
- **faster-whisper model download fails**: Check internet connectivity; model is cached in HuggingFace cache directory
- **Echo loop (Bay-Max responds to itself)**: Increase `BAYMAX_ECHO_SIMILARITY_THRESHOLD` or `BAYMAX_POST_SPEECH_COOLDOWN_MS`
- **False VAD triggers**: Increase `BAYMAX_VAD_THRESHOLD` (e.g., 0.75) and `BAYMAX_VAD_MIN_SPEECH_MS` (e.g., 500)

---

## Troubleshooting

- **MediaPipe import error**: Ensure `mediapipe>=0.10.0` is installed. On some HPC systems, you may need to install from a wheel.
- **No CUDA**: Set `BAYMAX_DEVICE=cpu` in `.env`
- **No display (cv2 error)**: The pipeline uses headless OpenCV (`cv2.imwrite`) and does not require a display server.
- **Stub fallback**: If MediaPipe fails to load, the system silently falls back to `StubPoseEstimator` (no pose detected). Check logs for warnings.

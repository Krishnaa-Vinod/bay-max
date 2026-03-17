# Live Verification Guide

## Overview

This guide explains how to verify the Bay-Max live webcam + TTS pipeline on a local machine. It covers the full end-to-end flow: webcam capture, face detection, user recognition, session management, proactive response generation, text-to-speech synthesis, and audio playback. The verification process produces a set of artifacts that prove each stage of the pipeline executed correctly.

## Prerequisites

- **Webcam**: A USB or built-in webcam accessible as a video device (e.g., `/dev/video0` on Linux).
- **Speakers** (optional): For hearing TTS audio output. If no speakers or PortAudio are available, WAV files are still saved.
- **Python 3.11+** installed.
- **Install dependencies**:

```bash
pip install -e ".[all,tts]"
```

This installs the core library, all optional backends (including vision and TTS), and the Kokoro TTS model.

## Quick Start

Three commands to verify the pipeline:

```bash
# 1. TTS-only smoke test (no webcam needed)
python scripts/verify_007.py --tts-only

# 2. Full webcam + TTS verification (30-second session)
python scripts/verify_007.py --duration 30

# 3. Inspect the generated artifacts
python scripts/verify_007.py --inspect-latest
```

## Verification Script

The verification script is located at `scripts/verify_007.py`. It supports the following flags:

| Flag                  | Description                                                      |
|-----------------------|------------------------------------------------------------------|
| `--tts-only`          | Run only the TTS smoke test. Does not require a webcam.          |
| `--duration SECONDS`  | Duration of the live webcam session in seconds. Default: 30.     |
| `--no-audio`          | Disable audio playback (WAV files are still saved).              |
| `--backend BACKEND`   | TTS backend to use (`null`, `kokoro`, `piper`). Default: `kokoro`. |
| `--voice VOICE`       | TTS voice identifier. Default: `af_heart`.                       |
| `--artifact-dir DIR`  | Directory for output artifacts. Default: `artifacts/verify_007/`. |
| `--inspect-latest`    | Print a summary of the most recent verification run artifacts.   |
| `--camera-index N`    | Camera device index. Default: `0`.                               |
| `--verbose`           | Enable debug-level logging.                                      |

## TTS Smoke Test

The `--tts-only` flag runs a focused test of the TTS subsystem without requiring a webcam.

**What it tests:**

1. Initializes the configured TTS backend (default: Kokoro).
2. Synthesizes a short test phrase ("Hello, I am Bay-Max. I am here to help.").
3. Saves the output as a WAV file.
4. Plays the audio through the default output device (if `--no-audio` is not set).
5. Logs a `SpeechEvent` artifact.

**Expected output:**

```
[TTS Smoke Test]
  Backend:    kokoro
  Voice:      af_heart
  Available:  yes
  Synthesize: ok (1.2s)
  WAV saved:  artifacts/verify_007/tts_smoke/speech_001.wav
  Playback:   ok
  Result:     PASS
```

**Artifacts produced:**

- `artifacts/verify_007/tts_smoke/speech_001.wav` -- the synthesized audio file.
- `artifacts/verify_007/tts_smoke/smoke_result.json` -- structured test result (`TTSSmokeTestResult`).

## Full Webcam Run

Running `python scripts/verify_007.py --duration 30` starts a full live verification session.

**What happens:**

1. The webcam starts capturing frames.
2. Face detection runs on each frame to identify visible faces.
3. Recognized users are matched against known profiles; unknown faces trigger new user creation.
4. A session is created or resumed for each recognized user.
5. The orchestrator generates proactive responses based on session context (greeting, check-in, farewell).
6. Each response is synthesized to speech via the active TTS backend and played aloud.
7. All events, responses, and speech artifacts are logged to the artifact directory.
8. After the configured duration, the session ends gracefully and a summary is written.

**Expected session flow:**

```
t=0s   Webcam opened, capturing frames
t=1s   Face detected, user recognized (or new user created)
t=2s   Session started, greeting response generated
t=2s   TTS: "Hello! Welcome to Bay-Max."
t=15s  Mid-session check-in response generated
t=15s  TTS: "How are you feeling today?"
t=30s  Session ending, farewell response generated
t=30s  TTS: "Take care! It was nice seeing you."
t=30s  Artifacts written, summary generated
```

## Artifact Inspection

After a verification run, the artifact directory contains a complete record of the session. Use `--inspect-latest` to print a summary, or examine the files directly.

**Artifact directory structure:**

```
artifacts/verify_007/
  manifest.json          # Run metadata: timestamp, duration, flags, git SHA
  summary.md             # Human-readable summary of the verification run
  events.jsonl           # All perception events (face detected, user recognized, etc.)
  responses.jsonl        # All generated SupportiveResponse objects
  speech.jsonl           # All TTS synthesis attempts and results
  speech_001.wav         # First synthesized audio file
  speech_002.wav         # Second synthesized audio file
  ...
  tts_smoke/             # TTS smoke test artifacts (if --tts-only was run)
    speech_001.wav
    smoke_result.json
```

**File descriptions:**

| File               | Format  | Contents                                                        |
|--------------------|---------|-----------------------------------------------------------------|
| `manifest.json`    | JSON    | Run metadata: timestamp, duration, configuration, git commit SHA. |
| `summary.md`       | Markdown| Human-readable summary with pass/fail status for each stage.    |
| `events.jsonl`     | JSONL   | One JSON object per line for each perception event.             |
| `responses.jsonl`  | JSONL   | One JSON object per line for each generated response.           |
| `speech.jsonl`     | JSONL   | One JSON object per line for each TTS synthesis attempt.        |
| `speech_NNN.wav`   | WAV     | Synthesized audio files, numbered sequentially.                 |

## Manual Test Cases

| ID    | Case Name                              | Expected Result                                                                 |
|-------|----------------------------------------|---------------------------------------------------------------------------------|
| VT701 | TTS smoke test with Kokoro             | WAV file produced, playback succeeds, `smoke_result.json` shows `success: true`. |
| VT702 | TTS smoke test with null backend       | Empty WAV file produced, no playback, `smoke_result.json` shows `success: true`. |
| VT703 | TTS fallback when Kokoro unavailable   | System falls back to null backend, logs a warning, run completes without error.  |
| VT704 | Webcam opens and captures frames       | `events.jsonl` contains `frame_captured` events within the first 2 seconds.      |
| VT705 | Face detection on live frames          | `events.jsonl` contains `face_detected` events with bounding box coordinates.    |
| VT706 | User recognition and session start     | `events.jsonl` contains `user_recognized` and `session_started` events.          |
| VT707 | Greeting response generated            | `responses.jsonl` contains a greeting response within 5 seconds of session start.|
| VT708 | Greeting synthesized to speech         | `speech.jsonl` contains an entry with `trigger_event: proactive_response` and `success: true`. Corresponding WAV file exists. |
| VT709 | Session ends after duration            | `events.jsonl` contains `session_ended` event. `summary.md` reports session duration. |
| VT710 | Artifacts complete after run           | `manifest.json`, `summary.md`, `events.jsonl`, `responses.jsonl`, and `speech.jsonl` all exist and are non-empty. |

## Troubleshooting

### No webcam available

**Symptom**: Error message about failing to open video device, or `cv2.VideoCapture` returns empty frames.

**Solution**: Use `--tts-only` to verify the TTS pipeline without a webcam. For full pipeline testing, connect a USB webcam or configure a virtual camera. On Linux, check that `/dev/video0` exists and has appropriate permissions (`sudo chmod 666 /dev/video0`). You can also try a different device index with `--camera-index 1`.

### No speakers / no audio output

**Symptom**: WAV files are saved but no audio is heard.

**Solution**: This is expected on headless machines. The pipeline is designed to degrade gracefully without audio output. WAV files are still saved to the artifact directory for offline review. If you have speakers but hear nothing, check your system audio settings and default output device.

### No PortAudio library

**Symptom**: Warning message like `sounddevice: PortAudio library not found` at startup.

**Solution**: Install PortAudio for your platform:

```bash
# Ubuntu / Debian
sudo apt-get install libportaudio2

# macOS (Homebrew)
brew install portaudio

# RHEL / CentOS
sudo yum install portaudio
```

After installing PortAudio, reinstall sounddevice: `pip install sounddevice --force-reinstall`. If PortAudio cannot be installed (e.g., in a container), the system continues without playback and WAV files are still saved.

### Kokoro TTS initialization failure

**Symptom**: Error during Kokoro model loading, or the backend reports `is_available: false`.

**Solution**: Ensure the Kokoro package is installed: `pip install kokoro`. Verify that model weights are accessible (they are downloaded automatically on first use). Check available disk space (the model is approximately 200 MB). If Kokoro fails to load, the system automatically falls back to the `null` backend. You can explicitly select it with `--backend null` to bypass Kokoro entirely.

## Makefile Targets

The following Makefile targets are available for iteration 007 verification:

```bash
# Run the full iteration 007 verification suite (TTS smoke + webcam)
make verify-007

# Run live webcam with TTS enabled (default 60-second session)
make live-webcam-tts

# Run live webcam without TTS (vision pipeline only)
make live-webcam
```

| Target             | Description                                                        |
|--------------------|--------------------------------------------------------------------|
| `verify-007`       | Runs the complete verification suite: TTS smoke test followed by a 30-second live webcam session. Produces artifacts in `artifacts/verify_007/`. |
| `live-webcam-tts`  | Starts a live webcam session with TTS enabled. Default duration is 60 seconds. Useful for interactive testing and demos. |
| `live-webcam`      | Starts a live webcam session without TTS. Tests the vision and session pipeline only. Useful when audio dependencies are not installed. |

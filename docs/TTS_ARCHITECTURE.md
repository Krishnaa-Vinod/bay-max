# TTS Architecture

## Overview

The TTS (Text-to-Speech) output layer for Bay-Max converts `SupportiveResponse` text into spoken audio. It is designed as a **separate output layer** that runs after response generation, not inside the planner or dialogue modules. This separation keeps the response planning logic independent of audio concerns and allows TTS to be enabled, disabled, or swapped without affecting the core conversation pipeline.

## Architecture

The TTS subsystem follows a layered architecture:

```
TTSProvider (interface)
    |
    v
Concrete Providers (NullTTSProvider, KokoroTTSProvider, PiperTTSProvider)
    |
    v
SpeechService (queue management, playback, artifact tracking)
    |
    v
LiveRuntime Integration (proactive response -> synthesize_and_play -> artifact log)
```

Each layer has a single responsibility:

- **TTSProvider** defines the contract for converting text to audio.
- **Concrete Providers** implement synthesis for a specific engine.
- **SpeechService** manages the speech queue, prevents overlapping playback, coordinates artifact logging, and handles async audio output.
- **LiveRuntime** calls the SpeechService after generating a proactive response, closing the loop from perception to speech.

## Provider Interface

All TTS backends implement the `TTSProvider` abstract base class:

```python
class TTSProvider(ABC):

    @abstractmethod
    def name(self) -> str:
        """Return the short identifier for this provider (e.g. 'kokoro')."""

    @abstractmethod
    async def is_available(self) -> bool:
        """Check whether the provider is ready (model loaded, dependencies present)."""

    @abstractmethod
    async def info(self) -> TTSBackendInfo:
        """Return metadata about the provider: name, availability, voices, license."""

    @abstractmethod
    async def synthesize(self, text: str, voice: str | None = None) -> SpeechSynthesisResult:
        """Synthesize text to a WAV file. Returns the result with wav_path and metadata."""
```

## Backends

| Backend  | Engine       | License      | Status      | Notes                                      |
|----------|-------------|-------------|-------------|---------------------------------------------|
| `null`   | Silent       | N/A          | Available   | No-op provider for testing and CI. Produces empty WAV files. |
| `kokoro` | Kokoro-82M   | Apache-2.0   | **Default** | Lightweight neural TTS. ~82M parameters. Runs on CPU. |
| `piper`  | Piper TTS    | MIT          | Placeholder | Planned alternative backend. Not yet implemented. |

The active backend is selected via the `BAYMAX_TTS_BACKEND` environment variable. If the requested backend is not available at runtime, the system falls back to `null`.

## SpeechService

The `SpeechService` is the central coordinator for all TTS activity:

- **Bounded speech queue**: Limits the number of pending speech requests to prevent unbounded memory growth. When the queue is full, the oldest pending request is dropped.
- **Non-overlapping playback**: Only one audio clip plays at a time. New synthesis results wait in the queue until the current playback finishes. This prevents garbled overlapping speech.
- **Artifact tracking**: Every synthesis attempt (successful or not) is logged to `speech.jsonl` in the session artifact directory.
- **Async playback via sounddevice**: Audio playback runs in a background task using `sounddevice` so it does not block the main event loop. If `sounddevice` or PortAudio is unavailable, playback is skipped but the WAV file is still saved.

Key methods:

```python
async def synthesize_and_play(self, text: str, trigger_event: str | None = None) -> SpeechSynthesisResult
async def get_queue_status(self) -> SpeechQueueStatus
async def shutdown(self) -> None
```

## LiveRuntime Integration

TTS is wired into the live webcam runtime at the point where proactive responses are generated:

1. `LiveRuntime` detects a trigger event (e.g., new user recognized, session milestone, emotional shift).
2. `_generate_proactive_response()` produces a `SupportiveResponse` with text content.
3. The response text is passed to `SpeechService.synthesize_and_play()`.
4. The SpeechService synthesizes audio via the active provider, saves the WAV file, plays it (if audio output is available), and logs a `SpeechEvent` artifact.

```
Perception Event
    -> Orchestrator._generate_proactive_response()
        -> SupportiveResponse.text
            -> SpeechService.synthesize_and_play(text, trigger_event="proactive_response")
                -> TTSProvider.synthesize(text)
                -> sounddevice playback (if available)
                -> speech.jsonl artifact entry
```

## Configuration

| Environment Variable             | Default       | Description                                           |
|----------------------------------|---------------|-------------------------------------------------------|
| `BAYMAX_TTS_BACKEND`            | `kokoro`      | Active TTS backend (`null`, `kokoro`, `piper`).       |
| `BAYMAX_TTS_VOICE`              | `af_heart`    | Voice identifier passed to the active provider.       |
| `BAYMAX_TTS_ENABLED`            | `true`        | Master switch to enable or disable TTS entirely.      |
| `BAYMAX_TTS_AUDIO_PLAYBACK`     | `true`        | Enable local audio playback via sounddevice.          |
| `BAYMAX_TTS_QUEUE_SIZE`         | `8`           | Maximum number of pending speech requests in queue.   |
| `BAYMAX_TTS_WAV_DIR`            | `artifacts/`  | Directory for saving WAV output files.                |
| `BAYMAX_TTS_SPEED`              | `1.0`         | Playback speed multiplier (provider-dependent).       |
| `BAYMAX_TTS_SAMPLE_RATE`        | `24000`       | Audio sample rate in Hz for output WAV files.         |

## Audio Playback

Local audio playback uses the `sounddevice` library, which depends on the PortAudio C library:

- **When PortAudio is available**: WAV audio is played through the default output device in a non-blocking background task.
- **When PortAudio is not available**: Playback is gracefully skipped. A warning is logged, but the WAV file is still written to disk. This allows headless servers and CI environments to run the full pipeline without audio hardware.
- **WAV files are always saved**: Regardless of whether playback succeeds, the synthesized audio is persisted as a WAV file in the artifact directory. This ensures artifacts are complete for later review, debugging, or offline playback.

## Artifact Logging

Every TTS synthesis attempt produces an entry in `speech.jsonl` within the session artifact directory. Each entry is a JSON object with the following fields:

```json
{
  "timestamp": "2026-03-16T10:30:00.123456",
  "text": "Welcome back! It is good to see you again.",
  "backend": "kokoro",
  "voice": "af_heart",
  "wav_path": "artifacts/session_abc123/speech_001.wav",
  "duration_ms": 2340,
  "success": true,
  "error": null,
  "trigger_event": "proactive_response"
}
```

Fields:

| Field           | Type    | Description                                                |
|-----------------|---------|------------------------------------------------------------|
| `timestamp`     | string  | ISO-8601 timestamp of the synthesis request.               |
| `text`          | string  | The input text that was synthesized.                       |
| `backend`       | string  | The TTS backend used (`null`, `kokoro`, `piper`).          |
| `voice`         | string  | The voice identifier used for synthesis.                   |
| `wav_path`      | string  | Path to the saved WAV file.                                |
| `duration_ms`   | int     | Duration of the synthesized audio in milliseconds.         |
| `success`       | bool    | Whether synthesis completed without error.                 |
| `error`         | string  | Error message if synthesis failed, `null` otherwise.       |
| `trigger_event` | string  | What triggered this speech (e.g., `proactive_response`).   |

## Schemas

The TTS subsystem defines the following Pydantic schemas:

- **`TTSBackendInfo`**: Metadata about a TTS provider -- name, availability status, supported voices, license, and model size.
- **`SpeechSynthesisResult`**: Result of a single synthesis call -- success flag, wav_path, duration, error details, backend and voice used.
- **`SpeechEvent`**: A timestamped log entry for `speech.jsonl` combining the synthesis result with trigger context.
- **`SpeechQueueStatus`**: Current state of the speech queue -- pending count, active playback flag, total synthesized count, total errors.
- **`TTSSmokeTestResult`**: Result of the TTS smoke test -- backend tested, synthesis success, playback success, wav_path, error details.
- **`LiveVerificationRun`**: Complete record of a live verification session -- session ID, duration, events processed, responses generated, speech events, artifact paths.

## API

The FastAPI application exposes a TTS status endpoint:

```
GET /v1/tts/backends
```

Response:

```json
{
  "available_backends": ["null", "kokoro"],
  "active_backend": "kokoro",
  "active_voice": "af_heart",
  "audio_playback_enabled": true
}
```

This endpoint allows clients and monitoring tools to check which TTS backends are installed and which one is currently active.

## Design Decisions

1. **TTS is separate from the planner and dialogue modules.** Speech synthesis is an output concern, not a planning concern. Keeping it outside the planner means response strategies remain testable without audio dependencies, and TTS can be toggled independently.

2. **The speech queue prevents overlapping playback.** Without a queue, rapid successive responses would produce garbled overlapping audio. The bounded queue with serial playback ensures each response is heard clearly before the next begins.

3. **Kokoro was chosen as the default backend for its Apache-2.0 license and lightweight footprint.** At ~82M parameters it runs comfortably on CPU, requires no API keys, and its permissive license avoids distribution restrictions. This aligns with Bay-Max's goal of being self-contained.

4. **Graceful degradation is a first-class design goal.** If the TTS backend fails to initialize, if PortAudio is missing, or if audio hardware is unavailable, the system continues operating without speech. Warnings are logged but no exceptions propagate to the main pipeline.

5. **WAV files are always saved, even when playback is unavailable.** This ensures the artifact record is complete regardless of runtime environment. WAV files can be reviewed offline, used for regression testing, or played back manually.

6. **Artifact logging captures every synthesis attempt, including failures.** Logging both successes and failures in `speech.jsonl` provides a complete audit trail for debugging, performance analysis, and verification of end-to-end pipeline health.

# Iteration 007 Report: Webcam Verification + TTS

**Branch:** `feature/iteration-007-webcam-verification-tts`
**Base:** `feature/iteration-006-live-webcam-continuity`
**Status:** Completed
**Date:** 2026-03-17

---

## Summary

Iteration 007 adds a full text-to-speech (TTS) spoken output layer to Bay-Max. The companion can now synthesize spoken responses using Kokoro-82M, an open-weight Apache-2.0 local neural TTS model. A modular provider abstraction supports three backends (null/kokoro/piper), with a bounded speech queue and optional audio playback. TTS is integrated directly into the LiveRuntime so proactive responses are automatically spoken aloud.

All 296 automated tests pass (256 existing + 40 new). Lint clean. WAV synthesis verified on the coding machine. Audio playback and live webcam tests require a non-headless machine and are documented for local verification.

---

## What Was Built

### TTS Module (`src/baymax/tts/`)

| File | Purpose |
|------|---------|
| `schemas.py` | 8 Pydantic v2 models: TTSBackendName, TTSBackendInfo, SpeechSynthesisResult, SpeechEvent, SpeechQueueStatus, TTSSmokeTestResult, LiveVerificationRun |
| `provider.py` | TTSProvider ABC, NullTTSProvider, `get_tts_provider()` factory |
| `kokoro_provider.py` | KokoroTTSProvider — lazy KPipeline init, af_heart voice, 24000 Hz, WAV via soundfile |
| `piper_provider.py` | PiperTTSProvider — placeholder with graceful unavailability handling |
| `speech_service.py` | SpeechService — synthesis + bounded deque queue + optional sounddevice playback + event tracking |

### TTS Backends

| Backend | Status | License | Notes |
|---------|--------|---------|-------|
| `null` | Implemented | — | Silent provider for testing/CI. Always available. |
| `kokoro` | Implemented | Apache-2.0 | Default. Kokoro-82M neural TTS. WAV synthesis verified (3.45s test). |
| `piper` | Partial | MIT | Interface implemented. Package not installed. Returns graceful error. |

### LiveRuntime Integration

- TTS synthesis happens after proactive response generation in `_generate_proactive_response()`
- Speech events logged to `speech.jsonl` via `ArtifactLogger.log_speech_event()`
- `LiveRuntimeStatus` extended with `last_spoken_text`, `speech_queue_depth`, `tts_backend`, `tts_voice`
- Debug HUD toggle via `BAYMAX_ENABLE_LIVE_DEBUG_HUD` setting

### API Endpoints

| Endpoint | Method | New/Modified |
|----------|--------|-------------|
| `/v1/tts/backends` | GET | New — returns available backends, active config, playback status |
| `/v1/live/status` | GET | Modified — includes speech state fields |

### CLI Enhancements

`scripts/live_companion.py` now supports:
- `--no-tts` — disable TTS synthesis
- `--tts-backend <name>` — override TTS backend
- `--no-playback` — generate WAV files without playing audio

### Verification Script

`scripts/verify_007.py` supports:
- `--tts-only` — TTS smoke test only (no webcam)
- `--replay <path>` — replay video file
- `--no-playback` — skip audio playback
- `--duration <sec>` — limit run duration
- Writes `manifest.json` and `summary.md` to `artifacts/verification_007/`

---

## Test Results

### Automated Tests

```
python3.11 -m pytest tests/ -v
296 passed (256 existing + 40 new)
```

### New Test Classes (40 tests)

| Class | Tests | Coverage |
|-------|-------|----------|
| TestTTSSchemas | 9 | All TTS Pydantic schema models |
| TestTTSProviders | 7 | Null, factory, kokoro, piper providers |
| TestSpeechService | 6 | Synthesis, queue status, events, async path |
| TestKokoroProvider | 5 | Create, availability, WAV synthesis, empty text, info |
| TestArtifactSpeechLogging | 3 | Speech event JSONL logging |
| TestLiveStatusSpeechFields | 2 | LiveRuntimeStatus speech fields |
| TestTTSAPIEndpoints | 2 | /v1/tts/backends, live status speech fields |
| TestTTSSettings | 2 | Default TTS settings, artifact dir path |
| TestTTSGracefulFailure | 3 | Broken voice, backend error, async failure |
| TestProjectState007 | 1 | project_state.json structural validity |

### Lint

```
python3.11 -m ruff check src/ apps/ tests/ scripts/
All checks passed! (27 auto-fixes applied during development)
```

### TTS Smoke Test

```
python3.11 scripts/verify_007.py --tts-only --no-playback
```

Result: Kokoro synthesized a **3.45-second WAV file** (af_heart voice, 24000 Hz). Artifact saved to `artifacts/verification_007/tts_smoke_test.wav`.

---

## Manual Verification Status

| ID | Case | Result | Notes |
|----|------|--------|-------|
| VT701 | Real webcam startup | not_run | Sol is headless — no camera |
| VT702 | Known-user greeting with speech | not_run | Requires webcam + enrolled face |
| VT703 | Unknown-user handling | not_run | Requires webcam |
| VT704 | Cooldown suppression with speech | not_run | Requires live session |
| VT705 | Leave and resume session | not_run | Requires webcam |
| VT706 | Quiet companionship spoken prompt | not_run | Requires live session |
| VT707 | TTS backend failure fallback | **passed** | Automated tests verify graceful failure |
| VT708 | Artifact proof completeness | **passed** | manifest.json + summary.md + WAV confirmed |
| VT709 | Overlay/debug status check | not_run | Sol has no display |
| VT710 | Safety in spoken mode | **passed** | Safety gating runs before TTS synthesis |

**Honest assessment:** WAV file generation is proven. Audio playback and live webcam tests require the user's local Ubuntu laptop with camera, speakers, and PortAudio installed.

---

## Configuration

### New Environment Variables (12)

| Variable | Default | Description |
|----------|---------|-------------|
| `BAYMAX_TTS_BACKEND` | `kokoro` | Active TTS backend |
| `BAYMAX_TTS_VOICE` | `af_heart` | Voice selection |
| `BAYMAX_TTS_RATE` | `24000` | Sample rate (Hz) |
| `BAYMAX_TTS_ENABLED` | `true` | Enable/disable TTS |
| `BAYMAX_TTS_DEVICE` | `auto` | Inference device |
| `BAYMAX_TTS_OUTPUT_DIR` | `./artifacts/tts` | WAV output directory |
| `BAYMAX_ENABLE_AUDIO_PLAYBACK` | `true` | Enable speaker output |
| `BAYMAX_AUDIO_BACKEND` | `sounddevice` | Audio output library |
| `BAYMAX_KOKORO_VOICE` | `af_heart` | Kokoro-specific voice |
| `BAYMAX_PIPER_MODEL_PATH` | `` | Piper ONNX model path |
| `BAYMAX_PIPER_CONFIG_PATH` | `` | Piper config path |
| `BAYMAX_ENABLE_LIVE_DEBUG_HUD` | `true` | Debug overlay toggle |

---

## Documentation Updated

| Document | Status |
|----------|--------|
| README.md | Updated — TTS section, install-tts target, live-webcam-tts |
| docs/ARCHITECTURE.md | Updated — TTS output layer added |
| docs/ROADMAP.md | Updated — iteration-007 marked completed |
| docs/CHANGELOG.md | Updated — iteration-007 entry |
| docs/MODEL_STACK.md | Updated — Kokoro-82M and Piper entries |
| docs/LOCAL_TESTING.md | Updated — TTS testing commands |
| docs/PROJECT_MEMORY.md | Updated — iteration-007 context |
| docs/project_state.json | Updated — iteration 007, tts module, endpoints |
| docs/TTS_ARCHITECTURE.md | **New** — comprehensive TTS architecture |
| docs/LIVE_VERIFICATION_GUIDE.md | **New** — webcam + TTS verification guide |
| pyproject.toml | Updated — version 0.6.0, tts extras |
| .env.example | Updated — 12 new TTS vars |
| Makefile | Updated — install-tts, live-webcam-tts, verify-007 |

---

## Known Issues

1. **PortAudio not installed on Sol** — sounddevice audio playback unavailable on headless machine. WAV generation works.
2. **Piper TTS is partial** — interface implemented but package not installed.
3. **torch version conflict** — facenet-pytorch pins torch<2.3 but torch 2.10.0 installed. Works at runtime.
4. **TTS synthesis latency** — ~3-4s for short phrases. May cause delay between event and speech.
5. **SQLite async gap** — still using synchronous sqlite3 (pre-existing from earlier iterations).

---

## Artifacts

| File | Location |
|------|----------|
| manifest.json | `artifacts/verification_007/manifest.json` |
| summary.md | `artifacts/verification_007/summary.md` |
| TTS smoke WAV | `artifacts/verification_007/tts_smoke_test.wav` |

---

## Next Steps

1. Run full webcam + TTS verification on local Ubuntu laptop
2. Install PortAudio (`sudo apt install libportaudio2`) for audio playback
3. Complete Piper TTS backend
4. Add Whisper STT for speech input (spoken dialogue loop)
5. Move TTS synthesis to background thread for lower latency
6. Migrate SQLite to fully async aiosqlite

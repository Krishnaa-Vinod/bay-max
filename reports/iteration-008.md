# Iteration 008: Bidirectional Speech Report

## Executive Summary

**Status**: ✅ COMPLETED
**Branch**: `feature/iteration-008-bidirectional-speech`
**Commit SHA**: `84d95d4`
**Version**: 0.7.0

Successfully implemented full bidirectional speech input pipeline for Bay-Max, enabling users to speak to the companion and receive spoken responses. The system now supports:

- **Speech Input**: Microphone capture → Silero VAD → faster-whisper ASR → echo suppression → dialogue integration
- **Companion Persona**: Baymax-inspired calm, gentle, literal tone injected into system prompts
- **Speaking Lock**: Prevents echo loops by discarding microphone input while TTS is active
- **Push-to-Talk**: Alternative mode for debugging alongside continuous VAD
- **Memory Integration**: Spoken turns enter the same session/memory pipeline as typed turns

All 356 tests pass (60 new), lint clean, documentation updated.

## Implementation Details

### New Audio Package (`src/baymax/audio/`)

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `schemas.py` | Pydantic models for audio data | `AudioChunkInfo`, `VADDecision`, `SpeechSegment`, `TranscriptionResult`, `EchoDecision` |
| `microphone.py` | Audio input capture | `MicrophoneCapture` (sounddevice), `NullMicrophone` (headless) |
| `vad.py` | Voice activity detection | `SileroVAD` (4-state machine), `NullVAD` (passthrough) |
| `transcriber.py` | Speech-to-text backends | `FasterWhisperProvider`, `NullASRProvider` |
| `echo_suppression.py` | Prevents TTS feedback loops | `EchoSuppressor` (text similarity-based) |
| `speech_input_service.py` | Pipeline orchestrator | `SpeechInputService` (async coordinator) |

### Speech Pipeline Flow

```
Microphone → VAD → ASR → Echo Check → ChatTurn(source='speech') → Memory → Dialogue → TTS
```

### Configuration (18 new env vars)

| Variable | Default | Purpose |
|----------|---------|---------|
| `BAYMAX_ENABLE_SPEECH_INPUT` | `true` | Master speech input toggle |
| `BAYMAX_STT_BACKEND` | `faster_whisper` | ASR provider selection |
| `BAYMAX_WHISPER_MODEL` | `base.en` | Whisper model size |
| `BAYMAX_VAD_BACKEND` | `silero` | Voice activity detector |
| `BAYMAX_VAD_THRESHOLD` | `0.65` | Speech detection sensitivity |
| `BAYMAX_MIC_MODE` | `vad` | Continuous vs push-to-talk |
| `BAYMAX_POST_SPEECH_COOLDOWN_MS` | `1500` | Post-TTS mic resume delay |
| `BAYMAX_ECHO_SIMILARITY_THRESHOLD` | `0.85` | Echo suppression sensitivity |

### API Extensions

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/audio/status` | GET | Speech input/output runtime status |
| `/v1/stt/backends` | GET | STT backend configuration |

### Persona Integration

Created `docs/COMPANION_PERSONA.md` with Baymax-inspired style guide:
- **Tone**: Calm, literal, helpful, gentle, nonjudgmental
- **Pattern**: Brief acknowledgment + remembered detail + supportive sentence + optional gentle question
- **Avoids**: Sarcasm, hype, excessive exclamation points, medical certainty

Injected via `_PERSONA_STYLE` constant in `src/baymax/dialogue/prompt_builder.py`.

## Testing Results

### Automated Tests
- **Total**: 356 tests (60 new for iteration 008)
- **Status**: ✅ All passed
- **Lint**: ✅ 0 errors (ruff check)
- **Coverage**: 12 test classes covering audio schemas, VAD state machine, ASR providers, echo suppression, speech service, persona injection, API endpoints

### Manual Verification
- **Hardware**: Sol (NVIDIA A100-SXM4-80GB, headless)
- **Real Speech**: ❌ Not tested (no microphone/speakers on Sol)
- **Stubs**: ✅ NullMicrophone, NullVAD, NullASRProvider verified
- **Model Loading**: ✅ faster-whisper base.en loads successfully
- **Transcription**: ✅ Silence transcription works (returns empty text)

**Note**: All VT801-VT810 manual tests marked `not_run` due to headless environment. Real hardware verification requires machine with microphone and speakers.

## Files Created/Modified

### New Files (11)
- `src/baymax/audio/__init__.py`
- `src/baymax/audio/schemas.py`
- `src/baymax/audio/microphone.py`
- `src/baymax/audio/vad.py`
- `src/baymax/audio/transcriber.py`
- `src/baymax/audio/echo_suppression.py`
- `src/baymax/audio/speech_input_service.py`
- `docs/COMPANION_PERSONA.md`
- `tests/test_iteration_008.py`
- `reports/iteration-008.json`
- `reports/iteration-008.md`

### Modified Files (19)
- `src/baymax/config/settings.py` - 18 new speech settings
- `src/baymax/schemas/memory.py` - ChatTurn.source field
- `src/baymax/live/schemas.py` - Speech status fields
- `src/baymax/live/runtime.py` - Full rewrite with speech integration
- `src/baymax/orchestrator/service.py` - Source parameter
- `src/baymax/dialogue/prompt_builder.py` - Persona injection
- `apps/api/main.py` - New audio endpoints, version bump
- `scripts/live_companion.py` - Speech CLI flags
- `pyproject.toml` - Version 0.7.0, speech dependencies
- `Makefile` - Speech targets
- `.env.example` - 18 new variables
- `README.md` - Speech input section
- `docs/ARCHITECTURE.md` - Iteration 008 summary
- `docs/CHANGELOG.md` - 0.7.0 entry
- `docs/MODEL_STACK.md` - VAD/ASR sections
- `docs/PROJECT_MEMORY.md` - Audio module context
- `docs/ROADMAP.md` - Status update
- `docs/LOCAL_TESTING.md` - Speech setup instructions
- `docs/project_state.json` - Iteration 008 status

## Known Issues & Limitations

1. **Hardware Testing Gap**: No real microphone/speaker testing on Sol (headless HPC environment)
2. **VAD Tuning**: Silero VAD threshold (0.65) may need adjustment for different microphone hardware
3. **Cooldown Timing**: Post-speech cooldown (1500ms) may need tuning for different room acoustics
4. **Model Dependencies**: faster-whisper models download on first use (requires internet)
5. **Audio Quality**: No noise reduction or audio preprocessing beyond VAD

## Next Recommended Tasks

1. **Hardware Verification**: Test on machine with microphone and speakers
2. **VAD Tuning**: Experiment with threshold and timing parameters for different environments
3. **ASR Optimization**: Add openai-whisper backend option for compatibility
4. **Audio Preprocessing**: Consider noise reduction, automatic gain control
5. **Barge-in Support**: Allow interrupting Bay-Max mid-speech (future iteration)
6. **Wake Word**: Optional wake word activation (future iteration)
7. **Speaker Diarization**: Multi-speaker support (future iteration)

## Architecture Impact

The speech input system integrates cleanly with existing patterns:
- **Memory**: Spoken turns use same ChatTurn schema with `source='speech'`
- **Dialogue**: No changes needed - same response generation pipeline
- **TTS**: Enhanced with speaking lock coordination
- **Live Runtime**: Extended with parallel speech input async task
- **API**: New endpoints follow existing patterns
- **Settings**: Uses standard Pydantic + env var pattern

The modular design allows each component (microphone, VAD, ASR, echo suppression) to be replaced independently.

---

**Deliverables**: ✅ Completed
**Branch Pushed**: ✅ `feature/iteration-008-bidirectional-speech`
**Tests Passing**: ✅ 356/356
**Documentation**: ✅ Updated
**Version Bumped**: ✅ 0.7.0
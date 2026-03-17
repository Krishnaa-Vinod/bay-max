# Emotion Architecture (Iteration 009)

## Overview

This document describes Bay-Max's facial affect analysis architecture, implemented in Iteration 009 to add emotion-aware response adaptation while maintaining strict non-clinical boundaries.

## Design Principles

1. **Non-Clinical Focus**: Bay-Max analyzes affect to adapt tone and strategy, never to diagnose medical or psychological conditions
2. **Replaceable Backends**: Multiple affect analysis backends are supported through a common interface
3. **Confidence Gating**: All affect-based decisions require sufficient confidence thresholds
4. **Graceful Degradation**: System operates normally when affect analysis fails or is unavailable
5. **Privacy First**: No raw face images are stored; only aggregated affect summaries enter long-term memory
6. **Explainability**: Affect decisions include evidence and can be traced through artifact logs

## Architecture Components

### Backend Abstraction Layer

**Interface**: `EmotionAnalyzer` (baymax.perception.emotion)

All backends implement:
- `analyze(frame, face_bbox) -> EmotionResult`
- `is_available() -> bool`
- `model_info() -> dict`

**Available Backends**:
- `NullEmotionAnalyzer`: Always available fallback, returns neutral affect
- `MediaPipeEmotionAnalyzer`: Uses MediaPipe Face Landmarker blendshapes for affect estimation

**Py-Feat Status**: Initially planned but incompatible with Python 3.11 environment due to setuptools conflicts. MediaPipe used as primary real backend instead.

### Affect Processing Pipeline

```
Raw Frame -> Face Detection -> Emotion Analysis -> Smoothing -> Strategy Bias -> Response
                                      ↓
                               Timeline Logging -> Memory Consolidation (if stable)
```

1. **Face Detection**: Uses existing face detection pipeline from earlier iterations
2. **Emotion Analysis**: Backend processes face region, returns valence/arousal/confidence
3. **Smoothing**: Exponential Moving Average (EMA) with stability tracking
4. **Strategy Bias**: Soft adjustments to response strategy when confidence sufficient
5. **Memory Consolidation**: Session-level affect summaries (only if stable and confident)

### Affect State Representation

**Dimensional Model**: Valence/Arousal space rather than discrete emotion labels

- **Valence**: -1.0 (negative) to +1.0 (positive)
- **Arousal**: 0.0 (calm) to 1.0 (activated)
- **Confidence**: 0.0 to 1.0 belief in the analysis
- **Stability Duration**: How long the state has been consistent

**Evidence Types**:
- MediaPipe: Face blendshape activations
- Processing metadata: latency, frame quality, method used

### Smoothing and Stability

**Exponential Moving Average (EMA)**:
- Configurable alpha parameter (default 0.25)
- Reduces noise from single-frame variations
- Maintains sample count and update timestamps

**Stability Tracking**:
- Monitors changes above threshold (default 0.1)
- Tracks duration of stable periods
- Requires minimum stability (default 30 seconds) for memory consolidation

### Strategy Integration

**Planner Bias** (baymax.planner.supportive_planner):
- Soft adjustments to response strategy based on affect patterns
- Never overrides critical interactions (greeting, recall requests)
- Confidence-gated (default 0.6 threshold)

**Affect Patterns**:
- Low valence + low arousal → Gentler, more validating responses
- Low valence + high arousal → Calming, empathetic approach
- High valence + high arousal → More engaging, upbeat tone
- High valence + low arousal → Warm, peaceful approach

**Prompt Context** (baymax.dialogue.prompt_builder):
- Internal affect observations added to prompt when confidence sufficient
- Non-diagnostic phrasing ("appears subdued" not "is depressed")
- Separate from user-facing conversation

### Memory Integration

**Consolidation Safety** (baymax.memory.clinical_safety):
- Clinical language blocklist enforcement
- Automatic sanitization of unsafe terms
- Content validation pipeline

**Session-Level Summaries**:
- Created only for stable, confident affect patterns
- Stored as episodic memories with lower salience
- Include stability evidence and duration context
- Example: "User appeared calm and content during this session over 18.2 minutes"

**Memory Requirements**:
- Minimum 5-minute session duration
- Confidence ≥ 0.65
- Stability duration ≥ 30 seconds
- Pass clinical safety validation

## Configuration

### Environment Variables

All affect settings use `BAYMAX_AFFECT_*` prefix:

```bash
# Core settings
BAYMAX_ENABLE_AFFECT=true
BAYMAX_AFFECT_BACKEND=mediapipe
BAYMAX_AFFECT_DEVICE=auto
BAYMAX_AFFECT_SAMPLE_EVERY_N_FRAMES=10

# Analysis thresholds
BAYMAX_AFFECT_CONFIDENCE_THRESHOLD=0.60
BAYMAX_AFFECT_SMOOTHING_ALPHA=0.25
BAYMAX_AFFECT_STABILITY_DURATION_SEC=30.0

# Memory consolidation
BAYMAX_AFFECT_MEMORY_CONFIDENCE_THRESHOLD=0.65

# Strategy bias
BAYMAX_AFFECT_BIAS_ENABLED=true
BAYMAX_AFFECT_BIAS_CONFIDENCE_THRESHOLD=0.60
BAYMAX_AFFECT_NEGATIVE_VALENCE_THRESHOLD=-0.30
BAYMAX_AFFECT_POSITIVE_VALENCE_THRESHOLD=0.40
BAYMAX_AFFECT_LOW_AROUSAL_THRESHOLD=0.40
BAYMAX_AFFECT_HIGH_AROUSAL_THRESHOLD=0.55

# Debug and artifacts
BAYMAX_AFFECT_DEBUG=false
BAYMAX_AFFECT_ARTIFACT_DIR=./artifacts/affect
```

### Sampling Strategy

- Affect analysis runs at lower cadence than face detection
- Default: Every 10th frame to balance accuracy with performance
- MediaPipe Face Landmarker is computationally intensive
- Sampling frequency configurable based on hardware capabilities

## Safety and Limitations

### Clinical Safety Boundaries

**Explicit Prohibitions**:
- No diagnostic claims or medical certainty statements
- No mental health condition labels (depression, anxiety, etc.)
- No treatment or medication recommendations
- No physiological symptom attribution

**Safe Phrasing Examples**:
- ✅ "User appeared subdued and thoughtful"
- ✅ "User seemed energetic and engaged"
- ❌ "User shows signs of depression"
- ❌ "User exhibits anxiety symptoms"

### Technical Limitations

**MediaPipe Blendshapes**:
- Simplified affect estimation from facial movement patterns
- Not clinically validated for emotion recognition
- Performance depends on lighting, face angle, occlusion
- Works best with front-facing, well-lit faces

**Confidence and Uncertainty**:
- System designed to remain uncertain when evidence is weak
- Low confidence predictions are discarded rather than guessed
- Occlusion, poor lighting, or extreme angles yield low confidence

**Cultural and Individual Variation**:
- Facial expressions vary significantly across cultures and individuals
- System makes no claims about universal emotion expression
- Affect patterns are treated as soft cues, not definitive states

## API Endpoints

### GET /v1/emotion/backends

Returns available and configured affect analysis backends:

```json
{
  "available_backends": ["null", "mediapipe"],
  "active_backend": "mediapipe",
  "active_model_or_runtime": "MediaPipe Face Landmarker with blendshapes",
  "enabled": true,
  "sample_every_n_frames": 10
}
```

### GET /v1/live/status (Extended)

Live runtime status now includes affect fields when enabled:

```json
{
  "affect_enabled": true,
  "affect_backend": "mediapipe",
  "emotion_valence": 0.23,
  "emotion_arousal": 0.45,
  "emotion_confidence": 0.78,
  "emotion_stable_duration_sec": 42.3,
  "emotion_debug_summary": "User appears calm and content"
}
```

## Artifact Logging

### Timeline Artifacts

**JSON Format** (`session_*_timeline.json`):
- Timestamped affect readings with raw and smoothed values
- Backend information and confidence scores
- Strategy change annotations

**CSV Format** (`session_*_timeline.csv`):
- Same data in spreadsheet-compatible format
- Suitable for analysis and visualization tools

### Strategy Change Log

**Format** (`session_*_strategy_changes.json`):
```json
[
  {
    "timestamp": "2026-03-17T22:15:30",
    "old_strategy": "encourage",
    "new_strategy": "empathize",
    "reason": "Low valence + low arousal detected",
    "affect_context": {"valence": -0.4, "arousal": 0.3, "confidence": 0.8}
  }
]
```

### Verification Manifests

Generated for replay testing and verification:
- Expected system behaviors for different affect patterns
- Verification checkpoints with confidence thresholds
- Notes about backend compatibility and limitations

## Testing and Verification

### Automated Test Coverage

**Schema Validation**: All affect-related Pydantic models
**Backend Tests**: Null and MediaPipe analyzer functionality
**Smoothing Logic**: EMA behavior and stability tracking
**Clinical Safety**: Language detection and sanitization
**Integration Tests**: Planner bias and prompt context inclusion
**Configuration**: Settings validation and defaults

### Replay Testing System

**Test Data Structure**:
```
test_data/affect_replay/
  positive_expression.json    # High valence test cases
  neutral_expression.json     # Baseline behavior
  low_affect_expression.json  # Low valence scenarios
  occlusion_test.json        # Low confidence handling
  smile_clear.jpg            # Test images
  neutral_clear.jpg
  subdued_clear.jpg
  turned_away.jpg
```

**Verification Script**: `scripts/affect_replay_test.py`
- Creates standard test sequences
- Runs backend analysis on saved images
- Generates verification reports and artifacts
- Compares actual vs expected affect ranges

### Manual Verification Guidelines

1. **Positive Expression**: System adapts to be naturally more engaging
2. **Subdued Expression**: System uses gentler, more validating tone
3. **Neutral Expression**: System maintains default supportive approach
4. **Poor Quality/Occluded**: System remains uncertain, no hard guesses
5. **Stable Sessions**: Affect summaries written to memory when stable
6. **Tone Consistency**: Baymax persona preserved across all affect states

## Implementation Notes

### MediaPipe Integration

**Face Landmarker Requirements**:
- Auto-downloads model on first use (`face_landmarker.task`)
- Requires RGB input format
- Returns 478 facial landmarks + blendshape coefficients
- Best performance with single faces in good lighting

**Blendshape-to-Affect Mapping**:
- Positive affect: smile activations, cheek movements, eye squinting (Duchenne-like)
- Negative affect: frowning, mouth corner depression
- Arousal: eye widening, jaw opening, overall facial activation
- Custom heuristic mapping (not clinically validated)

### Performance Considerations

**Computational Load**:
- MediaPipe Face Landmarker: ~50-100ms per analysis on CPU
- Sampling every 10th frame at 8fps = ~0.8fps affect analysis
- Memory overhead: Timeline logging grows with session duration

**Memory Management**:
- Affect timelines cleared between sessions
- Long-term memory: Only aggregated summaries, no raw affect data
- Artifact logging can be disabled for production use

### Future Extension Points

**Additional Backends**:
- Interface ready for py-feat integration when compatible
- Extensible to other affect analysis methods
- Support for audio-based affect analysis

**Enhanced Features**:
- Multi-person affect tracking with recognition integration
- Affect pattern learning over multiple sessions
- Integration with physiological sensors (future hardware)

**Clinical Integration**:
- Framework designed to support clinical use cases with appropriate validation
- Strict safety boundaries can be adjusted for research contexts
- Provider supervision and validation workflows could be added
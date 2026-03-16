# Local Testing Guide

This document explains how to test Bay-Max locally on a headless machine (e.g., HPC cluster nodes) without a display.

## Prerequisites

Install all dependencies including vision extras:

```bash
pip install -e ".[all]"
```

This installs `torch`, `facenet-pytorch`, `opencv-python`, and `mediapipe` alongside dev dependencies.

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

1. `pip install -e ".[dev]"` completes without errors
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

## Troubleshooting

- **MediaPipe import error**: Ensure `mediapipe>=0.10.0` is installed. On some HPC systems, you may need to install from a wheel.
- **No CUDA**: Set `BAYMAX_DEVICE=cpu` in `.env`
- **No display (cv2 error)**: The pipeline uses headless OpenCV (`cv2.imwrite`) and does not require a display server.
- **Stub fallback**: If MediaPipe fails to load, the system silently falls back to `StubPoseEstimator` (no pose detected). Check logs for warnings.

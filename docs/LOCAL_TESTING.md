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

# Run only iteration 003 tests
pytest tests/test_iteration_003.py -v

# Run with lint
make lint
```

## Gradio Demo

```bash
make run-demo
```

Opens on `http://0.0.0.0:7860`. The Frame Analysis tab shows both JSON results and an annotated image overlay.

## Smoke Test Checklist

1. `pip install -e ".[all]"` completes without errors
2. `make lint` passes (0 errors)
3. `make test` passes (109 tests)
4. `python scripts/local_test.py --enroll face.jpg --name "Test"` produces enrollment JSON
5. `python scripts/local_test.py --image test.jpg` produces analysis JSON and annotated artifact
6. Annotated artifact file exists in `./artifacts/`

## Troubleshooting

- **MediaPipe import error**: Ensure `mediapipe>=0.10.0` is installed. On some HPC systems, you may need to install from a wheel.
- **No CUDA**: Set `BAYMAX_DEVICE=cpu` in `.env`
- **No display (cv2 error)**: The pipeline uses headless OpenCV (`cv2.imwrite`) and does not require a display server.
- **Stub fallback**: If MediaPipe fails to load, the system silently falls back to `StubPoseEstimator` (no pose detected). Check logs for warnings.

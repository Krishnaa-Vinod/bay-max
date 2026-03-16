# Bay-Max Model Stack

This document describes the ML models and algorithms used in Bay-Max's perception pipeline.

## Face Detection

| Property | Value |
|----------|-------|
| **Model** | MTCNN (Multi-Task Cascaded Convolutional Networks) |
| **Library** | `facenet-pytorch` |
| **Input** | RGB image (any resolution) |
| **Output** | List of bounding boxes + detection confidence scores |
| **Default thresholds** | `(0.6, 0.7, 0.7)` for the three cascade stages |
| **Min face size** | 40 px |
| **Max faces per frame** | 5 (configurable via `BAYMAX_MAX_FACES_PER_FRAME`) |

MTCNN uses three cascaded CNNs (P-Net, R-Net, O-Net) to progressively refine face proposals. It runs locally with no external API calls.

## Face Embedding

| Property | Value |
|----------|-------|
| **Model** | InceptionResnetV1 |
| **Pretrained on** | VGGFace2 |
| **Library** | `facenet-pytorch` |
| **Embedding dimension** | 512 |
| **Input** | Aligned, cropped face tensor (160x160) |
| **Output** | L2-normalized 512-dimensional float vector |

MTCNN handles face alignment and cropping. The InceptionResnetV1 model produces a compact embedding vector that encodes facial identity. Weights are downloaded on first use and cached locally.

## Face Matching

| Property | Value |
|----------|-------|
| **Algorithm** | Cosine similarity |
| **Default threshold** | 0.75 (configurable via `BAYMAX_FACE_MATCH_THRESHOLD`) |
| **Input** | Query embedding + list of enrolled embeddings |
| **Output** | Best-matching user ID + confidence score, or None |

Embeddings are L2-normalized before comparison. The highest cosine similarity above the threshold is selected as the match. If no enrolled embedding exceeds the threshold, the face is classified as unknown.

## Face Tracking

| Property | Value |
|----------|-------|
| **Algorithm** | Greedy IoU + cosine similarity assignment |
| **IoU weight** | 0.4 |
| **Cosine weight** | 0.6 |
| **IoU threshold** | 0.3 (minimum to consider a match) |
| **Cosine threshold** | 0.6 (minimum to consider a match) |
| **Max age** | 5.0 seconds (tracks dropped if not updated) |

The tracker maintains a list of active tracks. Each frame, it computes a combined IoU + cosine score between current detections and existing tracks, then uses greedy assignment (highest score first) to maintain frame-to-frame identity continuity. New tracks are created for unmatched detections.

## Device Configuration

The compute device is controlled by the `BAYMAX_DEVICE` environment variable:

| Value | Behavior |
|-------|----------|
| `cuda_if_available_else_cpu` (default) | Uses CUDA if `torch.cuda.is_available()`, else CPU |
| `cuda` | Force CUDA (fails if unavailable) |
| `cpu` | Force CPU |

## Observation Events

The system writes observations to memory only on meaningful events:

| Event Type | Trigger |
|------------|---------|
| `first_recognition` | A known user is recognized for the first time in a session |
| `user_switch` | The primary recognized user changes within a session |
| `known_to_unknown` | A previously recognized user is no longer detected |

## Pose Estimation

| Property | Value |
|----------|-------|
| **Model** | MediaPipe Pose (legacy `mp.solutions.pose`) |
| **Library** | `mediapipe` |
| **Input** | RGB image (any resolution) |
| **Output** | 33 2D landmarks with visibility scores |
| **Min detection confidence** | 0.5 (configurable via `BAYMAX_POSE_MIN_CONFIDENCE`) |
| **Mode** | Static image mode (no temporal smoothing) |
| **Backend** | Configurable via `BAYMAX_POSE_BACKEND` (`mediapipe` or `stub`) |

MediaPipe Pose uses a two-stage pipeline: a lightweight body detector followed by a landmark regression network that predicts 33 keypoints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles, etc.). Each landmark has normalized (x, y) coordinates and a visibility score.

## Body-State Heuristics

| Heuristic | Algorithm | Thresholds |
|-----------|-----------|------------|
| **Posture** | Nose-to-shoulder ratio vs torso height | Slouch < 0.38, Recline < 0.15 |
| **Lean** | Nose offset relative to shoulder-hip midline | Forward < -0.03, Backward > 0.04 |
| **Motion** | Average landmark displacement across frames | High >= 0.04, Medium >= 0.015 |
| **Engagement** | Weighted score from all signals | High >= 0.65, Medium >= 0.35 |

All heuristics are documented in detail in `docs/ENGAGEMENT_HEURISTICS.md`.

### Body-State Observation Events

| Event Type | Trigger |
|------------|---------|
| `pose_first_seen` | First time body pose is detected in a session |
| `pose_lost` | Body pose was visible but is no longer detected |
| `posture_change` | Posture label changed (e.g. upright -> slouched) |
| `engagement_change` | Engagement level changed (e.g. medium -> high) |

## Stubbed Components

These remain unimplemented and return hardcoded values:

- **Emotion estimation**: Returns `EmotionLabel.NEUTRAL`
- **LanceDB vector store**: Interface exists, in-memory stub in use
- **Memory consolidation**: Returns empty
- **Frame source / webcam capture**: Returns empty frames

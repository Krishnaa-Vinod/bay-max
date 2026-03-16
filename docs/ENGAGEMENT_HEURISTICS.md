# Engagement Heuristics

This document describes the body-state and engagement heuristics used in Bay-Max's perception pipeline (Iteration 003). All heuristics are simple, deterministic, rule-based functions with transparent, documented thresholds. No clinical or diagnostic claims are made.

## Overview

The engagement pipeline runs after pose estimation and produces:

1. **Posture label** (upright / slouched / reclined / unknown)
2. **Lean direction** (forward / neutral / backward / unknown)
3. **Motion level** (low / medium / high / unknown)
4. **Engagement score** (0.0 - 1.0) and level (high / medium / low)

## Posture Detection

**Function**: `compute_posture(pose: PoseResult) -> PostureLabel`

Uses the vertical relationship between nose, shoulder midpoint, and hip midpoint.

### Algorithm

1. Compute `shoulder_mid_y` = average y of left and right shoulders
2. Compute `hip_mid_y` = average y of left and right hips
3. Compute `torso_height` = `hip_mid_y - shoulder_mid_y`
4. If `torso_height <= 0.01`: **RECLINED** (shoulders at or below hips)
5. If nose is visible:
   - Compute `nose_shoulder_ratio = (shoulder_mid_y - nose.y) / torso_height`
   - If ratio < 0.15: **RECLINED**
   - If ratio < 0.38: **SLOUCHED**
   - Otherwise: **UPRIGHT**
6. If nose is not visible, use torso height as proxy:
   - If `torso_height < 0.08`: **SLOUCHED**
   - Otherwise: **UPRIGHT**

### Thresholds

| Constant | Value | Meaning |
|----------|-------|---------|
| `POSTURE_SLOUCH_RATIO_THRESHOLD` | 0.38 | Below this nose-shoulder ratio -> slouched |
| `POSTURE_RECLINE_RATIO_THRESHOLD` | 0.15 | Below this -> reclined |

## Lean Detection

**Function**: `compute_lean(pose: PoseResult) -> LeanLabel`

Uses the nose position relative to the shoulder-hip midline.

### Algorithm

1. Compute `shoulder_mid_y` and `hip_mid_y` as above
2. Compute `torso_height = hip_mid_y - shoulder_mid_y`
3. If `torso_height <= 0.01`: **UNKNOWN**
4. If nose is visible:
   - Compute `nose_offset = (nose.y - shoulder_mid_y) / torso_height`
   - If `nose_offset > 0.04`: **BACKWARD**
   - If `nose_offset < -0.03`: **FORWARD**
   - Otherwise: **NEUTRAL**

### Thresholds

| Constant | Value | Meaning |
|----------|-------|---------|
| `LEAN_FORWARD_OFFSET` | -0.03 | Below this offset -> forward lean |
| `LEAN_BACKWARD_OFFSET` | 0.04 | Above this offset -> backward lean |

## Motion Tracking

**Class**: `MotionTracker(window_size=10)`

Tracks motion level using landmark displacement across consecutive frames.

### Algorithm

1. Extract positions of 5 key landmarks: nose, left/right shoulders, left/right hips
2. Store in a sliding window (default 10 frames)
3. Compute average Euclidean displacement between consecutive frames
4. Classify:
   - displacement >= 0.04: **HIGH**
   - displacement >= 0.015: **MEDIUM**
   - Otherwise: **LOW**

### Thresholds

| Constant | Value | Meaning |
|----------|-------|---------|
| `MOTION_HIGH_THRESHOLD` | 0.04 | Above this -> high motion |
| `MOTION_MEDIUM_THRESHOLD` | 0.015 | Above this -> medium motion |

## Engagement Scoring

**Function**: `compute_engagement(...) -> EngagementResult`

Computes a weighted score (0.0 - 1.0) from available signals.

### Scoring Weights

| Signal | Condition | Weight |
|--------|-----------|--------|
| Face visible | Any face detected | +0.20 |
| Face recognized | Known user matched | +0.10 |
| Pose visible | Body pose detected | +0.10 |
| Posture: upright | | +0.20 |
| Posture: slouched | | +0.05 |
| Posture: reclined | | +0.00 |
| Lean: forward | | +0.15 |
| Lean: neutral | | +0.10 |
| Lean: backward | | +0.00 |
| Motion: low | | +0.15 |
| Motion: medium | | +0.10 |
| Motion: high | | +0.05 |
| Unknown values | | +0.00 |

**Maximum possible score**: 0.90 (all positive signals)

### Level Thresholds

| Score Range | Engagement Level |
|-------------|-----------------|
| >= 0.65 | HIGH |
| >= 0.35 | MEDIUM |
| < 0.35 | LOW |

## Landmark Visibility

All heuristics require a minimum landmark visibility of 0.5 (`MIN_VISIBILITY`). Landmarks below this threshold are treated as absent.

## Observation Events

Body-state observations are written to memory only on meaningful transitions:

| Event Type | Trigger |
|------------|---------|
| `pose_first_seen` | First time body pose is detected in a session |
| `pose_lost` | Body pose was visible but is no longer detected |
| `posture_change` | Posture label changed (e.g. upright -> slouched) |
| `engagement_change` | Engagement level changed (e.g. medium -> high) |

## Source Code

- Heuristics: `src/baymax/perception/heuristics.py`
- Pose interface: `src/baymax/perception/pose_interface.py`
- MediaPipe adapter: `src/baymax/perception/mediapipe_pose.py`
- Schemas: `src/baymax/schemas/perception.py`

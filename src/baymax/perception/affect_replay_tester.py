"""Replay verification system for affect analysis testing."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import cv2
import numpy as np

from baymax.perception.emotion import get_emotion_analyzer
from baymax.perception.emotion_smoother import AffectSmoother
from baymax.perception.affect_artifacts import AffectArtifactLogger
from baymax.schemas.perception import EmotionResult

logger = logging.getLogger(__name__)


class AffectReplayTester:
    """Replay-based affect analysis testing for verification."""

    def __init__(
        self,
        test_data_dir: str = "./test_data/affect_replay",
        artifact_dir: str = "./artifacts/verification_009/replay",
    ):
        """Initialize replay tester.

        Args:
            test_data_dir: Directory containing test images/clips
            artifact_dir: Directory to save verification artifacts
        """
        self.test_data_dir = Path(test_data_dir)
        self.artifact_dir = Path(artifact_dir)

        # Create directories
        self.test_data_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Initialized AffectReplayTester")

    def create_test_sequence(self, sequence_name: str, test_cases: list[dict[str, Any]]) -> str:
        """Create a test sequence definition file.

        Args:
            sequence_name: Name of the test sequence
            test_cases: List of test case definitions

        Returns:
            Path to created sequence file
        """
        sequence_file = self.test_data_dir / f"{sequence_name}.json"

        sequence_data = {
            "sequence_name": sequence_name,
            "created_at": datetime.utcnow().isoformat(),
            "test_cases": test_cases,
            "instructions": [
                "Place test images in the same directory as this file",
                "Images should show clear facial expressions for reliable analysis",
                "Use JPEG or PNG format, preferably 640x480 or higher resolution",
                "Face should occupy at least 15% of the image area",
                "Good lighting and front-facing poses work best"
            ]
        }

        try:
            with open(sequence_file, "w", encoding="utf-8") as f:
                json.dump(sequence_data, f, indent=2)
            logger.info("Created test sequence: %s", sequence_file)
            return str(sequence_file)
        except Exception as e:
            logger.error("Error creating test sequence: %s", e)
            return ""

    def run_replay_sequence(
        self,
        sequence_name: str,
        backend: str = "mediapipe",
        smoothing_alpha: float = 0.25,
    ) -> dict[str, Any]:
        """Run a replay test sequence.

        Args:
            sequence_name: Name of the test sequence to run
            backend: Emotion analysis backend to use
            smoothing_alpha: Smoothing parameter for affect state

        Returns:
            Dictionary with test results and verification data
        """
        sequence_file = self.test_data_dir / f"{sequence_name}.json"

        if not sequence_file.exists():
            logger.error("Sequence file not found: %s", sequence_file)
            return {"error": f"Sequence file not found: {sequence_file}"}

        try:
            with open(sequence_file, "r", encoding="utf-8") as f:
                sequence_data = json.load(f)
        except Exception as e:
            logger.error("Error loading sequence file: %s", e)
            return {"error": f"Error loading sequence: {e}"}

        # Initialize components
        try:
            emotion_analyzer = get_emotion_analyzer(backend)
            affect_smoother = AffectSmoother(alpha=smoothing_alpha)
            artifact_logger = AffectArtifactLogger(str(self.artifact_dir))
        except Exception as e:
            logger.error("Error initializing components: %s", e)
            return {"error": f"Component initialization failed: {e}"}

        # Start session tracking
        session_id = uuid4()
        artifact_logger.start_session(session_id)

        results = {
            "sequence_name": sequence_name,
            "session_id": str(session_id),
            "backend": backend,
            "test_started_at": datetime.utcnow().isoformat(),
            "test_cases": [],
            "errors": [],
        }

        # Process each test case
        for i, test_case in enumerate(sequence_data.get("test_cases", [])):
            case_result = self._run_single_test_case(
                test_case, i, emotion_analyzer, affect_smoother, artifact_logger
            )
            results["test_cases"].append(case_result)

            if case_result.get("error"):
                results["errors"].append(f"Test case {i}: {case_result['error']}")

        # Generate final artifacts
        try:
            artifact_files = artifact_logger.save_session_artifacts()
            manifest_file = artifact_logger.create_verification_manifest(sequence_name)

            results.update({
                "test_completed_at": datetime.utcnow().isoformat(),
                "artifact_files": artifact_files,
                "manifest_file": manifest_file,
                "timeline_summary": artifact_logger.get_timeline_summary(),
            })

        except Exception as e:
            logger.error("Error generating artifacts: %s", e)
            results["errors"].append(f"Artifact generation failed: {e}")

        # Save test report
        report_file = self.artifact_dir / f"replay_test_{sequence_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            results["report_file"] = str(report_file)
            logger.info("Replay test completed: %s", report_file)
        except Exception as e:
            logger.error("Error saving test report: %s", e)

        return results

    def _run_single_test_case(
        self,
        test_case: dict[str, Any],
        case_index: int,
        emotion_analyzer,
        affect_smoother: AffectSmoother,
        artifact_logger: AffectArtifactLogger,
    ) -> dict[str, Any]:
        """Run a single test case.

        Args:
            test_case: Test case definition
            case_index: Index of the test case
            emotion_analyzer: Emotion analysis backend
            affect_smoother: Affect smoother instance
            artifact_logger: Artifact logger instance

        Returns:
            Test case result dictionary
        """
        case_result = {
            "case_index": case_index,
            "name": test_case.get("name", f"case_{case_index}"),
            "image_file": test_case.get("image_file", ""),
            "expected_valence_range": test_case.get("expected_valence_range"),
            "expected_arousal_range": test_case.get("expected_arousal_range"),
            "expected_behavior": test_case.get("expected_behavior", ""),
        }

        image_path = self.test_data_dir / test_case.get("image_file", "")

        if not image_path.exists():
            case_result["error"] = f"Image file not found: {image_path}"
            return case_result

        try:
            # Load and process image
            image = cv2.imread(str(image_path))
            if image is None:
                case_result["error"] = f"Failed to load image: {image_path}"
                return case_result

            # Create mock face bounding box (assume face occupies center 60% of image)
            h, w = image.shape[:2]
            margin_w, margin_h = w * 0.2, h * 0.2
            face_bbox = {
                "x1": margin_w,
                "y1": margin_h,
                "x2": w - margin_w,
                "y2": h - margin_h,
            }

            # Analyze emotion
            emotion_result = emotion_analyzer.analyze(image, face_bbox)
            case_result["emotion_result"] = {
                "backend": emotion_result.backend,
                "success": emotion_result.success,
                "confidence": emotion_result.confidence,
                "valence": emotion_result.valence,
                "arousal": emotion_result.arousal,
                "dominant_emotion_soft_label": emotion_result.dominant_emotion_soft_label,
                "processing_time_ms": emotion_result.evidence.processing_time_ms,
            }

            # Update smoother
            smoothed_state = affect_smoother.update(emotion_result)
            case_result["smoothed_state"] = {
                "valence": smoothed_state.valence,
                "arousal": smoothed_state.arousal,
                "confidence": smoothed_state.confidence,
                "stable_duration_sec": smoothed_state.stable_duration_sec,
                "sample_count": smoothed_state.sample_count,
            }

            # Log to timeline
            artifact_logger.log_emotion_result(emotion_result, smoothed_state)

            # Verify results
            case_result["verification"] = self._verify_test_case_result(
                case_result, test_case
            )

            logger.debug(
                "Processed test case %d: valence=%.3f, arousal=%.3f, confidence=%.3f",
                case_index, emotion_result.valence, emotion_result.arousal, emotion_result.confidence
            )

        except Exception as e:
            logger.error("Error processing test case %d: %s", case_index, e)
            case_result["error"] = str(e)

        return case_result

    def _verify_test_case_result(
        self,
        case_result: dict[str, Any],
        test_case: dict[str, Any],
    ) -> dict[str, Any]:
        """Verify test case result against expected outcomes.

        Args:
            case_result: Actual test case result
            test_case: Expected test case definition

        Returns:
            Verification result dictionary
        """
        verification = {
            "passed": True,
            "issues": [],
        }

        emotion_result = case_result.get("emotion_result", {})

        # Check if analysis succeeded
        if not emotion_result.get("success", False):
            verification["passed"] = False
            verification["issues"].append("Emotion analysis failed")

        # Check confidence
        confidence = emotion_result.get("confidence", 0.0)
        if confidence < 0.1:  # Very low confidence threshold for testing
            verification["issues"].append(f"Low confidence: {confidence:.3f}")

        # Check valence range
        expected_valence = test_case.get("expected_valence_range")
        if expected_valence:
            actual_valence = emotion_result.get("valence", 0.0)
            if not (expected_valence[0] <= actual_valence <= expected_valence[1]):
                verification["passed"] = False
                verification["issues"].append(
                    f"Valence {actual_valence:.3f} not in expected range {expected_valence}"
                )

        # Check arousal range
        expected_arousal = test_case.get("expected_arousal_range")
        if expected_arousal:
            actual_arousal = emotion_result.get("arousal", 0.0)
            if not (expected_arousal[0] <= actual_arousal <= expected_arousal[1]):
                verification["passed"] = False
                verification["issues"].append(
                    f"Arousal {actual_arousal:.3f} not in expected range {expected_arousal}"
                )

        verification["summary"] = (
            "PASSED" if verification["passed"] else f"FAILED ({len(verification['issues'])} issues)"
        )

        return verification

    def create_standard_test_sequences(self) -> list[str]:
        """Create standard test sequences for affect verification.

        Returns:
            List of created sequence file paths
        """
        sequences = []

        # Positive expression test
        positive_sequence = self.create_test_sequence("positive_expression", [
            {
                "name": "smile_high_confidence",
                "image_file": "smile_clear.jpg",
                "description": "Clear genuine smile, good lighting",
                "expected_valence_range": [0.2, 1.0],
                "expected_arousal_range": [0.3, 1.0],
                "expected_behavior": "System should be naturally more engaging and warm"
            },
            {
                "name": "smile_moderate",
                "image_file": "smile_moderate.jpg",
                "description": "Moderate smile, front-facing",
                "expected_valence_range": [0.1, 1.0],
                "expected_arousal_range": [0.2, 1.0],
                "expected_behavior": "System should maintain upbeat but calm tone"
            }
        ])
        if positive_sequence:
            sequences.append(positive_sequence)

        # Neutral expression test
        neutral_sequence = self.create_test_sequence("neutral_expression", [
            {
                "name": "neutral_clear",
                "image_file": "neutral_clear.jpg",
                "description": "Neutral expression, good lighting",
                "expected_valence_range": [-0.2, 0.2],
                "expected_arousal_range": [0.0, 0.6],
                "expected_behavior": "System should use default supportive approach"
            }
        ])
        if neutral_sequence:
            sequences.append(neutral_sequence)

        # Low-affect expression test
        low_affect_sequence = self.create_test_sequence("low_affect_expression", [
            {
                "name": "subdued_thoughtful",
                "image_file": "subdued_clear.jpg",
                "description": "Subdued or thoughtful expression",
                "expected_valence_range": [-1.0, -0.1],
                "expected_arousal_range": [0.0, 0.4],
                "expected_behavior": "System should use gentler, more validating tone"
            }
        ])
        if low_affect_sequence:
            sequences.append(low_affect_sequence)

        # Occlusion/low confidence test
        occlusion_sequence = self.create_test_sequence("occlusion_test", [
            {
                "name": "face_turned_away",
                "image_file": "turned_away.jpg",
                "description": "Face partially turned away or occluded",
                "expected_valence_range": None,  # Don't check ranges for low-confidence cases
                "expected_arousal_range": None,
                "expected_behavior": "System should remain uncertain, not make hard guesses"
            },
            {
                "name": "poor_lighting",
                "image_file": "poor_lighting.jpg",
                "description": "Poor lighting or blurry image",
                "expected_valence_range": None,
                "expected_arousal_range": None,
                "expected_behavior": "System should remain uncertain due to low quality"
            }
        ])
        if occlusion_sequence:
            sequences.append(occlusion_sequence)

        logger.info("Created %d standard test sequences", len(sequences))
        return sequences
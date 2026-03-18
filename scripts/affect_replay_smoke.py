#!/usr/bin/env python3
"""Smoke test for affect analysis runtime wiring.

This script verifies that the affect analyzer is properly wired
into the live runtime by processing test images and checking
that status fields are populated.
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)


def create_test_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a simple test frame (blank with gradient for testing)."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Add some variation
    for i in range(height):
        frame[i, :, :] = int(255 * (i / height))
    return frame


async def run_smoke_test(
    artifact_dir: str,
    backend: str = "mediapipe",
    num_frames: int = 30,
) -> dict:
    """Run a smoke test verifying affect runtime wiring."""
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "backend": backend,
        "num_frames": num_frames,
        "passed": False,
        "checks": {},
        "status_samples": [],
        "errors": [],
    }

    # Create artifact directory
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)

    try:
        # Import runtime components
        from baymax.config.settings import get_settings
        from baymax.live.runtime import LiveRuntime
        from baymax.live.schemas import LiveRuntimeStatus
        from baymax.perception.emotion import get_emotion_analyzer
        from baymax.perception.emotion_smoother import AffectSmoother

        settings = get_settings()

        # Check 1: Settings expose affect configuration
        results["checks"]["settings_affect_enabled"] = settings.enable_affect
        results["checks"]["settings_affect_backend"] = settings.affect_backend
        results["checks"]["settings_sample_every_n"] = settings.affect_sample_every_n_frames

        # Check 2: Emotion analyzer can be instantiated
        try:
            analyzer = get_emotion_analyzer(backend=backend)
            results["checks"]["analyzer_instantiated"] = True
            results["checks"]["analyzer_name"] = analyzer.name()
            results["checks"]["analyzer_available"] = analyzer.is_available()
        except Exception as e:
            results["checks"]["analyzer_instantiated"] = False
            results["errors"].append(f"Analyzer instantiation failed: {e}")

        # Check 3: Smoother can be instantiated
        try:
            _smoother = AffectSmoother(
                alpha=settings.affect_smoothing_alpha,
                confidence_threshold=settings.affect_confidence_threshold,
            )
            _ = _smoother  # Mark as used
            results["checks"]["smoother_instantiated"] = True
        except Exception as e:
            results["checks"]["smoother_instantiated"] = False
            results["errors"].append(f"Smoother instantiation failed: {e}")

        # Check 4: Runtime initialization includes affect components
        try:
            runtime = LiveRuntime(db_path=":memory:")
            results["checks"]["runtime_affect_enabled"] = runtime._affect_enabled
            results["checks"]["runtime_analyzer_initialized"] = runtime._affect_analyzer is not None
            results["checks"]["runtime_smoother_initialized"] = runtime._affect_smoother is not None
            if runtime._affect_analyzer:
                results["checks"]["runtime_analyzer_backend"] = runtime._affect_analyzer.name()
        except Exception as e:
            results["errors"].append(f"Runtime initialization failed: {e}")
            results["checks"]["runtime_affect_enabled"] = False

        # Check 5: Status schema has affect fields
        status = LiveRuntimeStatus()
        affect_fields = [
            "affect_enabled",
            "affect_backend",
            "emotion_valence",
            "emotion_arousal",
            "emotion_confidence",
            "emotion_stable_duration_sec",
            "emotion_debug_summary",
        ]
        results["checks"]["status_fields_present"] = all(
            hasattr(status, field) for field in affect_fields
        )

        # Check 6: Test affect analysis on a synthetic face region
        # Note: This will return low confidence since there's no real face
        if results["checks"].get("analyzer_instantiated"):
            try:
                test_frame = create_test_frame()
                fake_bbox = {"x1": 100, "y1": 100, "x2": 300, "y2": 300}
                emotion_result = analyzer.analyze(test_frame, fake_bbox)
                results["checks"]["analyze_returns_result"] = emotion_result is not None
                results["checks"]["result_has_valence"] = hasattr(emotion_result, "valence")
                results["checks"]["result_has_arousal"] = hasattr(emotion_result, "arousal")
                results["checks"]["result_has_confidence"] = hasattr(emotion_result, "confidence")
                results["status_samples"].append({
                    "frame": "synthetic",
                    "valence": emotion_result.valence,
                    "arousal": emotion_result.arousal,
                    "confidence": emotion_result.confidence,
                    "backend": emotion_result.backend,
                })
            except Exception as e:
                results["errors"].append(f"Analysis test failed: {e}")

        # Determine overall pass
        critical_checks = [
            "settings_affect_enabled",
            "analyzer_instantiated",
            "smoother_instantiated",
            "runtime_affect_enabled",
            "runtime_analyzer_initialized",
            "runtime_smoother_initialized",
            "status_fields_present",
        ]
        results["passed"] = all(
            results["checks"].get(check, False) for check in critical_checks
        )

        # Summary
        total_checks = len(results["checks"])
        passed_checks = sum(1 for v in results["checks"].values() if v not in [False, None])
        results["summary"] = {
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "error_count": len(results["errors"]),
            "overall": "PASS" if results["passed"] else "FAIL",
        }

    except Exception as e:
        results["errors"].append(f"Smoke test failed: {e}")
        results["passed"] = False

    # Save results
    results_file = artifact_path / "smoke_test_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    results["artifact_files"] = [str(results_file)]

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Smoke test for affect analysis runtime wiring"
    )
    parser.add_argument(
        "--artifact-dir",
        default="./artifacts/verification_0091/affect",
        help="Directory to save artifacts",
    )
    parser.add_argument(
        "--backend",
        default="mediapipe",
        choices=["null", "mediapipe"],
        help="Emotion analysis backend to test",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("=" * 60)
    print("Bay-Max Affect Analysis Runtime Smoke Test")
    print("=" * 60)
    print(f"Backend: {args.backend}")
    print(f"Artifact directory: {args.artifact_dir}")
    print()

    # Run the smoke test
    results = asyncio.run(run_smoke_test(
        artifact_dir=args.artifact_dir,
        backend=args.backend,
    ))

    # Print results
    print("Check Results:")
    print("-" * 40)
    for check, value in results["checks"].items():
        status = "PASS" if value not in [False, None] else "FAIL"
        print(f"  {check}: {status} ({value})")

    if results["errors"]:
        print()
        print("Errors:")
        for error in results["errors"]:
            print(f"  - {error}")

    if results["status_samples"]:
        print()
        print("Status Samples:")
        for sample in results["status_samples"]:
            print(f"  Frame: {sample['frame']}")
            print(f"    Valence: {sample['valence']:.3f}")
            print(f"    Arousal: {sample['arousal']:.3f}")
            print(f"    Confidence: {sample['confidence']:.3f}")
            print(f"    Backend: {sample['backend']}")

    print()
    print(f"Summary: {results['summary']}")
    print()

    if results["artifact_files"]:
        print("Artifact files:")
        for f in results["artifact_files"]:
            print(f"  - {f}")

    print()
    overall = results["summary"]["overall"]
    if overall == "PASS":
        print("SMOKE TEST PASSED")
        return 0
    else:
        print("SMOKE TEST FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())

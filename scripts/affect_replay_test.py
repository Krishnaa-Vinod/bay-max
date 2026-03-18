#!/usr/bin/env python3
"""Helper script for running affect replay verification tests."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from baymax.perception.affect_replay_tester import AffectReplayTester


def main():
    """Main entry point for replay testing."""
    parser = argparse.ArgumentParser(description="Run affect analysis replay verification tests")
    parser.add_argument(
        "command",
        choices=["create-sequences", "run-test", "list-sequences"],
        help="Command to execute"
    )
    parser.add_argument(
        "--sequence",
        help="Name of test sequence to run (for run-test command)"
    )
    parser.add_argument(
        "--backend",
        default="mediapipe",
        choices=["null", "mediapipe"],
        help="Emotion analysis backend to use (default: mediapipe)"
    )
    parser.add_argument(
        "--test-data-dir",
        default="./test_data/affect_replay",
        help="Directory containing test data (default: ./test_data/affect_replay)"
    )
    parser.add_argument(
        "--artifact-dir",
        default="./artifacts/verification_009/replay",
        help="Directory to save artifacts (default: ./artifacts/verification_009/replay)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Initialize tester
    tester = AffectReplayTester(
        test_data_dir=args.test_data_dir,
        artifact_dir=args.artifact_dir
    )

    if args.command == "create-sequences":
        print("Creating standard test sequences...")
        sequences = tester.create_standard_test_sequences()
        print(f"Created {len(sequences)} test sequences:")
        for seq_path in sequences:
            print(f"  - {seq_path}")
        print("\nNext steps:")
        print("1. Add test images to the test_data/affect_replay directory")
        print("2. Images should match the filenames specified in the sequence files")
        print("3. Run tests with: python scripts/affect_replay_test.py run-test --sequence <name>")

    elif args.command == "list-sequences":
        test_data_dir = Path(args.test_data_dir)
        sequence_files = list(test_data_dir.glob("*.json"))

        if not sequence_files:
            print("No test sequences found.")
            print("Run 'create-sequences' command first to create standard sequences.")
        else:
            print("Available test sequences:")
            for seq_file in sequence_files:
                seq_name = seq_file.stem
                print(f"  - {seq_name}")

    elif args.command == "run-test":
        if not args.sequence:
            print("Error: --sequence argument is required for run-test command")
            sys.exit(1)

        print(f"Running replay test: {args.sequence}")
        print(f"Backend: {args.backend}")
        print(f"Test data directory: {args.test_data_dir}")
        print(f"Artifact directory: {args.artifact_dir}")
        print("=" * 50)

        try:
            results = tester.run_replay_sequence(
                sequence_name=args.sequence,
                backend=args.backend
            )

            if "error" in results:
                print(f"Error: {results['error']}")
                sys.exit(1)

            # Print summary
            print("\nTest Results Summary:")
            print(f"Sequence: {results['sequence_name']}")
            print(f"Session ID: {results['session_id']}")
            print(f"Backend: {results['backend']}")

            test_cases = results.get("test_cases", [])
            passed_cases = sum(1 for case in test_cases if case.get("verification", {}).get("passed", False))

            print(f"Test cases: {len(test_cases)} total, {passed_cases} passed")

            if results.get("errors"):
                print(f"Errors: {len(results['errors'])}")
                for error in results["errors"]:
                    print(f"  - {error}")

            # Print individual test case results
            print("\nDetailed Results:")
            for case in test_cases:
                case_name = case.get("name", "unknown")
                verification = case.get("verification", {})
                status = verification.get("summary", "UNKNOWN")

                print(f"  {case_name}: {status}")

                if verification.get("issues"):
                    for issue in verification["issues"]:
                        print(f"    - {issue}")

                # Show emotion analysis results
                emotion = case.get("emotion_result", {})
                if emotion:
                    print(f"    Valence: {emotion.get('valence', 0):.3f}, "
                          f"Arousal: {emotion.get('arousal', 0):.3f}, "
                          f"Confidence: {emotion.get('confidence', 0):.3f}")

            # Print artifact information
            timeline_summary = results.get("timeline_summary", {})
            if timeline_summary:
                print("\nTimeline Summary:")
                print(f"  Points: {timeline_summary.get('timeline_points', 0)}")
                print(f"  High confidence: {timeline_summary.get('high_confidence_points', 0)}")

                if "mean_valence" in timeline_summary:
                    print(f"  Mean valence: {timeline_summary['mean_valence']:.3f}")
                    print(f"  Mean arousal: {timeline_summary['mean_arousal']:.3f}")

            # Print artifact files
            artifacts = results.get("artifact_files", [])
            if artifacts:
                print(f"\nGenerated Artifacts ({len(artifacts)} files):")
                for artifact in artifacts:
                    print(f"  - {artifact}")

            report_file = results.get("report_file")
            if report_file:
                print(f"\nFull test report: {report_file}")

            print("\nTest completed successfully!")

        except Exception as e:
            print(f"Error running test: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()

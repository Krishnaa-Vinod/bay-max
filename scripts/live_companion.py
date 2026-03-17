#!/usr/bin/env python3
"""Bay-Max Live Companion Runner.

Usage:
    # Webcam mode (requires display):
    python scripts/live_companion.py

    # Video replay mode (headless-friendly):
    python scripts/live_companion.py --replay path/to/video.mp4

    # Folder replay mode:
    python scripts/live_companion.py --replay path/to/frames_dir/

    # Override settings:
    python scripts/live_companion.py --replay video.mp4 --max-frames 100 --no-overlay

Environment variables (see .env.example):
    BAYMAX_LIVE_SOURCE        - "webcam" or "replay"
    BAYMAX_PREVIEW_FPS        - Target frame rate (default: 8)
    BAYMAX_ANALYSIS_INTERVAL_SEC - Seconds between analyses (default: 2.0)
    BAYMAX_LIVE_ARTIFACT_DIR  - Artifact output directory
    BAYMAX_DIALOGUE_BACKEND   - "rule_based" | "transformers" | "ollama"
"""

import argparse
import asyncio
import logging
import os
import sys

# Ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main():
    parser = argparse.ArgumentParser(
        description="Bay-Max Live Companion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--replay",
        type=str,
        default=None,
        help="Path to video file or folder of frames for replay mode",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Max frames to process (0 = unlimited)",
    )
    parser.add_argument(
        "--max-run-sec",
        type=float,
        default=0,
        help="Max run time in seconds (0 = unlimited)",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Disable the live overlay",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug overlay and verbose logging",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default=None,
        help="Override artifact output directory",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="baymax_live.db",
        help="SQLite database path",
    )
    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Override settings via env vars if CLI args provided
    if args.replay:
        os.environ["BAYMAX_LIVE_VIDEO_REPLAY_PATH"] = args.replay
        os.environ["BAYMAX_LIVE_SOURCE"] = "replay"
    if args.no_overlay:
        os.environ["BAYMAX_ENABLE_LIVE_OVERLAY"] = "false"
    if args.debug:
        os.environ["BAYMAX_ENABLE_DIALOGUE_DEBUG"] = "true"
    if args.artifact_dir:
        os.environ["BAYMAX_LIVE_ARTIFACT_DIR"] = args.artifact_dir
    if args.max_run_sec > 0:
        os.environ["BAYMAX_LIVE_MAX_RUN_SEC"] = str(args.max_run_sec)

    from baymax.live.runtime import LiveRuntime

    runtime = LiveRuntime(db_path=args.db_path)

    print("=" * 60)
    print("  Bay-Max Live Companion")
    print("=" * 60)
    if args.replay:
        print(f"  Mode: Replay ({args.replay})")
    else:
        print("  Mode: Webcam")
    print(f"  Artifact dir: {runtime.artifact_logger.artifact_dir}")
    print(f"  Max frames: {args.max_frames or 'unlimited'}")
    print(f"  Overlay: {'disabled' if args.no_overlay else 'enabled'}")
    print("=" * 60)
    print()

    summary = asyncio.run(runtime.run(
        source=args.replay,
        max_frames=args.max_frames,
    ))

    print()
    print("=" * 60)
    print("  Run Complete")
    print("=" * 60)
    print(f"  Frames: {summary.total_frames}")
    print(f"  Analyses: {summary.total_analyses}")
    print(f"  Events: {summary.total_events}")
    print(f"  Responses: {summary.total_responses}")
    print(f"  Suppressions: {summary.total_suppressions}")
    print(f"  Sessions created: {summary.sessions_created}")
    print(f"  Sessions resumed: {summary.sessions_resumed}")
    print(f"  Users recognized: {', '.join(summary.users_recognized) or 'none'}")
    print(f"  Backend: {summary.backend}")
    print(f"  Artifact dir: {runtime.artifact_logger.artifact_dir}")
    if summary.errors:
        print(f"  Errors: {len(summary.errors)}")
        for e in summary.errors:
            print(f"    - {e}")
    print("=" * 60)


if __name__ == "__main__":
    main()

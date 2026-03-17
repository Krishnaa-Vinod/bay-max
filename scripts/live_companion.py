#!/usr/bin/env python3
"""Bay-Max Live Companion Runner.

Usage:
    # Webcam mode with speech input (requires mic + display):
    python scripts/live_companion.py

    # Webcam mode without speech input:
    python scripts/live_companion.py --no-speech-input

    # Video replay mode (headless-friendly):
    python scripts/live_companion.py --replay path/to/video.mp4

    # Push-to-talk mode instead of VAD:
    python scripts/live_companion.py --mic-mode push_to_talk

    # Override settings:
    python scripts/live_companion.py --replay video.mp4 --max-frames 100 --no-overlay

Environment variables (see .env.example):
    BAYMAX_LIVE_SOURCE        - "webcam" or "replay"
    BAYMAX_PREVIEW_FPS        - Target frame rate (default: 8)
    BAYMAX_ANALYSIS_INTERVAL_SEC - Seconds between analyses (default: 2.0)
    BAYMAX_LIVE_ARTIFACT_DIR  - Artifact output directory
    BAYMAX_DIALOGUE_BACKEND   - "rule_based" | "transformers" | "ollama"
    BAYMAX_ENABLE_SPEECH_INPUT - "true" | "false"
    BAYMAX_STT_BACKEND        - "faster_whisper" | "null"
    BAYMAX_MIC_MODE           - "vad" | "push_to_talk"
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
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable TTS speech output",
    )
    parser.add_argument(
        "--tts-backend",
        type=str,
        default=None,
        help="TTS backend: kokoro | piper | null",
    )
    parser.add_argument(
        "--no-playback",
        action="store_true",
        help="Disable audio playback (WAV files still generated)",
    )
    # Iteration 008: Speech input flags
    parser.add_argument(
        "--no-speech-input",
        action="store_true",
        help="Disable speech input (microphone capture + STT)",
    )
    parser.add_argument(
        "--stt-backend",
        type=str,
        default=None,
        help="STT backend: faster_whisper | null",
    )
    parser.add_argument(
        "--mic-mode",
        type=str,
        default=None,
        help="Microphone mode: vad | push_to_talk",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default=None,
        help="Whisper model size: base.en | base | small.en | small",
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
    if args.no_tts:
        os.environ["BAYMAX_TTS_ENABLED"] = "false"
    if args.tts_backend:
        os.environ["BAYMAX_TTS_BACKEND"] = args.tts_backend
    if args.no_playback:
        os.environ["BAYMAX_ENABLE_AUDIO_PLAYBACK"] = "false"
    # Speech input overrides
    if args.no_speech_input:
        os.environ["BAYMAX_ENABLE_SPEECH_INPUT"] = "false"
    if args.stt_backend:
        os.environ["BAYMAX_STT_BACKEND"] = args.stt_backend
    if args.mic_mode:
        os.environ["BAYMAX_MIC_MODE"] = args.mic_mode
    if args.whisper_model:
        os.environ["BAYMAX_WHISPER_MODEL"] = args.whisper_model

    from baymax.live.runtime import LiveRuntime

    runtime = LiveRuntime(db_path=args.db_path)

    stt_label = args.stt_backend or "faster_whisper"
    speech_input_status = "disabled" if args.no_speech_input else stt_label
    mic_mode = args.mic_mode or "vad"

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
    tts_status = "disabled" if args.no_tts else (args.tts_backend or "kokoro")
    print(f"  TTS: {tts_status}")
    print(f"  Playback: {'disabled' if args.no_playback else 'enabled'}")
    print(f"  Speech input: {speech_input_status}")
    print(f"  Mic mode: {mic_mode}")
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

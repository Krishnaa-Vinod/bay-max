#!/usr/bin/env python3
"""Bay-Max Iteration 007 Local Verification Script.

Runs a short webcam+TTS session and captures proof artifacts under
artifacts/verification_007/.

Usage:
    # Full webcam + TTS verification (30s):
    python scripts/verify_007.py

    # Replay mode (no webcam needed):
    python scripts/verify_007.py --replay path/to/frames/

    # TTS smoke test only (no webcam):
    python scripts/verify_007.py --tts-only

    # Skip playback (WAV generation only):
    python scripts/verify_007.py --no-playback
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def tts_smoke_test(artifact_dir: str, no_playback: bool = False) -> dict:
    """Run a quick TTS smoke test and save a WAV artifact."""
    from baymax.config.settings import get_settings
    from baymax.tts.provider import get_tts_provider
    from baymax.tts.schemas import TTSSmokeTestResult

    settings = get_settings()
    backend = settings.tts_backend
    voice = settings.tts_voice

    print(f"  TTS backend: {backend}")
    print(f"  TTS voice: {voice}")

    provider = get_tts_provider(
        backend=backend,
        voice=voice,
        device=settings.tts_device,
        model_dir=settings.model_dir,
        sample_rate=settings.tts_rate,
    )

    test_text = "Hello! I am Bay-Max, your personal companion."
    wav_path = os.path.join(artifact_dir, "tts_smoke_test.wav")

    start = time.time()
    result = provider.synthesize(test_text, wav_path)
    elapsed = time.time() - start

    smoke = TTSSmokeTestResult(
        backend=backend,
        voice=voice,
        success=result.success,
        wav_path=result.wav_path,
        duration_sec=round(result.duration_sec, 2),
        sample_rate=result.sample_rate,
        error=result.error,
    )

    smoke_dict = smoke.model_dump(mode="json")
    smoke_dict["synthesis_time_sec"] = round(elapsed, 2)

    if result.success:
        print(f"  WAV generated: {result.wav_path}")
        print(f"  Audio duration: {result.duration_sec:.2f}s")
        print(f"  Synthesis time: {elapsed:.2f}s")

        if not no_playback:
            try:
                import sounddevice as sd
                import soundfile as sf

                data, sr = sf.read(result.wav_path)
                print("  Playing audio...")
                sd.play(data, sr)
                sd.wait()
                print("  Playback complete.")
                smoke_dict["playback_tested"] = True
            except ImportError:
                print("  sounddevice not available — skipping playback.")
                smoke_dict["playback_tested"] = False
            except OSError as e:
                print(f"  Audio device error: {e} — skipping playback.")
                smoke_dict["playback_tested"] = False
        else:
            smoke_dict["playback_tested"] = False
    else:
        print(f"  TTS failed: {result.error}")
        smoke_dict["playback_tested"] = False

    return smoke_dict


def main():
    parser = argparse.ArgumentParser(
        description="Bay-Max Iteration 007 Verification",
    )
    parser.add_argument(
        "--replay",
        type=str,
        default=None,
        help="Replay from video/folder instead of webcam",
    )
    parser.add_argument(
        "--tts-only",
        action="store_true",
        help="Only run TTS smoke test (no webcam)",
    )
    parser.add_argument(
        "--no-playback",
        action="store_true",
        help="Skip audio playback",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Max run duration in seconds (default 30)",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default="./artifacts/verification_007",
        help="Verification artifact directory",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    artifact_dir = args.artifact_dir
    os.makedirs(artifact_dir, exist_ok=True)

    print("=" * 60)
    print("  Bay-Max Iteration 007 Verification")
    print("=" * 60)

    results = {
        "verification_id": datetime.now(UTC).strftime(
            "%Y%m%d_%H%M%S"
        ),
        "started_at": datetime.now(UTC).isoformat(),
        "tts_smoke_test": None,
        "live_run": None,
        "artifact_dir": artifact_dir,
    }

    # 1. TTS Smoke Test
    print()
    print("--- TTS Smoke Test ---")
    smoke_result = tts_smoke_test(
        artifact_dir, no_playback=args.no_playback
    )
    results["tts_smoke_test"] = smoke_result

    if args.tts_only:
        results["ended_at"] = datetime.now(UTC).isoformat()
        _write_artifacts(artifact_dir, results)
        print()
        print("TTS-only verification complete.")
        return

    # 2. Live run
    print()
    print("--- Live Run ---")

    os.environ["BAYMAX_LIVE_ARTIFACT_DIR"] = artifact_dir
    os.environ["BAYMAX_LIVE_MAX_RUN_SEC"] = str(args.duration)
    if args.no_playback:
        os.environ["BAYMAX_ENABLE_AUDIO_PLAYBACK"] = "false"

    from baymax.live.runtime import LiveRuntime

    runtime = LiveRuntime(db_path="baymax_verify_007.db")

    mode = "replay" if args.replay else "webcam"
    print(f"  Mode: {mode}")
    print(f"  Duration: {args.duration}s")

    summary = asyncio.run(runtime.run(
        source=args.replay,
        max_frames=0,
    ))

    results["live_run"] = {
        "source": mode,
        "total_frames": summary.total_frames,
        "total_analyses": summary.total_analyses,
        "total_events": summary.total_events,
        "total_responses": summary.total_responses,
        "sessions_created": summary.sessions_created,
        "users_recognized": summary.users_recognized,
        "errors": summary.errors,
    }

    results["ended_at"] = datetime.now(UTC).isoformat()
    _write_artifacts(artifact_dir, results)

    print()
    print("=" * 60)
    print("  Verification Complete")
    print("=" * 60)
    print(f"  Artifacts: {artifact_dir}")
    print(f"  TTS: {'passed' if smoke_result.get('success') else 'failed'}")
    print(f"  Frames: {summary.total_frames}")
    print(f"  Events: {summary.total_events}")
    print(f"  Responses: {summary.total_responses}")
    print("=" * 60)


def _write_artifacts(artifact_dir: str, results: dict) -> None:
    """Write manifest.json and summary.md for verification."""
    manifest_path = os.path.join(artifact_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    summary_path = os.path.join(artifact_dir, "summary.md")
    tts = results.get("tts_smoke_test") or {}
    live = results.get("live_run") or {}

    lines = [
        "# Bay-Max Iteration 007 Verification Summary",
        "",
        f"**Verification ID**: {results.get('verification_id', 'N/A')}",
        f"**Started**: {results.get('started_at', 'N/A')}",
        f"**Ended**: {results.get('ended_at', 'N/A')}",
        "",
        "## TTS Smoke Test",
        f"- Backend: {tts.get('backend', 'N/A')}",
        f"- Voice: {tts.get('voice', 'N/A')}",
        f"- Success: {tts.get('success', False)}",
        f"- WAV: {tts.get('wav_path', 'N/A')}",
        f"- Duration: {tts.get('duration_sec', 0)}s",
        f"- Playback tested: {tts.get('playback_tested', False)}",
    ]

    if live:
        lines.extend([
            "",
            "## Live Run",
            f"- Source: {live.get('source', 'N/A')}",
            f"- Frames: {live.get('total_frames', 0)}",
            f"- Analyses: {live.get('total_analyses', 0)}",
            f"- Events: {live.get('total_events', 0)}",
            f"- Responses: {live.get('total_responses', 0)}",
            f"- Sessions: {live.get('sessions_created', 0)}",
            f"- Users: {', '.join(live.get('users_recognized', [])) or 'none'}",
        ])
        if live.get("errors"):
            lines.append("")
            lines.append("### Errors")
            for e in live["errors"]:
                lines.append(f"- {e}")

    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

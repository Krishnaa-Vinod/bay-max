#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Single-process local backend launcher for Bay-Max.

This launcher runs FastAPI and the live runtime in one asyncio process/thread so
runtime-bound state and SQLite access stay coherent for local development.
"""

import argparse
import asyncio
import importlib.util
import logging
import os
import signal
import sys
import urllib.error
import urllib.request
from contextlib import suppress

# Ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bay-Max single-process local backend",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1", help="FastAPI host")
    parser.add_argument("--port", type=int, default=8000, help="FastAPI port")
    parser.add_argument(
        "--mode",
        choices=["basic", "voice", "full"],
        default="full",
        help="Runtime mode: basic(no speech), voice(TTS only), full(TTS + speech input)",
    )
    parser.add_argument("--replay", type=str, default=None, help="Replay video/folder path")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to process (0 = unlimited)")
    parser.add_argument("--max-run-sec", type=float, default=0.0, help="Max runtime in seconds (0 = unlimited)")
    parser.add_argument("--db-path", type=str, default="baymax.db", help="SQLite db path")
    parser.add_argument("--no-overlay", action="store_true", help="Disable overlay rendering")
    parser.add_argument("--no-speech-input", action="store_true", help="Disable speech input")
    parser.add_argument("--stt-backend", type=str, default=None, help="STT backend override")
    parser.add_argument("--mic-mode", type=str, default=None, help="Mic mode override")
    parser.add_argument("--tts-backend", type=str, default=None, help="TTS backend override")
    parser.add_argument("--no-tts", action="store_true", help="Disable TTS output")
    parser.add_argument("--no-playback", action="store_true", help="Disable audio playback")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"], help="Logging level")
    return parser.parse_args()


def apply_env_overrides(args: argparse.Namespace) -> None:
    _apply_mode_overrides(args.mode)

    if args.replay:
        os.environ["BAYMAX_LIVE_VIDEO_REPLAY_PATH"] = args.replay
        os.environ["BAYMAX_LIVE_SOURCE"] = "replay"
    if args.max_run_sec > 0:
        os.environ["BAYMAX_LIVE_MAX_RUN_SEC"] = str(args.max_run_sec)
    if args.no_overlay:
        os.environ["BAYMAX_ENABLE_LIVE_OVERLAY"] = "false"
    if args.no_speech_input:
        os.environ["BAYMAX_ENABLE_SPEECH_INPUT"] = "false"
    if args.stt_backend:
        os.environ["BAYMAX_STT_BACKEND"] = args.stt_backend
    if args.mic_mode:
        os.environ["BAYMAX_MIC_MODE"] = args.mic_mode
    if args.tts_backend:
        os.environ["BAYMAX_TTS_BACKEND"] = args.tts_backend
    if args.no_tts:
        os.environ["BAYMAX_TTS_ENABLED"] = "false"
    if args.no_playback:
        os.environ["BAYMAX_ENABLE_AUDIO_PLAYBACK"] = "false"


def _apply_mode_overrides(mode: str) -> None:
    """Apply recommended local mode env defaults.

    Modes:
      - basic: no speech input, no TTS playback/synthesis
      - voice: TTS enabled, speech input disabled
      - full: TTS + speech input enabled (if dependencies are available)
    """
    if mode == "basic":
        os.environ["BAYMAX_TTS_ENABLED"] = "false"
        os.environ["BAYMAX_ENABLE_SPEECH_INPUT"] = "false"
        return

    if mode == "voice":
        os.environ["BAYMAX_TTS_ENABLED"] = "true"
        os.environ["BAYMAX_ENABLE_SPEECH_INPUT"] = "false"
        return

    # full
    os.environ["BAYMAX_TTS_ENABLED"] = "true"
    os.environ["BAYMAX_ENABLE_SPEECH_INPUT"] = "true"
    _select_dialogue_backend_for_full_mode()


def _ollama_is_available(base_url: str) -> bool:
    tags_url = base_url.rstrip("/") + "/api/tags"
    req = urllib.request.Request(tags_url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            return 200 <= getattr(resp, "status", 0) < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _select_dialogue_backend_for_full_mode() -> None:
    """Prefer grounded dialogue backends for full local mode.

    Priority:
      1) Ollama (if server reachable)
      2) Transformers (if explicitly configured and package is installed)
      3) Rule-based fallback
    """
    explicit = os.getenv("BAYMAX_DIALOGUE_BACKEND", "").strip().lower()
    lock_backend = os.getenv("BAYMAX_DIALOGUE_BACKEND_LOCKED", "").strip().lower() in {
        "1", "true", "yes", "on"
    }

    # Respect explicit non-rule-based backend selection. Allow stale rule_based
    # env pins to be auto-upgraded unless backend locking is requested.
    if explicit in {"ollama", "transformers"}:
        return
    if explicit == "rule_based" and lock_backend:
        return

    ollama_base = os.getenv("BAYMAX_OLLAMA_BASE_URL", "http://localhost:11434")
    if _ollama_is_available(ollama_base):
        os.environ["BAYMAX_DIALOGUE_BACKEND"] = "ollama"
        if not os.getenv("BAYMAX_OLLAMA_MODEL", "").strip():
            os.environ["BAYMAX_OLLAMA_MODEL"] = "qwen2.5:1.5b"
        return

    hf_model = os.getenv("BAYMAX_HF_CHAT_MODEL", "Qwen/Qwen2.5-1.5B-Instruct").strip()
    has_transformers = importlib.util.find_spec("transformers") is not None
    if hf_model and has_transformers:
        os.environ["BAYMAX_DIALOGUE_BACKEND"] = "transformers"
        return

    os.environ["BAYMAX_DIALOGUE_BACKEND"] = "rule_based"


async def run_local_backend(args: argparse.Namespace) -> int:
    import uvicorn

    from apps.api.main import app, set_live_runtime
    from baymax.live.runtime import LiveRuntime

    runtime = LiveRuntime(db_path=args.db_path)
    set_live_runtime(runtime)

    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        access_log=False,
    )
    server = uvicorn.Server(config)

    runtime_task = asyncio.create_task(
        runtime.run(source=args.replay, max_frames=args.max_frames),
        name="baymax-runtime",
    )
    server_task = asyncio.create_task(server.serve(), name="baymax-api")

    stop_event = asyncio.Event()
    stop_task = asyncio.create_task(stop_event.wait(), name="baymax-stop-wait")

    def request_stop() -> None:
        runtime._request_stop()
        server.should_exit = True
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(sig, request_stop)

    done, pending = await asyncio.wait(
        {runtime_task, server_task, stop_task},
        return_when=asyncio.FIRST_COMPLETED,
    )

    if runtime_task in done:
        server.should_exit = True
    if server_task in done:
        runtime._request_stop()

    for task in pending:
        if isinstance(task, asyncio.Task):
            task.cancel()

    with suppress(asyncio.CancelledError):
        await server_task

    with suppress(asyncio.CancelledError):
        summary = await runtime_task
        print("=" * 60)
        print("Local backend runtime summary")
        print("=" * 60)
        print(f"Frames: {summary.total_frames}")
        print(f"Analyses: {summary.total_analyses}")
        print(f"Sessions created: {summary.sessions_created}")
        print(f"Sessions resumed: {summary.sessions_resumed}")
        if summary.errors:
            print(f"Errors: {len(summary.errors)}")
            for err in summary.errors:
                print(f"- {err}")

    return 0


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    apply_env_overrides(args)
    logging.getLogger(__name__).info(
        "Local mode=%s dialogue_backend=%s tts_enabled=%s speech_input=%s",
        args.mode,
        os.getenv("BAYMAX_DIALOGUE_BACKEND", "rule_based"),
        os.getenv("BAYMAX_TTS_ENABLED", "true"),
        os.getenv("BAYMAX_ENABLE_SPEECH_INPUT", "true"),
    )
    raise SystemExit(asyncio.run(run_local_backend(args)))


if __name__ == "__main__":
    main()

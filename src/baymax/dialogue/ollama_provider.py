"""Ollama-compatible dialogue backend.

Connects to a locally running Ollama server and generates supportive responses
using any model available through the Ollama API.

Setup:
    1. Install Ollama: https://ollama.ai
    2. Pull a model: ollama pull qwen2.5:1.5b
    3. Set BAYMAX_OLLAMA_BASE_URL and BAYMAX_OLLAMA_MODEL in .env

This backend uses the Ollama /api/chat endpoint (OpenAI-compatible format).
"""

import json
import logging
import time

from baymax.core.enums import ResponseStrategy
from baymax.dialogue.interfaces import DialogueProvider
from baymax.dialogue.prompt_builder import build_messages
from baymax.dialogue.safety import check_output_safety
from baymax.schemas.memory import MemoryQueryResult
from baymax.schemas.response import (
    DialogueDebugTrace,
    GroundedPromptContext,
    SupportiveResponse,
)
from baymax.state.models import InteractionState

logger = logging.getLogger(__name__)


class OllamaDialogueProvider(DialogueProvider):
    """Dialogue provider backed by a locally running Ollama server.

    Uses the /api/chat endpoint with structured message format.
    Falls back gracefully if the server is unreachable.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "",
        temperature: float = 0.4,
        max_new_tokens: int = 220,
        enable_debug: bool = False,
        enable_safe_health_mode: bool = True,
    ) -> None:
        if not model:
            raise ValueError(
                "BAYMAX_OLLAMA_MODEL must be set to use the Ollama backend. "
                "Example: ollama pull qwen2.5:1.5b"
            )
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_new_tokens
        self._enable_debug = enable_debug
        self._safe_health_mode = enable_safe_health_mode
        self._available: bool | None = None  # None = not yet checked

    @property
    def backend_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    def is_available(self) -> bool:
        """Probe the Ollama server to check availability (cached)."""
        if self._available is not None:
            return self._available
        try:
            import urllib.request

            req = urllib.request.Request(
                f"{self._base_url}/api/tags",
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                self._available = resp.status == 200
        except Exception:
            self._available = False
            logger.warning("Ollama server not reachable at %s", self._base_url)
        return self._available

    def generate(
        self,
        strategy: ResponseStrategy,
        state: InteractionState,
        memories: MemoryQueryResult,
        prompt_context: GroundedPromptContext | None = None,
    ) -> SupportiveResponse:
        """Generate a response via the Ollama chat API."""
        t0 = time.time()

        if prompt_context is None:
            # Build a minimal context if none was passed
            from baymax.dialogue.prompt_builder import build_prompt_context

            prompt_context = build_prompt_context(
                strategy=strategy,
                state=state,
                memories=memories,
                recent_turns=[],
            )

        messages = build_messages(prompt_context)
        prompt_text = "\n".join(m["content"] for m in messages)
        prompt_tokens_approx = len(prompt_text.split())

        raw_output: str | None = None
        safety_flags: list[str] = []
        fallback_used = False
        text: str

        try:
            text = self._call_ollama(messages)
            raw_output = text

            # Safety check on LLM output
            if self._safe_health_mode:
                safety_decision = check_output_safety(text)
                if not safety_decision.is_safe:
                    safety_flags = safety_decision.flags
                    text = safety_decision.redirect_response or text
                    logger.warning(
                        "Ollama output failed safety check: %s", safety_flags
                    )

        except Exception as exc:
            logger.warning("Ollama generation failed: %s", exc, exc_info=True)
            # Surface the error upward so the orchestrator can trigger fallback
            raise

        latency_ms = (time.time() - t0) * 1000

        debug: DialogueDebugTrace | None = None
        if self._enable_debug:
            debug = DialogueDebugTrace(
                backend="ollama",
                model_name=self._model,
                prompt_tokens_approx=prompt_tokens_approx,
                latency_ms=round(latency_ms, 1),
                raw_output=raw_output,
                fallback_used=fallback_used,
                safety_flags=safety_flags,
            )

        return SupportiveResponse(
            session_id=state.session_id,
            user_id=state.user_id,
            strategy=strategy,
            message=text,
            backend="ollama",
            model_name=self._model,
            fallback_used=False,
            safety_flags=safety_flags,
            metadata={"debug": json.dumps(debug.model_dump(mode="json")) if debug else ""},
        )

    def _call_ollama(self, messages: list[dict]) -> str:
        """Make the HTTP call to the Ollama /api/chat endpoint."""
        import json as _json
        import urllib.error
        import urllib.request

        payload = _json.dumps(
            {
                "model": self._model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self._temperature,
                    "num_predict": self._max_tokens,
                },
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = _json.loads(resp.read().decode("utf-8"))
                # Ollama response: {"message": {"role": "assistant", "content": "..."}}
                content = body.get("message", {}).get("content", "")
                if not content:
                    raise ValueError(f"Empty content in Ollama response: {body}")
                return content.strip()
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"Ollama HTTP error {exc.code}: {exc.read().decode('utf-8', errors='replace')}"
            ) from exc

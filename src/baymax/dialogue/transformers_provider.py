"""HuggingFace Transformers dialogue backend.

Directly loads and runs a local chat model using the HuggingFace Transformers
pipeline. This backend does not require any server — the model runs in-process.

Recommended models (free, no gating required):
    Qwen/Qwen2.5-1.5B-Instruct  (default — good for laptop)
    Qwen/Qwen2.5-0.5B-Instruct  (lower-resource fallback)
    Qwen/Qwen2.5-7B-Instruct    (better on GPU)

Configuration (via .env):
    BAYMAX_DIALOGUE_BACKEND=transformers
    BAYMAX_HF_CHAT_MODEL=Qwen/Qwen2.5-1.5B-Instruct
    BAYMAX_HF_DTYPE=auto          # or float16, bfloat16, float32
    BAYMAX_DIALOGUE_TEMPERATURE=0.4
    BAYMAX_DIALOGUE_MAX_NEW_TOKENS=220
"""

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


class TransformersDialogueProvider(DialogueProvider):
    """Dialogue provider using a locally-loaded HuggingFace chat model.

    The model is lazy-loaded on first generate() call.
    Import errors are caught at init-time so the orchestrator can fall back
    gracefully if the 'transformers' package is not installed.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        dtype: str = "auto",
        temperature: float = 0.4,
        max_new_tokens: int = 220,
        enable_debug: bool = False,
        enable_safe_health_mode: bool = True,
        cache_dir: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._dtype_str = dtype
        self._temperature = temperature
        self._max_new_tokens = max_new_tokens
        self._enable_debug = enable_debug
        self._safe_health_mode = enable_safe_health_mode
        self._cache_dir = cache_dir

        self._pipeline = None
        self._load_error: str | None = None

        # Verify transformers is importable at construction time
        try:
            import transformers as _trans  # noqa: F401
        except ImportError as exc:
            self._load_error = (
                f"'transformers' package not installed. "
                f"Install it with: pip install transformers torch. "
                f"Original error: {exc}"
            )
            logger.warning(self._load_error)

    @property
    def backend_name(self) -> str:
        return "transformers"

    @property
    def model_name(self) -> str:
        return self._model_name

    def is_available(self) -> bool:
        return self._load_error is None

    def _ensure_pipeline(self) -> None:
        """Lazily load the model pipeline on first use."""
        if self._pipeline is not None:
            return
        if self._load_error is not None:
            raise RuntimeError(self._load_error)

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        logger.info(
            "Loading Transformers model %s (dtype=%s). This may take a moment...",
            self._model_name,
            self._dtype_str,
        )

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self._dtype_str, "auto")

        tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            cache_dir=self._cache_dir,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            cache_dir=self._cache_dir,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map="auto",
        )
        self._pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        logger.info("Transformers model %s loaded successfully", self._model_name)

    def generate(
        self,
        strategy: ResponseStrategy,
        state: InteractionState,
        memories: MemoryQueryResult,
        prompt_context: GroundedPromptContext | None = None,
    ) -> SupportiveResponse:
        """Generate a response using the locally-loaded chat model."""
        t0 = time.time()

        if prompt_context is None:
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
        text: str

        # Ensure the pipeline is loaded (may raise)
        self._ensure_pipeline()

        try:
            outputs = self._pipeline(
                messages,
                max_new_tokens=self._max_new_tokens,
                temperature=self._temperature,
                do_sample=self._temperature > 0,
                pad_token_id=self._pipeline.tokenizer.eos_token_id,
            )
            # Transformers chat pipeline returns [{generated_text: [{role, content}, ...]}]
            generated = outputs[0].get("generated_text", [])
            # The last message should be the assistant's reply
            if isinstance(generated, list) and generated:
                last = generated[-1]
                if isinstance(last, dict):
                    text = last.get("content", "").strip()
                else:
                    text = str(last).strip()
            else:
                text = str(generated).strip()

            raw_output = text

            if not text:
                raise ValueError("Empty output from Transformers pipeline")

            # Safety check
            if self._safe_health_mode:
                safety_decision = check_output_safety(text)
                if not safety_decision.is_safe:
                    safety_flags = safety_decision.flags
                    text = safety_decision.redirect_response or text

        except Exception as exc:
            logger.warning(
                "Transformers generation failed: %s", exc, exc_info=True
            )
            raise

        latency_ms = (time.time() - t0) * 1000

        debug: DialogueDebugTrace | None = None
        if self._enable_debug:
            debug = DialogueDebugTrace(
                backend="transformers",
                model_name=self._model_name,
                prompt_tokens_approx=prompt_tokens_approx,
                latency_ms=round(latency_ms, 1),
                raw_output=raw_output,
                fallback_used=False,
                safety_flags=safety_flags,
            )

        import json

        return SupportiveResponse(
            session_id=state.session_id,
            user_id=state.user_id,
            strategy=strategy,
            message=text,
            backend="transformers",
            model_name=self._model_name,
            fallback_used=False,
            safety_flags=safety_flags,
            metadata={"debug": json.dumps(debug.model_dump(mode="json")) if debug else ""},
        )

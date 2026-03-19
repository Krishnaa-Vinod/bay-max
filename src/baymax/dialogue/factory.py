"""Dialogue provider factory.

Instantiates the configured DialogueProvider based on settings.
Falls back to RuleBasedDialogue if the selected backend cannot be initialised.
"""

import importlib.util
import json
import logging
import urllib.error
import urllib.request

from baymax.config.settings import BaymaxSettings, get_settings
from baymax.dialogue.interfaces import DialogueProvider
from baymax.dialogue.rule_based import RuleBasedDialogue

logger = logging.getLogger(__name__)

AVAILABLE_BACKENDS: list[str] = ["auto_local", "rule_based", "ollama", "transformers"]


def _probe_ollama_models(base_url: str) -> list[str]:
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/tags",
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            if getattr(resp, "status", 0) < 200 or getattr(resp, "status", 0) >= 300:
                return []
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
            models = payload.get("models", [])
            names = [str(item.get("name", "")).strip() for item in models]
            return [name for name in names if name]
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return []


def _select_auto_local_backend(settings: BaymaxSettings) -> tuple[str, str]:
    """Choose a usable local backend in priority order for auto_local mode."""
    # 1) Prefer Ollama if server is reachable and has at least one model.
    ollama_models = _probe_ollama_models(settings.ollama_base_url)
    if ollama_models:
        if settings.ollama_model and settings.ollama_model in ollama_models:
            return "ollama", settings.ollama_model

        preferred_prefixes = (
            "qwen2.5:1.5b",
            "qwen2.5:3b",
            "qwen2.5:7b",
            "qwen2.5",
            "qwen",
        )
        for prefix in preferred_prefixes:
            match = next((name for name in ollama_models if name.startswith(prefix)), None)
            if match:
                return "ollama", match

        return "ollama", ollama_models[0]

    # 2) Try local transformers when package is installed.
    if importlib.util.find_spec("transformers") is not None:
        return "transformers", settings.hf_chat_model

    # 3) Last resort.
    return "rule_based", ""


def create_dialogue_provider(
    settings: BaymaxSettings | None = None,
) -> DialogueProvider:
    """Create and return the configured DialogueProvider.

    Selection priority:
        1. The backend specified by BAYMAX_DIALOGUE_BACKEND.
        2. If that backend fails to initialise and BAYMAX_ENABLE_RULE_BASED_FALLBACK
           is true, fall back to RuleBasedDialogue.
        3. If fallback is disabled, raise the original exception.

    Args:
        settings: Optional settings instance. Uses get_settings() if not provided.

    Returns:
        An initialised DialogueProvider.
    """
    if settings is None:
        settings = get_settings()

    backend = settings.dialogue_backend.lower()
    enable_fallback = settings.enable_rule_based_fallback
    enable_debug = settings.enable_dialogue_debug

    if backend == "auto_local":
        selected_backend, selected_model = _select_auto_local_backend(settings)
        logger.info(
            "Resolved auto_local dialogue backend -> %s (%s)",
            selected_backend,
            selected_model or "default",
        )
        backend = selected_backend
        if backend == "ollama" and selected_model:
            settings.ollama_model = selected_model

    if backend == "rule_based":
        logger.info("Using rule_based dialogue backend")
        return RuleBasedDialogue()

    if backend == "ollama":
        try:
            from baymax.dialogue.ollama_provider import OllamaDialogueProvider

            provider = OllamaDialogueProvider(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                temperature=settings.dialogue_temperature,
                max_new_tokens=settings.dialogue_max_new_tokens,
                enable_debug=enable_debug,
                enable_safe_health_mode=settings.enable_safe_health_mode,
            )
            logger.info(
                "Using Ollama dialogue backend (model=%s, url=%s)",
                settings.ollama_model,
                settings.ollama_base_url,
            )
            return provider
        except Exception as exc:
            logger.warning(
                "Failed to initialise Ollama dialogue backend: %s", exc
            )
            if enable_fallback:
                logger.info("Falling back to rule_based dialogue provider")
                return RuleBasedDialogue()
            raise

    if backend == "transformers":
        try:
            from baymax.dialogue.transformers_provider import (
                TransformersDialogueProvider,
            )

            provider = TransformersDialogueProvider(
                model_name=settings.hf_chat_model,
                dtype=settings.hf_dtype,
                temperature=settings.dialogue_temperature,
                max_new_tokens=settings.dialogue_max_new_tokens,
                enable_debug=enable_debug,
                enable_safe_health_mode=settings.enable_safe_health_mode,
                cache_dir=settings.model_dir if settings.model_dir else None,
            )
            if not provider.is_available():
                raise RuntimeError(
                    f"Transformers backend unavailable: {provider._load_error}"
                )
            logger.info(
                "Using Transformers dialogue backend (model=%s)",
                settings.hf_chat_model,
            )
            return provider
        except Exception as exc:
            logger.warning(
                "Failed to initialise Transformers dialogue backend: %s", exc
            )
            if enable_fallback:
                logger.info("Falling back to rule_based dialogue provider")
                return RuleBasedDialogue()
            raise

    # Unknown backend
    logger.warning(
        "Unknown dialogue backend '%s'. Falling back to rule_based.", backend
    )
    if enable_fallback:
        return RuleBasedDialogue()
    raise ValueError(
        f"Unknown dialogue backend: '{backend}'. "
        f"Valid options: {AVAILABLE_BACKENDS}"
    )


def get_active_model(settings: BaymaxSettings | None = None) -> str:
    """Return the active model name for the configured backend."""
    if settings is None:
        settings = get_settings()
    backend = settings.dialogue_backend.lower()
    if backend == "ollama":
        return settings.ollama_model
    if backend == "transformers":
        return settings.hf_chat_model
    return ""

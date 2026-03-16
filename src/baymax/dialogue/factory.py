"""Dialogue provider factory.

Instantiates the configured DialogueProvider based on settings.
Falls back to RuleBasedDialogue if the selected backend cannot be initialised.
"""

import logging

from baymax.config.settings import BaymaxSettings, get_settings
from baymax.dialogue.interfaces import DialogueProvider
from baymax.dialogue.rule_based import RuleBasedDialogue

logger = logging.getLogger(__name__)

AVAILABLE_BACKENDS: list[str] = ["rule_based", "ollama", "transformers"]


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

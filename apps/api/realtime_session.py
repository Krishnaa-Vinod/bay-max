"""Realtime voice session provisioning endpoints."""

from __future__ import annotations

import os
from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel

from baymax.config.settings import get_settings

router = APIRouter(prefix="/v1/realtime", tags=["realtime"])


class RealtimeSessionResponse(BaseModel):
    """Provisioning payload for browser realtime voice session setup."""

    enabled: bool
    mode: str
    model: str
    reason: str = ""
    issued_at: str


@router.post("/session", response_model=RealtimeSessionResponse)
async def create_realtime_session() -> RealtimeSessionResponse:
    """Return realtime voice session configuration for the UI.

    This endpoint intentionally does not expose API keys.
    The UI uses this response to choose realtime vs local-chained mode.
    """
    settings = get_settings()
    model = settings.realtime_voice_model

    if not settings.enable_realtime_voice:
        return RealtimeSessionResponse(
            enabled=False,
            mode="local_chained_voice",
            model=model,
            reason="Realtime voice disabled by config",
            issued_at=datetime.utcnow().isoformat() + "Z",
        )

    # Require an upstream API key to consider realtime as available.
    if not os.getenv("OPENAI_API_KEY"):
        return RealtimeSessionResponse(
            enabled=False,
            mode="local_chained_voice",
            model=model,
            reason="OPENAI_API_KEY is not set",
            issued_at=datetime.utcnow().isoformat() + "Z",
        )

    return RealtimeSessionResponse(
        enabled=True,
        mode="realtime_voice",
        model=model,
        reason="",
        issued_at=datetime.utcnow().isoformat() + "Z",
    )

"""URL fetch and source summarization helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from html import unescape
from typing import Any
from urllib.request import Request, urlopen


@dataclass
class FetchResult:
    """Fetched web page excerpt."""

    url: str
    title: str
    excerpt: str
    fetched_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "snippet": self.excerpt,
            "source": "fetch_url",
            "fetched_at": self.fetched_at,
            "confidence": None,
        }


def _clean_text(value: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", value)
    text = unescape(without_tags)
    return re.sub(r"\s+", " ", text).strip()


def fetch_url(url: str, timeout_sec: float = 8.0) -> FetchResult:
    """Fetch one URL and return normalized title/excerpt."""
    req = Request(
        url,
        headers={
            "User-Agent": "baymax/0.11 (+local assistant)",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urlopen(req, timeout=timeout_sec) as response:
        payload = response.read().decode("utf-8", errors="ignore")

    title_match = re.search(r"<title[^>]*>([\s\S]*?)</title>", payload)
    title = _clean_text(title_match.group(1)) if title_match else url

    body_match = re.search(r"<body[^>]*>([\s\S]*?)</body>", payload)
    body_text = _clean_text(body_match.group(1) if body_match else payload)
    excerpt = body_text[:1200]

    return FetchResult(
        url=url,
        title=title,
        excerpt=excerpt,
        fetched_at=datetime.now(UTC).isoformat(),
    )


def summarize_sources(items: list[dict[str, Any]]) -> str:
    """Build a compact source summary for prompt/context injection."""
    lines: list[str] = []
    for item in items[:5]:
        title = str(item.get("title") or "Untitled")
        snippet = str(item.get("snippet") or "").strip()
        url = str(item.get("url") or "")
        if snippet:
            lines.append(f"- {title}: {snippet} ({url})")
        else:
            lines.append(f"- {title} ({url})")
    return "\n".join(lines)

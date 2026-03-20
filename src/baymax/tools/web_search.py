"""Web search provider abstraction for live tool usage."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from html import unescape
from typing import Any
from urllib.parse import parse_qs, quote_plus, urlparse
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


@dataclass
class WebSearchResult:
    """Single search result item."""

    title: str
    url: str
    snippet: str
    source: str = "duckduckgo"
    fetched_at: str = ""
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "fetched_at": self.fetched_at,
            "confidence": self.confidence,
        }


def _normalize_redirect_url(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    if parsed.path == "/l/":
        query = parse_qs(parsed.query)
        candidate = query.get("uddg", [""])[0]
        if candidate:
            return candidate
    return raw_url


def _strip_tags(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value)
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def search_web(
    query: str,
    provider: str = "duckduckgo",
    max_results: int = 5,
    timeout_sec: float = 8.0,
) -> list[WebSearchResult]:
    """Search the public web and return structured results."""
    if provider != "duckduckgo":
        raise ValueError(f"Unsupported web search provider: {provider}")

    endpoint = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
    req = Request(
        endpoint,
        headers={
            "User-Agent": "baymax/0.11 (+local assistant)",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )

    with urlopen(req, timeout=timeout_sec) as response:
        payload = response.read().decode("utf-8", errors="ignore")

    blocks = re.findall(r"<div class=\"result\"[\s\S]*?</div>\s*</div>", payload)
    results: list[WebSearchResult] = []
    for block in blocks:
        title_match = re.search(
            r"<a[^>]+class=\"result__a\"[^>]+href=\"([^\"]+)\"[^>]*>([\s\S]*?)</a>",
            block,
        )
        if not title_match:
            continue

        raw_url, raw_title = title_match.groups()
        snippet_match = re.search(
            r"<a[^>]+class=\"result__snippet\"[^>]*>([\s\S]*?)</a>|<div[^>]+class=\"result__snippet\"[^>]*>([\s\S]*?)</div>",
            block,
        )
        raw_snippet = ""
        if snippet_match:
            raw_snippet = snippet_match.group(1) or snippet_match.group(2) or ""

        results.append(
            WebSearchResult(
                title=_strip_tags(raw_title),
                url=_normalize_redirect_url(unescape(raw_url)),
                snippet=_strip_tags(raw_snippet),
                fetched_at=datetime.now(UTC).isoformat(),
            )
        )
        if len(results) >= max_results:
            break

    return results

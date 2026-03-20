"""Web and runtime tool integrations for Bay-Max."""

from baymax.tools.web_search import WebSearchResult, search_web
from baymax.tools.web_fetch import FetchResult, fetch_url, summarize_sources

__all__ = [
    "WebSearchResult",
    "FetchResult",
    "search_web",
    "fetch_url",
    "summarize_sources",
]

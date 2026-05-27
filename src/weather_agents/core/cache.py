"""Lightweight LRU cache for LLM responses.

Deduplicates identical requests within a configurable time window.
Keyed by (model, messages_json) hash.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict


class LLMCache:
    """LRU cache for LLM completions with TTL expiration."""

    def __init__(self, max_size: int = 128, ttl_seconds: int = 60) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._cache: OrderedDict[str, tuple[float, str]] = OrderedDict()

    def _make_key(
        self,
        model: str,
        messages: list[dict],
        params: dict | None = None,
    ) -> str:
        raw = json.dumps([model, messages, params or {}], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def get(
        self,
        model: str,
        messages: list[dict],
        params: dict | None = None,
    ) -> str | None:
        """Return cached response or None if miss/expired."""
        key = self._make_key(model, messages, params)
        entry = self._cache.get(key)
        if entry is None:
            return None
        ts, response = entry
        if time.time() - ts >= self._ttl:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return response

    def set(
        self,
        model: str,
        messages: list[dict],
        response: str,
        params: dict | None = None,
    ) -> None:
        """Cache a non-empty response. Refuses to cache empty/very-short content."""
        if not response or len(response) < 10:
            return
        key = self._make_key(model, messages, params)
        self._cache[key] = (time.time(), response)
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)

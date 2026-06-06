"""Web search implementation — DDG + Bing race with HTML fallbacks, cache, throttle.

Extracted from builtin.py to keep that file lean. Cache helpers
(``_cache_get`` / ``_cache_put`` / ``_search_cache_key``) live here and are
also re-exported from builtin so existing tests keep working.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any


async def _get_http():
    """Lazy import the shared httpx client living in builtin.py."""
    from weather_agents.tools.builtin import _get_http as _builtin_get_http

    return await _builtin_get_http()


# -- Web Search (ddgs library with HTML fallback) --

_web_search_timestamps: list[float] = []
_WEB_SEARCH_MAX_PER_SEC = 2  # throttle to avoid rate-limiting

# Short-lived result cache: a successful search is reused for identical queries
# within the TTL. This makes the agent's repeat/refine searches instant and
# avoids hammering the backends — the single biggest speed win for research
# turns that fire several similar queries.
_web_search_cache: dict[str, tuple[float, str]] = {}
_WEB_SEARCH_TTL = 300.0  # seconds
_WEB_SEARCH_CACHE_MAX = 64


def _search_cache_key(query: str, n: int) -> str:
    return f"{query.strip().lower()}|{n}"


# Same idea for fetched pages: research turns often fetch the same URL more than
# once (e.g. re-read after a search). Reuse within the TTL.
_fetch_page_cache: dict[str, tuple[float, str]] = {}
_FETCH_PAGE_TTL = 300.0  # seconds
_FETCH_PAGE_CACHE_MAX = 32


def _cache_get(store: dict[str, tuple[float, str]], key: str, ttl: float) -> str | None:
    hit = store.get(key)
    if hit is None:
        return None
    ts, val = hit
    if time.monotonic() - ts < ttl:
        return val
    del store[key]
    return None


def _cache_put(store: dict[str, tuple[float, str]], key: str, value: str, cap: int) -> None:
    if len(store) >= cap:
        oldest = min(store, key=lambda k: store[k][0])
        del store[oldest]
    store[key] = (time.monotonic(), value)


async def _race_search(
    coros: dict[str, Any],
    timeout: float,
    errors: list[str],
) -> list[dict] | None:
    """Run search coroutines concurrently; return the first *non-empty* result.

    A backend that finishes empty doesn't end the race — we keep waiting for a
    sibling until one yields results or the overall timeout elapses. Pending
    tasks are cancelled on exit so a slow/blocked backend never holds us up.
    """
    tasks = {asyncio.ensure_future(c): name for name, c in coros.items()}
    pending = set(tasks)
    found: list[dict] | None = None
    try:
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while pending and found is None:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            done, pending = await asyncio.wait(
                pending, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
            )
            if not done:
                break
            for d in done:
                name = tasks[d]
                try:
                    r = d.result()
                except Exception as exc:  # noqa: BLE001 — record and try siblings
                    errors.append(f"{name}: {exc}")
                    continue
                if r:
                    found = r
                    break
                errors.append(f"{name}: 0 results")
    finally:
        for t in pending:
            t.cancel()
    return found


async def _web_search(query: str, num_results: int = 5, **kwargs) -> str:
    """Search the web using available backends (DuckDuckGo, Bing). Cached + rate-limited."""
    now = time.monotonic()
    n = min(num_results, 10)

    # Cache hit → return instantly (skips network + rate-limit).
    key = _search_cache_key(query, n)
    cached = _cache_get(_web_search_cache, key, _WEB_SEARCH_TTL)
    if cached is not None:
        return cached

    global _web_search_timestamps
    _web_search_timestamps = [t for t in _web_search_timestamps if now - t < 1.0]
    if len(_web_search_timestamps) >= _WEB_SEARCH_MAX_PER_SEC:
        wait = 1.0 - (now - _web_search_timestamps[0]) + 0.05
        if wait > 0:
            await asyncio.sleep(wait)

    results: list[dict] | None = None
    errors: list[str] = []

    # Race the two fast *direct* HTML scrapes — Bing and DuckDuckGo — and take
    # whichever returns results first, so we never pay a blocked backend's full
    # timeout. In mainland China Bing wins (DDG is blocked); in DDG-friendly
    # networks DDG wins. The loser is cancelled. (The ddgs *library* is a
    # last-ditch fallback below — it's reliable but slow, hitting several
    # upstreams serially, so it shouldn't gate the common case.)
    # Look up backends via the builtin module so tests can monkeypatch
    # `builtin._bing_html_search` / `builtin._ddg_html_fallback` and have those
    # patches actually intercept the call (otherwise the local closures bypass
    # the patch — silent test breakage).
    from weather_agents.tools import builtin as _b

    results = await _race_search(
        {
            "bing-html": _b._bing_html_search(query, n),
            "ddg-html": _b._ddg_html_fallback(query, n),
        },
        timeout=8.0,
        errors=errors,
    )
    if not results:
        try:
            results = await asyncio.wait_for(_ddg_api_search(query, n), timeout=8.0)
        except Exception as e:
            errors.append(f"ddgs: {e}")
    if not results:
        bing_key = os.environ.get("BING_API_KEY")
        if bing_key:
            try:
                results = await asyncio.wait_for(_bing_search(query, n, bing_key), timeout=8.0)
            except Exception as e3:
                errors.append(f"bing: {e3}")

    _web_search_timestamps.append(time.monotonic())

    if not results:
        detail = "; ".join(errors) if errors else "no backends available"
        return f"No results found for '{query}' ({detail})"

    output_parts = [f"Search results for: {query}\n"]
    for i, r in enumerate(results, 1):
        output_parts.append(f"{i}. {r['title']}")
        output_parts.append(f"   {r['url']}")
        if r.get("snippet"):
            output_parts.append(f"   {r['snippet']}")
        output_parts.append("")
    text = "\n".join(output_parts)
    _cache_put(_web_search_cache, key, text, _WEB_SEARCH_CACHE_MAX)  # successes only
    return text


def _ddg_api_search_sync(query: str, num_results: int) -> list[dict]:
    """Synchronous core of the ddgs search — must run in a worker thread."""
    from ddgs import DDGS

    results: list[dict] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=num_results):
            results.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                }
            )
    return results


async def _ddg_api_search(query: str, num_results: int) -> list[dict]:
    """Primary: ddgs library. Run in a thread because ddgs is synchronous and
    can stall for tens of seconds on transient yahoo/bing backend errors.
    Without to_thread() such a stall freezes the entire event loop — stream
    rendering, Esc handling and other concurrent tool calls all wedge."""
    return await asyncio.to_thread(_ddg_api_search_sync, query, num_results)


def _parse_ddg_html(html: str, max_results: int) -> list[dict]:
    """Extract search results from DuckDuckGo HTML (used by fallback)."""
    from re import DOTALL
    from re import compile as re_compile
    from re import sub as re_sub

    results: list[dict] = []
    pattern = re_compile(
        r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
        r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
        DOTALL,
    )
    for match in pattern.finditer(html):
        if len(results) >= max_results:
            break
        url = match.group(1)
        title = re_sub(r"<[^>]+>", "", match.group(2)).strip()
        snippet = re_sub(r"<[^>]+>", "", match.group(3)).strip()

        if "uddg=" in url:
            from urllib.parse import parse_qs, unquote, urlparse

            try:
                qs = parse_qs(urlparse(url).query)
                url = unquote(qs.get("uddg", [url])[0])
            except Exception:
                pass

        if title:
            results.append({"title": title, "url": url, "snippet": snippet})
    return results


async def _ddg_html_fallback(query: str, num_results: int) -> list[dict]:
    """Fallback: scrape DuckDuckGo HTML when the library is unavailable."""
    client = await _get_http()
    resp = await client.get(
        "https://html.duckduckgo.com/html/",
        params={"q": query},
        headers={"User-Agent": "Mozilla/5.0 (compatible; Skyloom/1.0)"},
    )
    if resp.status_code != 200:
        return []
    return _parse_ddg_html(resp.text, num_results)


def _parse_bing_html(html_text: str, max_results: int) -> list[dict]:
    """Extract results from Bing's SERP HTML (keyless). Works where DuckDuckGo
    is blocked (e.g. mainland China), so it's our primary backend."""
    import html as _html
    from re import DOTALL, IGNORECASE
    from re import compile as re_compile
    from re import sub as re_sub

    block_rx = re_compile(r'<li class="b_algo"[^>]*>(.*?)</li>', DOTALL | IGNORECASE)
    a_rx = re_compile(r'<h2[^>]*>\s*<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>', DOTALL | IGNORECASE)
    p_rx = re_compile(r"<p[^>]*>(.*?)</p>", DOTALL | IGNORECASE)

    results: list[dict] = []
    for block_match in block_rx.finditer(html_text):
        if len(results) >= max_results:
            break
        block = block_match.group(1)
        a_match = a_rx.search(block)
        if not a_match:
            continue
        url = a_match.group(1)
        title = _html.unescape(re_sub(r"<[^>]+>", "", a_match.group(2))).strip()
        p_match = p_rx.search(block)
        snippet = (
            _html.unescape(re_sub(r"<[^>]+>", "", p_match.group(1))).strip() if p_match else ""
        )
        if title and url.startswith("http"):
            results.append({"title": title, "url": url, "snippet": snippet})
    return results


async def _bing_html_search(query: str, num_results: int) -> list[dict]:
    """Primary keyless backend: scrape Bing's SERP. Fast and reachable inside
    mainland China (unlike DuckDuckGo), so it usually returns on the first try
    — which stops the agent from re-searching in a loop."""
    client = await _get_http()
    resp = await client.get(
        "https://www.bing.com/search",
        params={"q": query, "setlang": "en"},
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            )
        },
    )
    if resp.status_code != 200:
        return []
    return _parse_bing_html(resp.text, num_results)


async def _bing_search(query: str, num_results: int, api_key: str) -> list[dict]:
    """Search using Bing Web Search API (requires BING_API_KEY env var)."""
    client = await _get_http()
    resp = await client.get(
        "https://api.bing.microsoft.com/v7.0/search",
        params={"q": query, "count": num_results, "mkt": "zh-CN"},
        headers={"Ocp-Apim-Subscription-Key": api_key},
        timeout=10,
    )
    if resp.status_code != 200:
        return []
    data = resp.json()
    results = []
    for r in (data.get("webPages") or {}).get("value", [])[:num_results]:
        results.append(
            {
                "title": r.get("name", ""),
                "url": r.get("url", ""),
                "snippet": r.get("snippet", ""),
            }
        )
    return results


# ── Skill-promoted tools ──────────────────────────────────────────────
# These three helpers used to be injected at skill-activation time by
# code_reviewer / security_auditor / web_research (the Python skill
# modules that were dropped in the SKILL.md-unification refactor). They
# stay useful on their own — there's no reason a non-active skill can't
# call lint_file — so they're now permanently registered alongside the
# other builtins. The SKILL.md files that replaced those modules just
# reference these tools by name.

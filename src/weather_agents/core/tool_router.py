"""Tool-subset selection for LLM calls.

Without filtering, every chat turn ships ~50 tool schemas (built-ins + MCP +
skill-required + delegation) to the model. That dilutes attention (the LLM
picks plausible-but-wrong tools more often) and burns 8-15k input tokens per
turn. This module narrows the active tool set to ~12 by lightweight scoring
against the user's latest message.

The router intentionally avoids embeddings / LLM calls — it must run in <1ms
on every turn, before the real LLM call. A coarse keyword/substring score is
good enough to keep the right tools in and bad enough to be cheap.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from weather_agents.core.tool import Tool, ToolRegistry


# Infrastructure tools that must ALWAYS be visible to the LLM regardless of
# query content — without them the agent loses delegation, shared-memory I/O,
# skill discovery, and long-term recall. These are cheap (the LLM rarely
# misfires them) so always-include is the safe default.
_ALWAYS_INCLUDE: frozenset[str] = frozenset(
    {
        "delegate_to",
        "read_shared_memory",
        "list_shared_memory",
        "list_skills",
        "use_skill",
        "recall_facts",
        "remember_fact",
    }
)

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*|[一-鿿]+")

# Common stopwords that don't carry tool-routing signal. Kept short on
# purpose — the goal is to filter noise, not aggressive stemming.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "and",
        "or",
        "but",
        "to",
        "for",
        "of",
        "in",
        "on",
        "at",
        "with",
        "by",
        "do",
        "did",
        "does",
        "i",
        "me",
        "my",
        "you",
        "your",
        "it",
        "this",
        "that",
        "what",
        "how",
        "can",
        "could",
        "would",
        "should",
        "please",
        "tell",
        "show",
        "help",
        "ok",
        "yes",
        "no",
        "好",
        "的",
        "是",
        "我",
        "你",
        "他",
        "她",
        "它",
        "这",
        "那",
        "什么",
        "怎么",
        "请",
        "帮",
        "麻烦",
    }
)


def _tokenize(text: str) -> set[str]:
    return {
        t.lower()
        for t in _TOKEN_RE.findall(text or "")
        if len(t) >= 2 and t.lower() not in _STOPWORDS
    }


def _score_tool(tool: Tool, query_tokens: set[str]) -> int:
    """Higher score = more likely relevant. Cheap heuristic, not exact."""
    if not query_tokens:
        return 0
    score = 0
    # Tool name tokens carry the strongest signal: an exact substring match
    # on a tool name almost always means the user/LLM wants this tool.
    name_tokens = _tokenize(tool.name.replace("_", " "))
    for qt in query_tokens:
        if qt in name_tokens:
            score += 5
        # Substring hits on the raw name catch fuzzy cases like "fileread"
        # against tool name "read_file".
        elif qt in tool.name.lower():
            score += 3
    # Description tokens are weaker — many tools share generic words.
    desc_tokens = _tokenize(tool.description)
    for qt in query_tokens:
        if qt in desc_tokens:
            score += 1
    return score


def select_relevant_tools(
    registry: ToolRegistry,
    candidate_names: list[str],
    query: str,
    *,
    top_k: int = 12,
    must_include: set[str] | None = None,
) -> list[str]:
    """Return up to ~top_k tool names ordered by relevance to the query.

    Always-included infrastructure tools and ``must_include`` (e.g. active
    skill required_tools) are appended regardless of score. When the candidate
    set is already small (≤ top_k + |must_include|), no filtering is applied.

    A short or empty query means we have no signal to filter — returning the
    full candidate set is correct in that case (we'd rather burn tokens than
    hide a tool the user actually needs).
    """
    must = set(must_include or set()) | _ALWAYS_INCLUDE
    must_present = [n for n in candidate_names if n in must]
    remaining = [n for n in candidate_names if n not in must]

    # No filtering when the set is already small or the query is too short
    # to score reliably. Threshold of 3 tokens ≈ "show me weather" — anything
    # smaller is too generic for the heuristic to help.
    query_tokens = _tokenize(query)
    if len(remaining) <= top_k or len(query_tokens) < 2:
        return must_present + remaining

    scored: list[tuple[int, str]] = []
    for name in remaining:
        tool = registry.get(name)
        if tool is None:
            continue
        scored.append((_score_tool(tool, query_tokens), name))

    # Stable sort by descending score; tools with zero score still appear but
    # only fill the remaining slots up to top_k.
    scored.sort(key=lambda x: x[0], reverse=True)
    picked = [n for _s, n in scored[:top_k]]
    return must_present + picked

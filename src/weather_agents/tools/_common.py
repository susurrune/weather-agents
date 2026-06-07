"""Shared primitives for the builtin tool modules.

Constants + tiny helpers used across the themed tool files (git_tools,
file/net/search groups). Kept dependency-free so any tool module can import it
without risking an import cycle back through ``builtin``.
"""

from __future__ import annotations

# Output size caps — shared so every tool truncates consistently.
_MAX_FILE_BYTES = 50_000
_MAX_SHELL_OUTPUT = 20_000
_MAX_SEARCH_OUTPUT = 10_000
_MAX_CODE_SEARCH_FILES = 5_000
_MAX_GREP_FILES = 10_000
_MAX_GREP_MATCHES = 200


def _truncate(text: str, limit: int, label: str = "output") -> str:
    """Truncate text with a visible marker so the LLM knows there was more."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n[... truncated, total {len(text)} chars of {label}]"

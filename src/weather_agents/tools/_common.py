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


# ── Write-protection paths ──

_WRITE_PROTECT_EXACT = {
    "/",
    "/*",
    "/.",
    "~",
    "~/",
    ".",
    "..",
    "*",
    "\\",
    "\\\\",
}

_WRITE_PROTECT_ROOTS = {
    "/etc",
    "/boot",
    "/bin",
    "/sbin",
    "/lib",
    "/lib64",
    "/usr",
    "/opt",
    "/var",
    "/root",
    "/proc",
    "/sys",
    "/dev",
    "/private/etc",
    "/private/var",
    "/private/tmp",
    "c:\\windows",
    "c:\\program files",
    "c:\\program files (x86)",
    "c:\\programdata",
    "d:\\windows",
    "d:\\program files",
}


def _is_protected_path(path: str) -> bool:
    """Check if a path is inside a system-protected directory tree."""
    import os as _os

    expanded = _os.path.expanduser(path)
    candidates: list[str] = [_os.path.normpath(expanded).lower()]
    try:
        real = _os.path.realpath(expanded).lower()
        if real != candidates[0]:
            candidates.append(real)
    except (OSError, ValueError):
        pass

    for resolved in candidates:
        normalized = resolved.rstrip(_os.sep)
        if normalized in _WRITE_PROTECT_EXACT:
            return True
        if len(normalized) <= 3 and (normalized.endswith(":") or normalized.endswith(":\\")):
            return True
        for root in _WRITE_PROTECT_ROOTS:
            r = _os.path.normpath(root).lower()
            if normalized == r or normalized.startswith(r + _os.sep):
                return True
    return False


def _truncate(text: str, limit: int, label: str = "output") -> str:
    """Truncate text with a visible marker so the LLM knows there was more."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n[... truncated, total {len(text)} chars of {label}]"

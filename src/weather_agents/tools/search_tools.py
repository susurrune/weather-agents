"""File search / code search / grep tools.

Extracted from ``builtin.py``. Depends on ``_common.py`` for truncation + caps.
"""

from __future__ import annotations

import os
import re as _re
from pathlib import Path

from weather_agents.tools._common import (
    _MAX_CODE_SEARCH_FILES,
    _MAX_GREP_FILES,
    _MAX_GREP_MATCHES,
    _MAX_SEARCH_OUTPUT,
    _is_protected_path,
    _truncate,
)

# -- Search Tools --


async def _list_directory(path: str = ".", include_hidden: bool = False, **kwargs) -> str:
    """List files and directories with basic metadata."""
    path = os.path.expanduser(path)
    try:
        entries = []
        for entry in sorted(os.scandir(path), key=lambda e: (not e.is_dir(), e.name)):
            if not include_hidden and entry.name.startswith("."):
                continue
            if entry.is_dir():
                entries.append(f"  [dir]  {entry.name}/")
            else:
                try:
                    size = entry.stat().st_size
                    if size < 1024:
                        size_str = f"{size}B"
                    elif size < 1024 * 1024:
                        size_str = f"{size / 1024:.1f}KB"
                    else:
                        size_str = f"{size / 1024 / 1024:.1f}MB"
                    entries.append(f"  {size_str:>8}  {entry.name}")
                except OSError:
                    entries.append(f"           {entry.name}")
        if not entries:
            return f"Empty directory: {path}"
        header = f"Directory: {os.path.abspath(path)} ({len(entries)} items)\n"
        return header + "\n".join(entries)
    except FileNotFoundError:
        return f"Error: Directory not found: {path}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error listing directory: {e}"


async def _tree(
    directory: str = ".", max_depth: int = 3, include_hidden: bool = False, **kwargs
) -> str:
    """Show directory tree structure."""
    directory = os.path.expanduser(directory)
    lines = []
    try:
        base = os.path.abspath(directory)
        lines.append(base)
        _tree_walk(base, "", lines, 0, int(max_depth), include_hidden)
        if len(lines) > 200:
            lines = lines[:200]
            lines.append("... (truncated)")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def _tree_walk(
    path: str, prefix: str, lines: list, depth: int, max_depth: int, include_hidden: bool
) -> None:
    if depth >= max_depth or len(lines) > 200:
        return
    try:
        entries = sorted(os.scandir(path), key=lambda e: (not e.is_dir(), e.name))
    except PermissionError:
        return
    if not include_hidden:
        entries = [e for e in entries if not e.name.startswith(".")]
    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "+-" if is_last else "|-"
        lines.append(f"{prefix}{connector} {entry.name}{'/' if entry.is_dir() else ''}")
        if entry.is_dir():
            extension = "   " if is_last else "|  "
            _tree_walk(entry.path, prefix + extension, lines, depth + 1, max_depth, include_hidden)


async def _move_file(src: str, dst: str, **kwargs) -> str:
    """Move or rename a file or directory."""
    src, dst = os.path.expanduser(src), os.path.expanduser(dst)
    if _is_protected_path(src):
        return f"Error: refusing to move protected path: {src}"
    try:
        import shutil

        os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
        shutil.move(src, dst)
        return f"Moved: {src} -> {dst}"
    except Exception as e:
        return f"Error moving: {e}"


async def _copy_file(src: str, dst: str, **kwargs) -> str:
    """Copy a file or directory."""
    src, dst = os.path.expanduser(src), os.path.expanduser(dst)
    if _is_protected_path(src):
        return f"Error: refusing to copy protected path: {src}"
    try:
        import shutil

        os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        return f"Copied: {src} -> {dst}"
    except Exception as e:
        return f"Error copying: {e}"


async def _delete_file(path: str, **kwargs) -> str:
    """Delete a file or empty directory (non-recursive for safety)."""
    path = os.path.expanduser(path)
    if _is_protected_path(path):
        return f"Error: refusing to delete protected path: {path}"
    try:
        if os.path.isdir(path):
            os.rmdir(path)
            return f"Deleted directory: {path}"
        else:
            os.remove(path)
            return f"Deleted: {path}"
    except OSError as e:
        if "directory not empty" in str(e).lower():
            return f"Error: Directory not empty (recursive delete not supported for safety): {path}"
        return f"Error deleting: {e}"


async def _get_cwd(**kwargs) -> str:
    """Return current working directory."""
    return os.getcwd()


async def _set_user_profile(key: str = "", value: str = "", **kwargs) -> str:
    """Persist a fact about the user into the local profile (用户画像)."""
    key = (key or "").strip()
    if not key:
        return "Error: key is required"
    from weather_agents.core.profile import set_profile_field

    set_profile_field(key, value)
    return f"已记住：{key} = {value}"


async def _remember(note: str = "", **kwargs) -> str:
    """Append a free-form emotional/contextual memory about the user."""
    note = (note or "").strip()
    if not note:
        return "Error: note is required"
    from weather_agents.core.profile import append_memory

    if not append_memory(note):
        return "Error: empty note"
    return f"已记在心里：{note}"


async def _set_persona(agent: str = "", persona: str = "", **kwargs) -> str:
    """Save a custom persona for an agent (applied from the next turn)."""
    agent = (agent or "").strip().lower()
    persona = (persona or "").strip()
    if not agent or not persona:
        return "Error: agent and persona are both required"
    from weather_agents.core.profile import save_persona

    if not save_persona(agent, persona):
        return f"Error: unknown agent '{agent}' (use fog/rain/frost/snow/dew/fair)"
    return f"已保存「{agent}」的新角色设定（下次启动该 agent 时生效）。"


async def _file_search(directory: str, pattern: str, max_depth: int = 0, **kwargs) -> str:
    """Glob-search for files. Uses pathlib for cross-platform correctness.

    Set max_depth > 0 to limit recursion depth.
    """
    from pathlib import Path

    try:
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            return f"Error: not a directory: {directory}"
        pattern_parts = pattern.count("**")
        if pattern_parts == 0 and max_depth <= 0:
            max_depth = 10  # default cap for non-recursive patterns
        matches: list[str] = []
        for fp in root.rglob(pattern):
            if not fp.is_file():
                continue
            if max_depth > 0:
                relative_depth = len(fp.relative_to(root).parents) - 1
                if relative_depth > max_depth:
                    continue
            matches.append(str(fp))
            if len(matches) >= 500:
                return _truncate("\n".join(matches), _MAX_SEARCH_OUTPUT, f"matches ({500}+ found)")
    except OSError as e:
        return f"Error searching: {e}"
    if not matches:
        return f"No files matching '{pattern}' found in {directory}"
    truncated = len(matches) > 50
    out = "\n".join(matches[:50])
    if truncated:
        out += f"\n\n[... {len(matches) - 50} more matches not shown]"
    return out


async def _code_search(
    directory: str,
    query: str,
    regex: bool = False,
    **kwargs,
) -> str:
    """Search for text or regex in source files. Set regex=True for regex mode."""
    from pathlib import Path

    suffixes = {
        ".py",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".go",
        ".rs",
        ".java",
        ".yaml",
        ".yml",
        ".json",
        ".toml",
        ".md",
        ".c",
        ".cpp",
        ".h",
        ".css",
        ".html",
        ".vue",
        ".svelte",
        ".sh",
        ".bash",
        ".ps1",
        ".sql",
        ".r",
        ".rb",
        ".php",
    }
    skip_dirs = {
        ".git",
        "node_modules",
        ".venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".next",
        "target",
        "vendor",
        ".tox",
        ".eggs",
    }

    try:
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            return f"Error: not a directory: {directory}"
    except OSError as e:
        return f"Error: {e}"

    matcher: object
    if regex:
        try:
            matcher = _re.compile(query)
        except _re.error as e:
            return f"Error: invalid regex '{query}': {e}"
    else:
        matcher = query

    matches: list[str] = []
    files_scanned = 0
    for fp in root.rglob("*"):
        if not fp.is_file() or fp.suffix not in suffixes:
            continue
        if any(part in skip_dirs for part in fp.parts):
            continue
        files_scanned += 1
        if files_scanned > _MAX_CODE_SEARCH_FILES:
            return _truncate(
                "\n".join(matches),
                _MAX_SEARCH_OUTPUT,
                f"matches (stopped after {_MAX_CODE_SEARCH_FILES} files)",
            )
        try:
            with fp.open(encoding="utf-8", errors="ignore") as fh:
                for i, line in enumerate(fh, 1):
                    hit = (
                        matcher.search(line)  # type: ignore[union-attr]
                        if regex
                        else (query in line)
                    )
                    if hit:
                        matches.append(f"{fp}:{i}:{line.rstrip()}")
                        if len(matches) >= 100:
                            return _truncate("\n".join(matches), _MAX_SEARCH_OUTPUT, "matches")
        except OSError:
            continue
    if not matches:
        return f"No matches for '{query}' in {directory} (scanned {files_scanned} files)"
    return _truncate("\n".join(matches), _MAX_SEARCH_OUTPUT, "matches")


# -- Grep Tool (general-purpose text search) --


_BINARY_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".so",
    ".dll",
    ".exe",
    ".bin",
    ".dat",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".ico",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".svgz",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".otf",
    ".mp3",
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wav",
    ".flac",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".o",
    ".a",
    ".lib",
    ".class",
    ".jar",
    ".war",
}

_SKIP_DIRS_GREP = {
    ".git",
    "node_modules",
    ".venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".next",
    "target",
    "vendor",
    ".tox",
    ".eggs",
    ".cache",
}


async def _grep(
    directory: str,
    pattern: str,
    glob: str = "",
    regex: bool = False,
    ignore_case: bool = False,
    context_around: int = 0,
    context_before: int = 0,
    context_after: int = 0,
    **kwargs,
) -> str:
    """Search for text or regex in all text files (not limited to code extensions).

    Supports file-type filtering via glob, regex mode, case-insensitive mode,
    and context lines. Skips binary files and common generated directories.
    """
    try:
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            return f"Error: not a directory: {directory}"
    except OSError as e:
        return f"Error: {e}"

    # Compile pattern
    flags = _re.IGNORECASE if ignore_case else 0
    if regex:
        try:
            compiled = _re.compile(pattern, flags)
        except _re.error as e:
            return f"Error: invalid regex '{pattern}': {e}"
    else:
        compiled = _re.compile(_re.escape(pattern), flags)

    # Build glob filter
    glob_pattern: str | None = None
    if glob:
        from fnmatch import translate as _fm_translate

        glob_pattern = _fm_translate(glob) if glob else None

    matches: list[str] = []
    files_scanned = 0

    for fp in root.rglob("*"):
        if not fp.is_file():
            continue
        if fp.suffix.lower() in _BINARY_EXTENSIONS:
            continue
        if any(part in _SKIP_DIRS_GREP for part in fp.parts):
            continue

        # Glob filter
        if glob_pattern:
            import re as _re2

            if not _re2.match(glob_pattern, fp.name):
                continue

        files_scanned += 1
        if files_scanned > _MAX_GREP_FILES:
            out = "\n".join(matches) if matches else "No matches found within limit."
            return _truncate(
                out, _MAX_SEARCH_OUTPUT, f"grep results (stopped after {_MAX_GREP_FILES} files)"
            )

        try:
            with fp.open(encoding="utf-8", errors="ignore") as fh:
                lines = fh.readlines()
        except OSError:
            continue

        ctx = context_around or context_before or 0
        ctx_after = context_around or context_after or 0

        for i, line in enumerate(lines):
            if compiled.search(line):
                ctx_before = max(0, context_before)
                if ctx > 0 or ctx_before > 0 or ctx_after > 0:
                    # Show context lines
                    start = max(0, i - max(ctx, ctx_before))
                    end = min(len(lines), i + max(ctx, ctx_after) + 1)
                    for j in range(start, end):
                        prefix = ":" if i == j else "-"
                        matches.append(f"{fp}:{j + 1}:{prefix}:{lines[j].rstrip()}")
                    matches.append("--")
                else:
                    matches.append(f"{fp}:{i + 1}:{line.rstrip()}")
                if len(matches) >= _MAX_GREP_MATCHES:
                    return _truncate(
                        "\n".join(matches),
                        _MAX_SEARCH_OUTPUT,
                        f"grep matches ({_MAX_GREP_MATCHES}+ found)",
                    )

    if not matches:
        return f"No matches for '{pattern}' in {directory} (scanned {files_scanned} files)"
    return _truncate("\n".join(matches), _MAX_SEARCH_OUTPUT, "grep matches")


# Git tools live in git_tools.py; imported here so existing tests &
# `register_builtin_tools()` keep working with the original names.
from weather_agents.tools.git_tools import (  # noqa: E402, F401
    _git_add,
    _git_checkout,
    _git_commit,
    _git_diff,
    _git_log,
    _git_status,
    _run_git_command,
)

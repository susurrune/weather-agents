"""Built-in tool implementations."""

from __future__ import annotations

import asyncio
import ipaddress
import os
import re as _re
import shlex
import time
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from weather_agents.core.constants import TASK_DONE_SENTINEL
from weather_agents.core.tool import Tool, ToolParameter, ToolRegistry

if TYPE_CHECKING:
    # httpx adds ~50ms at import; HTTP tools are rarely used by ``wa --help``
    # or other no-LLM paths. Defer to first call via ``_get_http``.
    import httpx

_MAX_FILE_BYTES = 50_000
_MAX_SHELL_OUTPUT = 20_000
_MAX_SEARCH_OUTPUT = 10_000
_MAX_CODE_SEARCH_FILES = 5_000
_MAX_GREP_FILES = 10_000
_MAX_GREP_MATCHES = 200

# Paths that write/delete tools should never touch.
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
    # Unix system dirs
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
    # macOS firmlinks: /etc, /var, /tmp are actually located under /private.
    # Without these, ``/private/etc/passwd`` (the real path) bypasses the check.
    "/private/etc",
    "/private/var",
    "/private/tmp",
    # Windows system dirs
    "c:\\windows",
    "c:\\program files",
    "c:\\program files (x86)",
    "c:\\programdata",
    "d:\\windows",
    "d:\\program files",
}


def _is_protected_path(path: str) -> bool:
    """Check if a path is inside a system-protected directory tree.

    Checks both the normalised lexical form AND the symlink-resolved real
    path: a user-controlled symlink (``~/safe`` → ``/etc/passwd``) would
    otherwise sneak past a pure ``normpath`` check.
    """
    expanded = os.path.expanduser(path)
    candidates: list[str] = [os.path.normpath(expanded).lower()]
    # realpath may raise on weird inputs (long paths on Windows, etc.); the
    # lexical form is already in ``candidates`` so a failure here is safe.
    try:
        real = os.path.realpath(expanded).lower()
        if real != candidates[0]:
            candidates.append(real)
    except (OSError, ValueError):
        pass

    for resolved in candidates:
        normalized = resolved.rstrip(os.sep)
        # Exact-match dangerous paths
        if normalized in _WRITE_PROTECT_EXACT:
            return True
        # Drive roots: "c:", "c:\", "d:", "d:\" (Windows)
        if len(normalized) <= 3 and (normalized.endswith(":") or normalized.endswith(":\\")):
            return True
        # Rooted protected directories
        for root in _WRITE_PROTECT_ROOTS:
            r = os.path.normpath(root).lower()
            if normalized == r or normalized.startswith(r + os.sep):
                return True
    return False


def _truncate(text: str, limit: int, label: str = "output") -> str:
    """Truncate text with a visible marker so the LLM knows there was more."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n[... truncated, total {len(text)} chars of {label}]"


async def _task_done(summary: str = "", **kwargs) -> str:
    """Signal that the current task is complete."""
    return TASK_DONE_SENTINEL


# -- File Tools --


async def _read_file(path: str, offset: int = 0, limit: int = 0, **kwargs) -> str:
    path = os.path.expanduser(path)
    try:
        with open(path, encoding="utf-8") as f:
            if offset or limit:
                lines = f.readlines()
                total_lines = len(lines)
                start = max(0, offset)
                end = start + limit if limit > 0 else total_lines
                window = lines[start:end]
                header = f"[lines {start + 1}-{min(end, total_lines)} of {total_lines} in {path}]\n"
                return header + _truncate("".join(window), _MAX_FILE_BYTES, "file")
            content = f.read()
        return _truncate(content, _MAX_FILE_BYTES, "file")
    except FileNotFoundError:
        return f"Error: File not found: {path}"
    except UnicodeDecodeError:
        return f"Error: {path} is not a UTF-8 text file (binary?)"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error reading file: {e}"


async def _write_file(path: str, content: str, **kwargs) -> str:
    path = os.path.expanduser(path)
    if _is_protected_path(path):
        return f"Error: refusing to write to protected path: {path}"
    existed = os.path.exists(path)
    old_size = os.path.getsize(path) if existed else 0
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        suffix = f" (overwritten, was {old_size}B)" if existed else ""
        return f"Successfully wrote to {path}{suffix}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error writing file: {e}"


async def _edit_file(path: str, old_text: str, new_text: str, count: int = 1, **kwargs) -> str:
    path = os.path.expanduser(path)
    if _is_protected_path(path):
        return f"Error: refusing to edit protected path: {path}"
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        content = content.replace(old_text, new_text, count)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully edited {path} ({count} occurrences)"
    except Exception as e:
        return f"Error editing file: {e}"


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


# -- Git Tools --


async def _git_status(repo: str = ".", **kwargs) -> str:
    """Run `git status --porcelain` in the given repository directory."""
    return await _run_git_command(["status", "--porcelain", "--branch"], cwd=repo)


async def _git_diff(staged: bool = False, path: str = "", repo: str = ".", **kwargs) -> str:
    """Run `git diff` (unstaged) or `git diff --staged` (staged).

    Specify path to diff a single file or directory.
    """
    args = ["diff"]
    if staged:
        args.append("--staged")
    if path:
        args.append("--")
        args.append(path)
    return await _run_git_command(args, cwd=repo)


async def _git_log(
    count: int = 10,
    oneline: bool = True,
    all_branches: bool = False,
    repo: str = ".",
    **kwargs,
) -> str:
    """Run `git log` with configurable format."""
    args = ["log"]
    if oneline:
        args.append("--oneline")
    if all_branches:
        args.append("--all")
    args.extend(["-n", str(min(count, 50))])
    return await _run_git_command(args, cwd=repo)


async def _git_add(files: str, repo: str = ".", **kwargs) -> str:
    """Stage one or more files for commit.

    Accepts space-separated file paths. Refuses `-A` / `--all` / `.` for safety.
    """
    file_list = shlex.split(files)
    if not file_list:
        return "Error: no files specified"
    dangerous = {"-A", "--all", ".", "*"}
    for f in file_list:
        if f in dangerous:
            return "Error: refusing to stage everything — specify individual files"
    return await _run_git_command(["add"] + file_list, cwd=repo)


async def _git_commit(message: str, repo: str = ".", **kwargs) -> str:
    """Create a git commit with the given message.

    Refuses empty commits (the -m flag ensures no interactive editor).
    """
    if not message or not message.strip():
        return "Error: commit message cannot be empty"
    msg = message.strip()
    if len(msg) > 2000:
        return "Error: commit message too long (>2000 chars)"
    return await _run_git_command(["commit", "-m", msg], cwd=repo)


async def _git_checkout(branch: str, create: bool = False, repo: str = ".", **kwargs) -> str:
    """Switch to a branch. Set create=true to create a new branch."""
    args = ["checkout"]
    if create:
        args.append("-b")
    args.append(branch)
    return await _run_git_command(args, cwd=repo)


async def _run_git_command(args: list[str], cwd: str = ".") -> str:
    """Execute a git command and return its formatted output."""
    work_dir = os.path.expanduser(cwd)
    if not os.path.isdir(work_dir):
        return f"Error: not a directory: {cwd}"

    # Verify this is inside a git repo
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "rev-parse",
            "--git-dir",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=work_dir,
        )
        _stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        if proc.returncode != 0:
            return f"Error: not a git repository: {work_dir}"
    except FileNotFoundError:
        return "Error: git not found. Is git installed?"
    except TimeoutError:
        return "Error: git rev-parse timed out"

    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=work_dir,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=30)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return "Error: git command timed out (30s)"

        stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
        stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""

        parts: list[str] = []
        if stdout:
            parts.append(_truncate(stdout, _MAX_SHELL_OUTPUT, "git output"))
        if stderr:
            parts.append("STDERR:\n" + _truncate(stderr, 5000, "stderr"))
        if proc.returncode != 0 and proc.returncode is not None:
            parts.append(f"[exit code: {proc.returncode}]")
        return "\n".join(parts) if parts else "Git command completed with no output."
    except FileNotFoundError:
        return "Error: git not found. Is git installed?"
    except Exception as e:
        return f"Error executing git command: {e}"


# -- Shell Tool (safe mode) --

_BLOCKED_COMMANDS = {
    # Disk / filesystem destruction
    "dd",
    "mkfs",
    "fdisk",
    "parted",
    "format",
    "diskpart",
    # Power / boot
    "shutdown",
    "reboot",
    "init",
    "poweroff",
    "halt",
    "grub-mkconfig",
    "update-grub",
    # User / privilege
    "passwd",
    "adduser",
    "userdel",
    "useradd",
    "su",
    "sudo",
    "doas",
    # Firewall / network state
    "iptables",
    "nft",
    "ip6tables",
    "ufw",
    "firewall-cmd",
    # Kernel / system control
    "sysctl",
    "modprobe",
    "insmod",
    "rmmod",
    # Permission / ownership changes
    "chmod",
    "chown",
    "chgrp",
    # Windows: destructive / system commands
    "del",
    "erase",
    "rmdir",
    "rd",
    "cipher",
    "reg",
    "regedit",
    "bcdedit",
    "icacls",
    "cacls",
    "takeown",
    "sc",
    "net",
    "wmic",
    "schtasks",
    # Recursive delete (blocked outright for safety; use delete_file tool instead)
    "rm",
}

# Paths whose recursive deletion is always refused (even with proper flags).
_PROTECTED_ROOTS = {
    "/",
    "//",
    "/*",
    "/.",
    "/home",
    "/root",
    "/etc",
    "/var",
    "/usr",
    "/boot",
    "/bin",
    "/sbin",
    "/lib",
    "/lib64",
    "/opt",
    "~",
    "~/",
    ".",
    "..",
    "*",
    "c:\\",
    "c:/",
    "c:",
    "d:\\",
    "d:/",
    "d:",
    "\\",
    "\\\\",
}


def _is_dangerous_rm(args: list[str]) -> bool:
    """rm -rf-style invocation pointed at a protected root?

    Considered dangerous if recursive AND any operand resolves to a protected
    root path (system dirs, user home, drive roots, ".", "..", "*").
    """
    flags_joined = " ".join(a for a in args if a.startswith("-"))
    has_recursive = any(f in flags_joined for f in ("r", "R")) or "--recursive" in args
    if not has_recursive:
        return False
    for a in args[1:]:
        if a.startswith("-"):
            continue
        candidate = os.path.normpath(os.path.expanduser(a)).lower()
        if candidate in _PROTECTED_ROOTS or a.strip() in _PROTECTED_ROOTS:
            return True
        # Drive-root patterns on Windows like "C:\" "D:\"
        if len(candidate) <= 3 and candidate.endswith((":\\", ":/")):
            return True
    return False


async def _shell_exec(command: str, timeout: int = 30, cwd: str = "", **kwargs) -> str:
    """Execute a shell command safely using argument list form.

    Note: NOT a real shell — pipelines, redirections, and shell globbing are not
    interpreted. Use individual commands. Dangerous binaries are blocked.

    Set cwd to change the working directory for the command.
    """
    if len(command) > 4000:
        return "Error: command too long (>4000 chars)"
    try:
        args = shlex.split(command, posix=os.name != "nt")
    except ValueError as e:
        return f"Invalid command syntax: {e}"
    if not args:
        return "Empty command."

    base = os.path.basename(args[0]).lower().removesuffix(".exe")
    if base in _BLOCKED_COMMANDS:
        return f"Blocked: '{base}' is not allowed for security reasons."

    if base == "rm" and _is_dangerous_rm(args):
        return "Blocked: refusing recursive deletion of a protected path"

    # Block shell metacharacter injection attempts when used as plain args
    for a in args[1:]:
        if any(meta in a for meta in (";", "&&", "||", "`", "$(")):
            return f"Blocked: shell metacharacter in argument: {a!r}"

    # Resolve working directory
    work_dir: str | None = None
    if cwd:
        work_dir = os.path.expanduser(cwd)
        if not os.path.isdir(work_dir):
            return f"Error: cwd is not a directory: {cwd}"

    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=work_dir,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return f"Command timed out after {timeout}s"

        stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
        stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""

        parts = [f"[cwd: {work_dir or os.getcwd()}]"]
        if stdout:
            parts.append(_truncate(stdout, _MAX_SHELL_OUTPUT, "stdout"))
        if stderr:
            parts.append("STDERR:\n" + _truncate(stderr, 5000, "stderr"))
        if proc.returncode != 0 and proc.returncode is not None:
            parts.append(f"[exit code: {proc.returncode}]")
        return "\n".join(parts) if len(parts) > 1 else "Command completed with no output."
    except FileNotFoundError:
        return f"Command not found: {args[0]}"
    except OSError as e:
        return f"Command not executable: {e}"
    except Exception as e:
        return f"Error executing command: {e}"


# -- HTTP Tools --

_http_client: httpx.AsyncClient | None = None

# Allow override via env var: WA_ALLOW_PRIVATE_NET=1 to disable SSRF guard.
_ALLOW_PRIVATE_NET = os.environ.get("WA_ALLOW_PRIVATE_NET", "0") == "1"


async def _get_http() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        import httpx as _httpx

        _http_client = _httpx.AsyncClient(
            timeout=_httpx.Timeout(30, connect=10),
            limits=_httpx.Limits(max_keepalive_connections=10, max_connections=20),
            follow_redirects=True,
            max_redirects=10,
            headers={"User-Agent": "WeatherAgents/1.0"},
        )
    return _http_client


def _validate_url(url: str) -> str | None:
    """Return None if URL is safe; otherwise an error string.

    Blocks: non-http(s) schemes, private/loopback/link-local/unspecified/
    multicast IPs, IMDS endpoint, file:// scheme. Override with
    WA_ALLOW_PRIVATE_NET=1.

    Hardened against IPv4 short-form bypasses (``127.1``, ``0x7f000001``,
    ``2130706433``, ``0177.0.0.1``, ``0``) — ``ipaddress.ip_address`` rejects
    these, but ``socket.inet_aton`` (used by getaddrinfo internally) accepts
    them and httpx would happily connect. Trailing-dot variants of
    ``localhost.`` are normalised too.

    Limitation: DNS rebinding (a public hostname that resolves to a private
    IP at request time) is NOT blocked here — fix at the socket layer or
    require WA_ALLOW_PRIVATE_NET=1 in trusted environments.
    """
    import socket

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"Error: only http/https URLs allowed (got {parsed.scheme!r})"
    if not parsed.netloc:
        return f"Error: Invalid URL: {url}"
    if _ALLOW_PRIVATE_NET:
        return None
    host = parsed.hostname or ""
    # Strip a trailing dot — DNS treats ``localhost.`` and ``localhost`` the
    # same, but a substring/equality check on the raw hostname would not.
    norm = host.lower().rstrip(".")
    if norm in {"localhost", "ip6-localhost", "metadata.google.internal"}:
        return f"Error: refusing to reach internal host {host!r} (set WA_ALLOW_PRIVATE_NET=1 to override)"

    # Try strict IPv4/IPv6 first; if that fails, try ``inet_aton`` to catch
    # short / octal / hex / integer forms (127.1, 0x7f000001, 2130706433).
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address | None = None
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        try:
            packed = socket.inet_aton(host)
            ip = ipaddress.IPv4Address(socket.inet_ntoa(packed))
        except OSError:
            return None  # genuine hostname — relies on caller's DNS trust
    if (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_unspecified
        or ip.is_multicast
    ):
        return f"Error: refusing to reach private/loopback IP {ip} (set WA_ALLOW_PRIVATE_NET=1 to override)"
    return None


async def _http_get(url: str, **kwargs) -> str:
    if err := _validate_url(url):
        return err
    import httpx

    try:
        client = await _get_http()
        resp = await client.get(url)
        return f"Status: {resp.status_code}\n" + _truncate(resp.text, _MAX_SHELL_OUTPUT, "body")
    except httpx.TimeoutException:
        return "Error: request timed out"
    except httpx.RequestError as e:
        return f"Error: {e}"


async def _http_post(url: str, data: str = "", **kwargs) -> str:
    if err := _validate_url(url):
        return err
    import httpx

    try:
        client = await _get_http()
        headers = {}
        if data.strip().startswith(("{", "[")):
            headers["Content-Type"] = "application/json"
        resp = await client.post(url, content=data, headers=headers)
        return f"Status: {resp.status_code}\n" + _truncate(resp.text, _MAX_SHELL_OUTPUT, "body")
    except httpx.TimeoutException:
        return "Error: request timed out"
    except httpx.RequestError as e:
        return f"Error: {e}"


# -- Web Search (ddgs library with HTML fallback) --

_web_search_timestamps: list[float] = []
_WEB_SEARCH_MAX_PER_SEC = 2  # throttle to avoid rate-limiting


async def _web_search(query: str, num_results: int = 5, **kwargs) -> str:
    """Search the web using available backends (DuckDuckGo, Bing). Rate-limited."""
    now = time.monotonic()
    global _web_search_timestamps
    _web_search_timestamps = [t for t in _web_search_timestamps if now - t < 1.0]
    if len(_web_search_timestamps) >= _WEB_SEARCH_MAX_PER_SEC:
        wait = 1.0 - (now - _web_search_timestamps[0]) + 0.05
        if wait > 0:
            await asyncio.sleep(wait)

    results: list[dict] | None = None
    errors: list[str] = []

    # Try DuckDuckGo first. The ddgs library is synchronous and its
    # internal HTTP retries against yahoo/bing backends can stall for
    # tens of seconds when those endpoints misbehave — exactly the
    # "agent suddenly stopped" symptom seen by users. asyncio.wait_for
    # gives the whole search a hard wall-clock cap so the chat loop can
    # always recover, while _ddg_api_search itself uses asyncio.to_thread
    # to keep the event loop responsive during the blocking call.
    try:
        results = await asyncio.wait_for(_ddg_api_search(query, min(num_results, 10)), timeout=12.0)
    except Exception as e:
        errors.append(f"ddgs: {e}")
        try:
            results = await asyncio.wait_for(
                _ddg_html_fallback(query, min(num_results, 10)), timeout=10.0
            )
        except Exception as e2:
            errors.append(f"ddg-html: {e2}")

    # Try Bing if DDG failed and key is available
    if not results:
        bing_key = os.environ.get("BING_API_KEY")
        if bing_key:
            try:
                results = await asyncio.wait_for(
                    _bing_search(query, min(num_results, 10), bing_key), timeout=10.0
                )
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
    return "\n".join(output_parts)


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
            from urllib.parse import parse_qs, unquote

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
        headers={"User-Agent": "Mozilla/5.0 (compatible; WeatherAgents/1.0)"},
    )
    if resp.status_code != 200:
        return []
    return _parse_ddg_html(resp.text, num_results)


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


async def _lint_python_file(path: str) -> str:
    """AST-walk a Python file for common review smells.

    Cheap heuristic linter — flags bare ``except``, residual ``print()``
    debug calls, and ``eval`` / ``exec`` usage. Not a replacement for a
    full linter; intended for inline review inside an agent loop where
    spinning up ruff/mypy would be overkill.
    """
    import ast

    expanded = os.path.expanduser(path)
    if not os.path.isfile(expanded):
        return f"File not found: {path}"

    try:
        with open(expanded, encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=expanded)
    except SyntaxError as e:
        return f"Syntax error at line {e.lineno}: {e.msg}"

    issues: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            issues.append(f"[WARN] line {node.lineno}: bare except clause")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ):
            issues.append(f"[INFO] line {node.lineno}: print() call — verify if intentional")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("eval", "exec")
        ):
            issues.append(
                f"[CRITICAL] line {node.lineno}: {node.func.id}() — potential code injection"
            )

    if not issues:
        return f"Lint passed for {path}. No issues detected."
    return "\n".join(issues)


async def _scan_python_deps(directory: str = ".") -> str:
    """Surface Python dependency files and run pip-audit if available.

    Best-effort — returns a textual summary the LLM can read. No
    structured output, since the agent layer cleans this up anyway.
    """
    import asyncio
    import subprocess

    expanded = os.path.expanduser(directory)
    if not os.path.isdir(expanded):
        return f"Directory not found: {directory}"

    results: list[str] = []
    req_path = os.path.join(expanded, "requirements.txt")
    if os.path.isfile(req_path):
        results.append("[requirements.txt] Dependency audit:")
        try:
            with open(req_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        results.append(f"  - {line}")
        except OSError:
            results.append("  (could not read)")

    pyproj = os.path.join(expanded, "pyproject.toml")
    if os.path.isfile(pyproj):
        results.append("[pyproject.toml] Found — run `pip-audit` for full scan")

    # pip-audit pass: optional, skipped if not installed.
    try:
        proc = await asyncio.create_subprocess_exec(
            "pip-audit",
            "--path",
            expanded,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            results.append("[pip-audit] timed out after 30s")
        else:
            stdout = stdout_bytes.decode(errors="replace")
            rc = proc.returncode or 0
            if rc == 0:
                results.append("[pip-audit] No known vulnerabilities found.")
            else:
                # Tail to keep the snippet bounded — full output is rarely useful.
                results.append(f"[pip-audit]\n{stdout[-500:]}")
        _ = subprocess  # silence unused-import lint when subprocess fallback unused
    except FileNotFoundError:
        results.append("[pip-audit] Not installed. Install with: pip install pip-audit")
    except Exception:
        pass

    if not results:
        return f"No Python dependency files found in {directory}."
    return "\n".join(results)


async def _fetch_web_page(url: str, extract_text: bool = True) -> str:
    """Download a web page and (optionally) strip HTML chrome.

    Lives next to web_search / http_get in the builtins — same
    permission/audit story as those, plus a cheap regex-based text
    extraction so the LLM doesn't have to parse raw HTML.
    """
    import httpx

    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(
                url,
                headers={"User-Agent": "WeatherAgents/1.0 WebResearcher"},
            )
            if resp.status_code != 200:
                return f"HTTP {resp.status_code}: Could not fetch {url}"

            if not extract_text:
                return resp.text[:5000]

            html = resp.text
            html = _re.sub(
                r"<(script|style|noscript|iframe|svg)[^>]*>.*?</\1>",
                "",
                html,
                flags=_re.DOTALL | _re.IGNORECASE,
            )
            html = _re.sub(r"<[^>]+>", " ", html)
            html = _re.sub(r"\s+", " ", html).strip()
            html = (
                html.replace("&amp;", "&")
                .replace("&lt;", "<")
                .replace("&gt;", ">")
                .replace("&quot;", '"')
                .replace("&#39;", "'")
                .replace("&nbsp;", " ")
            )
            return html[:3000] if len(html) > 3000 else html
    except Exception as e:
        return f"Error fetching {url}: {e}"


# -- Register all tools --

_registered = False


def register_builtin_tools(registry: ToolRegistry | None = None) -> None:
    global _registered
    if _registered and registry is None:
        return
    reg = registry or ToolRegistry()

    def _read_file_cache_extra(kw: dict) -> str:
        """Mix file mtime into the read_file cache key.

        Without this, two consecutive read_file calls with the same args
        but a file that changed between calls would return the cached
        first result — silently feeding the LLM stale content. With
        mtime in the key, any disk change invalidates the entry.
        ``missing`` is a sentinel that never matches a successful read.
        """
        path = kw.get("path", "")
        if not isinstance(path, str) or not path:
            return "no-path"
        try:
            return str(os.stat(os.path.expanduser(path)).st_mtime_ns)
        except OSError:
            return "missing"

    tools = [
        Tool(
            name="read_file",
            description="Read a text file (max 50KB). Use offset/limit to read specific line ranges.",
            parameters=[
                ToolParameter(name="path", type="string", description="File path to read"),
                ToolParameter(
                    name="offset",
                    type="integer",
                    description="Start line (0-based, default 0)",
                    required=False,
                    default=0,
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="Max lines to read (0 = all, default 0)",
                    required=False,
                    default=0,
                ),
            ],
            handler=_read_file,
            cache_key_extra=_read_file_cache_extra,
        ),
        Tool(
            name="write_file",
            description="Write content to a file, creating parent directories if needed. Refuses to write to system-protected paths.",
            parameters=[
                ToolParameter(name="path", type="string", description="File path to write"),
                ToolParameter(name="content", type="string", description="Content to write"),
            ],
            handler=_write_file,
            dangerous=True,
        ),
        Tool(
            name="edit_file",
            description="Edit a file by replacing old_text with new_text (optionally specify count for multiple occurrences)",
            parameters=[
                ToolParameter(name="path", type="string", description="File path to edit"),
                ToolParameter(name="old_text", type="string", description="Text to find"),
                ToolParameter(name="new_text", type="string", description="Replacement text"),
                ToolParameter(
                    name="count",
                    type="integer",
                    description="Number of occurrences to replace (default 1)",
                    required=False,
                    default=1,
                ),
            ],
            handler=_edit_file,
            dangerous=True,
        ),
        Tool(
            name="file_search",
            description="Search for files matching a glob pattern recursively. Use max_depth to limit recursion.",
            parameters=[
                ToolParameter(
                    name="directory", type="string", description="Directory to search in"
                ),
                ToolParameter(name="pattern", type="string", description="Glob pattern to match"),
                ToolParameter(
                    name="max_depth",
                    type="integer",
                    description="Max recursion depth (0 = unlimited, default 10)",
                    required=False,
                    default=0,
                ),
            ],
            handler=_file_search,
        ),
        Tool(
            name="code_search",
            description="Search for text or regex in source files (max 5000 files scanned, 100 matches). Set regex=true for regex mode.",
            parameters=[
                ToolParameter(name="directory", type="string", description="Directory to search"),
                ToolParameter(
                    name="query", type="string", description="Search query (text or regex)"
                ),
                ToolParameter(
                    name="regex",
                    type="boolean",
                    description="Treat query as a regex (default false)",
                    required=False,
                    default=False,
                ),
            ],
            handler=_code_search,
        ),
        Tool(
            name="grep",
            description="Search text/regex in ALL text files (no hardcoded extension limit). Supports glob, regex, ignore_case, context lines.",
            parameters=[
                ToolParameter(
                    name="directory", type="string", description="Directory to search in"
                ),
                ToolParameter(
                    name="pattern", type="string", description="Text or regex pattern to search for"
                ),
                ToolParameter(
                    name="glob",
                    type="string",
                    description="Optional file name pattern filter (e.g. '*.py', '*.md')",
                    required=False,
                    default="",
                ),
                ToolParameter(
                    name="regex",
                    type="boolean",
                    description="Treat pattern as regex (default false)",
                    required=False,
                    default=False,
                ),
                ToolParameter(
                    name="ignore_case",
                    type="boolean",
                    description="Case-insensitive search (default false)",
                    required=False,
                    default=False,
                ),
                ToolParameter(
                    name="context_around",
                    type="integer",
                    description="Lines to show before and after each match",
                    required=False,
                    default=0,
                ),
                ToolParameter(
                    name="context_before",
                    type="integer",
                    description="Lines to show before each match",
                    required=False,
                    default=0,
                ),
                ToolParameter(
                    name="context_after",
                    type="integer",
                    description="Lines to show after each match",
                    required=False,
                    default=0,
                ),
            ],
            handler=_grep,
        ),
        Tool(
            name="git_status",
            description="Show the working tree status (git status --porcelain --branch)",
            parameters=[
                ToolParameter(
                    name="repo",
                    type="string",
                    description="Repository directory (default: '.')",
                    required=False,
                    default=".",
                ),
            ],
            handler=_git_status,
        ),
        Tool(
            name="git_diff",
            description="Show changes (git diff). Set staged=true for staged changes, path to diff a single file.",
            parameters=[
                ToolParameter(
                    name="staged",
                    type="boolean",
                    description="Show staged changes (default false)",
                    required=False,
                    default=False,
                ),
                ToolParameter(
                    name="path",
                    type="string",
                    description="Limit diff to a specific file or directory",
                    required=False,
                    default="",
                ),
                ToolParameter(
                    name="repo",
                    type="string",
                    description="Repository directory (default: '.')",
                    required=False,
                    default=".",
                ),
            ],
            handler=_git_diff,
        ),
        Tool(
            name="git_log",
            description="Show commit history (git log)",
            parameters=[
                ToolParameter(
                    name="count",
                    type="integer",
                    description="Number of recent commits to show (default 10, max 50)",
                    required=False,
                    default=10,
                ),
                ToolParameter(
                    name="oneline",
                    type="boolean",
                    description="Compact one-line format (default true)",
                    required=False,
                    default=True,
                ),
                ToolParameter(
                    name="all_branches",
                    type="boolean",
                    description="Include all branches (default false)",
                    required=False,
                    default=False,
                ),
                ToolParameter(
                    name="repo",
                    type="string",
                    description="Repository directory (default: '.')",
                    required=False,
                    default=".",
                ),
            ],
            handler=_git_log,
        ),
        Tool(
            name="git_add",
            description="Stage files for commit. Accepts space-separated paths. Refuses -A/--all for safety.",
            parameters=[
                ToolParameter(
                    name="files",
                    type="string",
                    description="Space-separated file paths to stage",
                ),
                ToolParameter(
                    name="repo",
                    type="string",
                    description="Repository directory (default: '.')",
                    required=False,
                    default=".",
                ),
            ],
            handler=_git_add,
            dangerous=True,
        ),
        Tool(
            name="git_commit",
            description="Create a commit with the given message",
            parameters=[
                ToolParameter(
                    name="message",
                    type="string",
                    description="Commit message",
                ),
                ToolParameter(
                    name="repo",
                    type="string",
                    description="Repository directory (default: '.')",
                    required=False,
                    default=".",
                ),
            ],
            handler=_git_commit,
            dangerous=True,
        ),
        Tool(
            name="git_checkout",
            description="Switch to a branch. Set create=true to create a new branch first.",
            parameters=[
                ToolParameter(
                    name="branch",
                    type="string",
                    description="Branch name",
                ),
                ToolParameter(
                    name="create",
                    type="boolean",
                    description="Create the branch before switching (default false)",
                    required=False,
                    default=False,
                ),
                ToolParameter(
                    name="repo",
                    type="string",
                    description="Repository directory (default: '.')",
                    required=False,
                    default=".",
                ),
            ],
            handler=_git_checkout,
            dangerous=True,
        ),
        Tool(
            name="shell_exec",
            description="Run a single command. Dangerous binaries are blocked. Set cwd to change working directory.",
            parameters=[
                ToolParameter(
                    name="command", type="string", description="Shell command to execute"
                ),
                ToolParameter(
                    name="timeout",
                    type="number",
                    description="Timeout in seconds (default 30)",
                    required=False,
                    default=30,
                ),
                ToolParameter(
                    name="cwd",
                    type="string",
                    description="Working directory for the command",
                    required=False,
                    default="",
                ),
            ],
            handler=_shell_exec,
            dangerous=True,
        ),
        Tool(
            name="http_get",
            description=(
                "Fetch a web page or HTTP URL (GET request). Use this to "
                "download article content, read web pages, fetch HTML, or "
                "retrieve any URL. Returns status + body. Max 10 redirects."
            ),
            parameters=[
                ToolParameter(name="url", type="string", description="URL to request"),
            ],
            handler=_http_get,
        ),
        Tool(
            name="http_post",
            description="Make an HTTP POST request with optional JSON body",
            parameters=[
                ToolParameter(name="url", type="string", description="URL to post to"),
                ToolParameter(
                    name="data",
                    type="string",
                    description="Request body (JSON auto-detected)",
                    required=False,
                    default="",
                ),
            ],
            handler=_http_post,
        ),
        Tool(
            name="web_search",
            description="Search the web using DuckDuckGo (rate-limited: 2/sec, max 10 results)",
            parameters=[
                ToolParameter(name="query", type="string", description="Search query"),
                ToolParameter(
                    name="num_results",
                    type="number",
                    description="Number of results (max 10, default 5)",
                    required=False,
                    default=5,
                ),
            ],
            handler=_web_search,
        ),
        Tool(
            name="list_directory",
            description="List files and directories with sizes. Set include_hidden=true to show hidden entries.",
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Directory path (default: '.')",
                    required=False,
                    default=".",
                ),
                ToolParameter(
                    name="include_hidden",
                    type="boolean",
                    description="Show hidden files/dirs (default false)",
                    required=False,
                    default=False,
                ),
            ],
            handler=_list_directory,
        ),
        Tool(
            name="tree",
            description="Show directory tree structure. Set include_hidden=true to show hidden entries.",
            parameters=[
                ToolParameter(
                    name="directory",
                    type="string",
                    description="Root directory (default: '.')",
                    required=False,
                    default=".",
                ),
                ToolParameter(
                    name="max_depth",
                    type="integer",
                    description="Max depth (default: 3)",
                    required=False,
                    default=3,
                ),
                ToolParameter(
                    name="include_hidden",
                    type="boolean",
                    description="Show hidden files/dirs (default false)",
                    required=False,
                    default=False,
                ),
            ],
            handler=_tree,
        ),
        Tool(
            name="move_file",
            description="Move or rename a file or directory",
            parameters=[
                ToolParameter(name="src", type="string", description="Source path"),
                ToolParameter(name="dst", type="string", description="Destination path"),
            ],
            handler=_move_file,
            dangerous=True,
        ),
        Tool(
            name="copy_file",
            description="Copy a file or directory tree",
            parameters=[
                ToolParameter(name="src", type="string", description="Source path"),
                ToolParameter(name="dst", type="string", description="Destination path"),
            ],
            handler=_copy_file,
            dangerous=True,
        ),
        Tool(
            name="delete_file",
            description="Delete a file or empty directory (non-recursive, protected paths refused)",
            parameters=[
                ToolParameter(
                    name="path", type="string", description="File or directory to delete"
                ),
            ],
            handler=_delete_file,
            dangerous=True,
        ),
        Tool(
            name="get_cwd",
            description="Get the current working directory path",
            parameters=[],
            handler=_get_cwd,
        ),
        Tool(
            name="task_done",
            description="Signal that the task is complete. Call this when you have fully satisfied the user's request. Do NOT use this if the user's message is a simple greeting or question that doesn't need multiple steps.",
            parameters=[
                ToolParameter(
                    name="summary",
                    type="string",
                    description="Brief summary of what was accomplished",
                ),
            ],
            handler=_task_done,
        ),
        # Promoted from skill-injected to always-available. The
        # code_reviewer / security_auditor / web_research SKILL.md files
        # reference these by name.
        Tool(
            name="lint_file",
            description=(
                "Run static analysis on a Python file to detect common issues "
                "(bare except, eval, print calls, etc.). Lives in code_reviewer's "
                "toolkit but is always available."
            ),
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Path to the Python file to lint",
                    required=True,
                ),
            ],
            handler=_lint_python_file,
        ),
        Tool(
            name="scan_deps",
            description=(
                "Scan Python dependencies for known vulnerabilities. Checks "
                "requirements.txt and runs pip-audit if available. Used by the "
                "security_auditor skill."
            ),
            parameters=[
                ToolParameter(
                    name="directory",
                    type="string",
                    description="Project directory to scan (default: current directory)",
                    required=False,
                ),
            ],
            handler=_scan_python_deps,
        ),
        Tool(
            name="fetch_page",
            description=(
                "Fetch a web page and (optionally) extract its visible text. "
                "Strips HTML tags, scripts, and styles. Used by the web_research "
                "skill to read past search-result snippets."
            ),
            parameters=[
                ToolParameter(
                    name="url",
                    type="string",
                    description="URL of the page to fetch",
                    required=True,
                ),
                ToolParameter(
                    name="extract_text",
                    type="boolean",
                    description="Extract visible text from HTML? (default: true)",
                    required=False,
                ),
            ],
            handler=_fetch_web_page,
        ),
    ]

    for tool in tools:
        reg.register(tool)


async def close_http_client() -> None:
    """Close the shared httpx client. Called on shutdown to free connections."""
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None

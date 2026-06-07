"""Git tools — status / diff / log / add / commit / checkout.

Extracted from ``builtin.py``. Imports ``_truncate`` + ``_MAX_SHELL_OUTPUT``
from ``_common.py`` so the handlers stay self-contained.
"""

from __future__ import annotations

import asyncio
import os
import shlex

from weather_agents.tools._common import _MAX_SHELL_OUTPUT, _truncate

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

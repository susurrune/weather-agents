"""Workspace auto-detection and lifecycle management.

Rules:
- First launch: auto-select best drive, create ``workspace/``
- Multi-drive: skip C:, pick drive with most free space
- Single drive (C: only): use C:\\workspace
- Unix: use ~/workspace
- User can override via config ``workspace.path``
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

_WORKSPACE_SUBDIRS = ["files", "output", "temp"]


@dataclass
class DriveInfo:
    letter: str
    path: str
    total_bytes: int
    free_bytes: int


def _get_drive_list() -> list[DriveInfo]:
    """Enumerate fixed drives with free-space info."""
    drives: list[DriveInfo] = []
    if os.name == "nt":
        import string

        for letter in string.ascii_uppercase:
            root = f"{letter}:\\"
            if not os.path.exists(root):
                continue
            try:
                usage = shutil.disk_usage(root)
                drives.append(
                    DriveInfo(
                        letter=letter,
                        path=root,
                        total_bytes=usage.total,
                        free_bytes=usage.free,
                    )
                )
            except OSError:
                continue
    else:
        # Unix — treat / as the single candidate
        try:
            usage = shutil.disk_usage("/")
            drives.append(
                DriveInfo(letter="", path="/", total_bytes=usage.total, free_bytes=usage.free)
            )
        except OSError:
            pass
    return drives


def detect_best_workspace_root() -> Path:
    """Pick the best drive for the workspace directory.

    * Windows with multiple drives: skip C:, pick drive with most free bytes.
    * Windows with only C: → ``C:\\workspace``.
    * Unix → ``~/workspace``.
    """
    drives = _get_drive_list()

    if os.name == "nt":
        candidates = [d for d in drives if d.letter.upper() != "C"]
        if not candidates:
            candidates = drives  # fallback to C:
        # Sort by free space descending
        candidates.sort(key=lambda d: d.free_bytes, reverse=True)
        best = candidates[0]
        return Path(best.path) / "workspace"
    else:
        return Path.home() / "workspace"


def resolve_workspace_path(config_value: str) -> Path:
    """Resolve workspace path from config.

    - ``"auto"`` → call :func:`detect_best_workspace_root`
    - explicit path → expand ``~`` and return
    """
    if config_value.lower() == "auto":
        return detect_best_workspace_root()
    return Path(os.path.expanduser(config_value)).resolve()


def init_workspace(root: Path) -> Path:
    """Create workspace directory tree on first use. Idempotent.

    Creates::

        workspace/
        ├── files/    # agent-generated files
        ├── output/   # task results, exports
        └── temp/     # scratch / ephemeral
    """
    root.mkdir(parents=True, exist_ok=True)
    for sub in _WORKSPACE_SUBDIRS:
        (root / sub).mkdir(exist_ok=True)
    # Touch a .workspace marker so tools can identify it
    marker = root / ".workspace"
    if not marker.exists():
        marker.write_text(f"# Skyloom workspace — created automatically\npath: {root}\n")
    return root


def format_bytes(n: int) -> str:
    """Human-readable byte count (e.g. 128.5 GB)."""
    val: float = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(val) < 1024.0:
            return f"{val:.1f} {unit}"
        val /= 1024.0
    return f"{val:.1f} PB"

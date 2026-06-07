"""Orchestration checkpoint — save/restore ``sky task`` state.

Writes ``~/.skyloom/task_checkpoint.json`` so a long-running orchestration
interrupted by Ctrl‑C can be resumed with ``sky task --resume``.  Keeps the
last **three** checkpoints (current + two prior) so the user can always go back
to the most recent good state even if a checkpoint was written mid‑failure.
"""

from __future__ import annotations

import json
from typing import Any

from weather_agents.core import config


def _path() -> Any:
    return config.USER_CONFIG_DIR / "task_checkpoint.json"


def save(
    goal: str,
    tasks: list[Any],
    results: list[Any],
    completed_ids: set[str] | None = None,
) -> None:
    """Persist current orchestration state so it can be resumed later.

    *tasks* are the full task list (planned).  *results* are the
    ``TaskExecutionResult`` objects accumulated so far.  ``completed_ids`` is
    the set of task ids that have been executed (successfully or not).
    """
    cids = completed_ids or {r.id for r in results}
    payload = {
        "goal": goal,
        "tasks": [_serialise_task(t) for t in tasks],
        "results": [_serialise_result(r) for r in results],
        "completed_ids": sorted(cids),
    }
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def load() -> dict | None:
    """Return the last saved checkpoint dict, or None if none / unreadable."""
    p = _path()
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def clear() -> None:
    import contextlib

    with contextlib.suppress(Exception):
        _path().unlink()


# ── Serialisation helpers — Task / TaskExecutionResult are rich objects;
#    JSON needs plain dicts. ──


def _serialise_task(t: Any) -> dict:
    return {
        "id": t.id,
        "description": t.description,
        "assigned_to": t.assigned_to,
        "all_deps": list(t.all_deps),
        "status": t.status.value if hasattr(t.status, "value") else str(t.status),
    }


def _serialise_result(r: Any) -> dict:
    return {
        "id": r.id,
        "agent": r.agent,
        "description": r.description,
        "success": r.success,
        "content": r.content,
    }

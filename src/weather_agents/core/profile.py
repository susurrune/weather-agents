"""Local user profile (用户画像) + per-agent custom personas.

Everything here lives under ``~/.skyloom/`` and never leaves the machine:

- ``profile.json`` — a free-form dict of facts about the user (name, how they
  like to be addressed, preferences, background…). Shared by every agent, since
  it's the same person, and injected into each agent's prompt so they remember
  the user across sessions.
- ``personas/<agent>.md`` — an optional custom role for a specific agent. When
  present it *replaces* that agent's built-in persona, letting the user (or the
  agent itself, via the ``set_persona`` tool) redefine who the agent is.

All reads reference ``config.USER_CONFIG_DIR`` lazily so tests that redirect it
(and the first-run config-dir migration) are respected.
"""

from __future__ import annotations

import contextlib
import json

from weather_agents.core import config

_VALID_AGENTS = {"fog", "rain", "frost", "snow", "dew", "fair"}


def _profile_path():
    return config.USER_CONFIG_DIR / "profile.json"


def _persona_path(agent: str):
    return config.USER_CONFIG_DIR / "personas" / f"{agent}.md"


# ── User profile ───────────────────────────────────────────────────────────


def load_profile() -> dict:
    """Return the user profile dict (empty if none / unreadable)."""
    p = _profile_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_profile(data: dict) -> None:
    p = _profile_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def set_profile_field(key: str, value: str) -> None:
    key = (key or "").strip()
    if not key:
        return
    data = load_profile()
    data[key] = value
    save_profile(data)


def clear_profile_field(key: str | None = None) -> None:
    """Remove one field, or wipe the whole profile when ``key`` is None."""
    if key is None:
        with contextlib.suppress(Exception):
            _profile_path().unlink()
        return
    data = load_profile()
    data.pop(key, None)
    save_profile(data)


def format_profile_for_prompt(lang: str = "zh") -> str:
    """Render the profile as a short prompt block, or '' when empty."""
    data = load_profile()
    if not data:
        return ""
    lines = [f"- {k}：{v}" if lang != "en" else f"- {k}: {v}" for k, v in data.items()]
    body = "\n".join(lines)
    if lang == "en":
        return "\n\n## About the user (remember this and use it naturally)\n" + body
    return "\n\n## 关于用户（记住，并在对话中自然运用，不要生硬复述）\n" + body


# ── Per-agent custom persona ─────────────────────────────────────────────


def load_persona(agent: str) -> str | None:
    """Return the user-defined persona for ``agent``, or None if unset."""
    p = _persona_path(agent)
    if not p.exists():
        return None
    try:
        text = p.read_text(encoding="utf-8").strip()
        return text or None
    except Exception:
        return None


def save_persona(agent: str, text: str) -> bool:
    """Persist a custom persona for ``agent``. Returns False for unknown agents."""
    if agent not in _VALID_AGENTS:
        return False
    p = _persona_path(agent)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.strip() + "\n", encoding="utf-8")
    return True


def clear_persona(agent: str) -> None:
    """Drop the custom persona, reverting ``agent`` to its built-in default."""
    with contextlib.suppress(Exception):
        _persona_path(agent).unlink()

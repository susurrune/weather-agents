"""Agent icon system — dynamic status indicators.

Static decorative icons are removed. Each agent is identified by its
display name with Rich color styling. During processing, the agent
spinners (defined in main.py AGENT_SPINNERS) provide dynamic status.
"""

from __future__ import annotations

from pathlib import Path

_ICONS_DIR = Path(__file__).resolve().parent.parent / "assets" / "icons"

AGENT_COLOR_MAP: dict[str, str] = {
    "fog": "bright_white",
    "rain": "blue",
    "frost": "cyan",
    "snow": "bright_white",
    "dew": "green",
    "fair": "#FFD700",
}


def svg_path(name: str) -> str:
    """Return the filesystem path to an agent's SVG icon file."""
    return str(_ICONS_DIR / f"{name}.svg")


def icon_text(name: str) -> str:
    """Return the plain-text icon string (used in system prompts / logs)."""
    return {
        "fog": "~",
        "rain": "/",
        "frost": "+",
        "snow": "·",
        "dew": ",",
        "fair": "*",
    }.get(name, name)

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

# Public alias — use this, not the raw dict, for stable API.
AGENT_COLORS = AGENT_COLOR_MAP

# Agent icon glyph map — used for tests and any dict-based lookup.
AGENT_EMOJI: dict[str, str] = {
    "fog": "≋",
    "rain": "╱",
    "frost": "✱",
    "snow": "❉",
    "dew": "∘",
    "fair": "☼",
}


def svg_path(name: str) -> str:
    """Return the filesystem path to an agent's SVG icon file."""
    return str(_ICONS_DIR / f"{name}.svg")


def icon_text(name: str) -> str:
    """Return the plain-text icon for an agent.

    Used in dashboards, prompts, logs, and any UI where SVG can't render.
    The glyphs are deliberately chosen from Unicode blocks that render
    as monochrome text on virtually every terminal (Math, Box-Drawing,
    Miscellaneous Symbols, older Dingbats subset) — NOT from "Symbols
    and Pictographs" which would force colored emoji presentation.

    Each glyph is metaphorically tied to its agent:
      fog  ≋  three wavy lines — drifting mist
      rain ╱  slanted line — falling raindrop
      frost ✱  pointed asterisk — frost crystal
      snow ❉  balloon-spoked star — snowflake
      dew  ∘  ring — dewdrop
      fair ☼  sun with rays — clear sky
    """
    return {
        "fog": "≋",
        "rain": "╱",
        "frost": "✱",
        "snow": "❉",
        "dew": "∘",
        "fair": "☼",
    }.get(name, name)

"""Interactive mode controller — single source of truth for plan/auto/default.

Before this module, ``INTERACTIVE_MODE`` was a bare module-level string in
cli/main.py, mutated from three different code paths, never persisted, and
not type-safe. This consolidates:

  - the enum of valid modes,
  - load-from-config / save-to-config,
  - the cycle order used by Shift+Tab,
  - human-readable labels for the prompt UI.

The controller is the only place that touches user config for this concern,
so adding a fourth mode in the future is a one-file change.
"""

from __future__ import annotations

from enum import StrEnum

from weather_agents.core.config import _save_user_cfg, load_config


class InteractiveMode(StrEnum):
    """User-facing interactive modes for `wa chat`.

    DEFAULT — router decides per-message: short questions answer directly,
              tool work runs autonomously. No human gate.
    PLAN    — every turn produces a plan first; user presses Enter to execute.
    AUTO    — model continues across turns on its own (legacy behavior).
    """

    DEFAULT = "default"
    PLAN = "plan"
    AUTO = "auto"

    @classmethod
    def parse(cls, value: str | None) -> InteractiveMode | None:
        if not value:
            return None
        try:
            return cls(value.strip().lower())
        except ValueError:
            return None


# UI cycle order used by Shift+Tab. Keep DEFAULT first so first-time users
# discover the smart mode before stumbling into PLAN's confirm-gate flow.
_CYCLE: tuple[InteractiveMode, ...] = (
    InteractiveMode.DEFAULT,
    InteractiveMode.PLAN,
    InteractiveMode.AUTO,
)


_LABEL_STYLE = {
    InteractiveMode.DEFAULT: ("DEFAULT", "bold cyan"),
    InteractiveMode.PLAN: ("PLAN", "bold magenta"),
    InteractiveMode.AUTO: ("AUTO", "bold yellow"),
}


_DESCRIPTIONS = {
    InteractiveMode.DEFAULT: "smart — router picks the right depth per message",
    InteractiveMode.PLAN: "plan first, confirm before execute",
    InteractiveMode.AUTO: "autonomous reasoning & execution",
}


class ModeController:
    """Owns the current interactive mode and its persistence."""

    def __init__(self, initial: InteractiveMode | None = None) -> None:
        self._mode = initial or self._load() or InteractiveMode.DEFAULT

    @property
    def current(self) -> InteractiveMode:
        return self._mode

    def set(self, mode: InteractiveMode, *, persist: bool = True) -> InteractiveMode:
        self._mode = mode
        if persist:
            self._save(mode)
        return mode

    def cycle(self) -> InteractiveMode:
        idx = _CYCLE.index(self._mode) if self._mode in _CYCLE else -1
        return self.set(_CYCLE[(idx + 1) % len(_CYCLE)])

    def label(self) -> tuple[str, str]:
        """Return (text, rich-style) tuple for prompt rendering."""
        return _LABEL_STYLE[self._mode]

    def describe(self) -> str:
        return _DESCRIPTIONS[self._mode]

    @staticmethod
    def _load() -> InteractiveMode | None:
        cfg = load_config()
        raw = getattr(getattr(cfg, "cli", None), "interactive_mode", None)
        return InteractiveMode.parse(raw)

    @staticmethod
    def _save(mode: InteractiveMode) -> None:
        # Delegate to the project's deep-merge helper. Earlier this method
        # did its own load+write, which silently dropped every other user
        # setting if the YAML was unreadable. _save_user_cfg merges instead.
        _save_user_cfg({"cli": {"interactive_mode": mode.value}})

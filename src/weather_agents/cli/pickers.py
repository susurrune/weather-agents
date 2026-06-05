"""Interactive selection widgets for the CLI.

The arrow-key picker is the shared "pick one from a known list" widget — used
by ``/skills``, ``/model``, ``/apikey``, and the setup wizard. Pulling it (and
the small model-catalog flattener it pairs with) out of ``cli/main.py`` lets the
wizard and the slash-command handlers depend on one widget module instead of on
the 4k-line main module — which is what makes a later wizard extraction
possible without an import cycle.
"""

from __future__ import annotations

import sys

from rich.live import Live
from rich.table import Table
from rich.text import Text

from weather_agents.cli.console import console
from weather_agents.cli.keys import get_key


def flatten_catalog(catalog: dict) -> list[tuple[str, str]]:
    """Return [(provider, model_name), ...] preserving provider order."""
    out = []
    for prov, models in catalog.items():
        for m in models:
            out.append((prov, m["name"]))
    return out


def arrow_pick_from_list(
    items: list[tuple[str, str]],
    title: str = "Select",
    *,
    active_keys: set[str] | None = None,
    viewport: int = 15,
) -> str | None:
    """Arrow-key driven picker over a flat ``(key, label)`` list.

    Up/Down move the cursor (with viewport scrolling when the list is
    longer than ``viewport`` rows). Typing letters / digits builds an
    incremental case-insensitive substring filter; Backspace pops the
    filter. Enter confirms, Esc cancels. Returns the selected ``key``
    or ``None`` on cancel.

    ``active_keys`` is an optional set of keys that should render with
    an active dot marker — used by /skills so the user sees which
    skills are already loaded while browsing.

    The picker is the standard interactive widget for any "pick one
    from a known list" command. _interactive_model_select was the
    first instance of this pattern in sky; this generalises it so
    /skills (and future commands like /agent / /session pick) can
    share the implementation.
    """
    if not items:
        return None
    if not sys.stdin.isatty():
        # Non-TTY (piped / test): no Live cursor possible — caller should
        # fall back to a numeric prompt path.
        return None

    active_keys = active_keys or set()
    filter_buf: list[str] = []
    cursor_idx = 0  # index into the filtered view
    scroll_top = 0

    with Live(
        Table(show_header=False, box=None, padding=0),
        console=console,
        refresh_per_second=20,
        transient=True,
    ) as live:
        while True:
            filt = "".join(filter_buf).lower()
            view = (
                [(k, lbl) for (k, lbl) in items if filt in k.lower() or filt in lbl.lower()]
                if filt
                else list(items)
            )
            if not view:
                cursor_idx = 0
            else:
                cursor_idx = max(0, min(cursor_idx, len(view) - 1))
                # Keep the cursor inside the visible viewport.
                if cursor_idx < scroll_top:
                    scroll_top = cursor_idx
                elif cursor_idx >= scroll_top + viewport:
                    scroll_top = cursor_idx - viewport + 1
            scroll_top = max(0, scroll_top)

            tbl = Table(show_header=False, box=None, padding=0, expand=True)
            tbl.add_column(ratio=1)

            header = Text()
            header.append(f"\n  {title}", style="bold")
            header.append("  (↑↓ select  enter confirm  esc cancel", style="dim")
            if filt:
                header.append(f"  filter: {''.join(filter_buf)}", style="cyan dim")
            else:
                header.append("  type to filter", style="dim")
            header.append(")", style="dim")
            tbl.add_row(header)
            tbl.add_row(Text())

            if not view:
                tbl.add_row(Text("  no matches", style="yellow"))
            else:
                visible = view[scroll_top : scroll_top + viewport]
                if scroll_top > 0:
                    tbl.add_row(Text(f"  ↑ {scroll_top} more above", style="dim"))
                for off, (k, lbl) in enumerate(visible):
                    abs_idx = scroll_top + off
                    is_cursor = abs_idx == cursor_idx
                    is_active = k in active_keys
                    marker = "❯" if is_cursor else " "
                    dot = "●" if is_active else "○"
                    line = Text()
                    line.append(f" {marker} ", style="bold cyan" if is_cursor else "")
                    line.append(
                        f"{dot} ",
                        style="green" if is_active else ("bold cyan" if is_cursor else "dim"),
                    )
                    line.append(lbl, style="bold cyan" if is_cursor else "")
                    tbl.add_row(line)
                tail = len(view) - (scroll_top + len(visible))
                if tail > 0:
                    tbl.add_row(Text(f"  ↓ {tail} more below", style="dim"))

            live.update(tbl)

            try:
                key = get_key()
            except KeyboardInterrupt:
                return None

            if key == "enter":
                if view:
                    return view[cursor_idx][0]
                continue
            if key == "esc":
                return None
            if key == "up":
                cursor_idx = max(0, cursor_idx - 1)
                continue
            if key == "down":
                if view:
                    cursor_idx = min(len(view) - 1, cursor_idx + 1)
                continue
            if key == "backspace":
                if filter_buf:
                    filter_buf.pop()
                    cursor_idx = 0
                    scroll_top = 0
                continue
            # Treat printable single chars as filter input. Skip control
            # codes and named tokens (handled above).
            if isinstance(key, str) and len(key) == 1 and key.isprintable():
                filter_buf.append(key)
                cursor_idx = 0
                scroll_top = 0

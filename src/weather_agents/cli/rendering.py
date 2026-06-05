"""Response rendering for the CLI — streaming display + final answer panel.

Pure presentation: every function here takes data and returns a Rich
renderable. No terminal I/O, no module-global mutable state — that keeps them
trivially testable and reusable across the streaming REPL, the ``sky task``
fast path, and the orchestration fallback. Extracted from the 5k-line
``cli/main.py`` so the "what an agent reply looks like" logic lives in one place.
"""

from __future__ import annotations

import os
import re

from rich import box
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from weather_agents.core.icons import AGENT_COLORS

# Per-agent animated spinner themes for streaming / status indicators.
AGENT_SPINNERS: dict[str, str] = {
    "fog": "dots",
    "rain": "line",
    "frost": "star",
    "snow": "dots2",
    "dew": "bounce",
    "fair": "arc",
}

# Short Chinese tagline shown under the agent name in the final response
# panel — gives each personality a recognizable identity without bloating
# the layout.
AGENT_TAGLINES: dict[str, str] = {
    "fog": "调研 · 信息整合",
    "rain": "实现 · 内容生成",
    "frost": "审阅 · 性能与安全",
    "snow": "规划 · 任务编排",
    "dew": "执行 · 命令与部署",
    "fair": "陪伴 · 共情对话",
}

# Theme used by Rich Markdown when rendering ```fenced code blocks```.
# monokai gives high-contrast syntax highlighting that pairs well with
# typical terminal backgrounds (dark or light). Override with
# WA_CODE_THEME to taste (e.g. dracula, github-dark, solarized-dark).
CODE_THEME = os.environ.get("WA_CODE_THEME", "monokai")


def strip_hr(markup: str) -> str:
    """Remove decorative horizontal rule lines from LLM markdown output.

    Handles ASCII and Unicode separator characters (dashes, underscores,
    asterisks, em-dash, horizontal-bar, box-drawing) — including patterns
    with spaces between characters like ``- - -``.
    """
    return re.sub(
        r"^[ \t]*([\-_*—–―─━])(?:\s*\1){2,}[ \t]*\n?",
        "",
        markup,
        flags=re.MULTILINE,
    )


def build_stream_display(
    agent,
    status_text: str,
    md_content: str,
) -> Table:
    """Live renderable during streaming: vertical colored bar | spinner |
    agent name | status, then markdown content beneath the bar.

    The left bar gives the whole turn a visible spine so the eye can scan
    where each agent's reply begins and ends, especially during nested
    delegations where multiple agents are streaming sequentially.
    """
    color = AGENT_COLORS.get(agent.name, "white")
    spinner_name = AGENT_SPINNERS.get(agent.name, "dots")

    tbl = Table(show_header=False, box=None, padding=0, expand=True)
    tbl.add_column(width=2, justify="center")  # colored vertical bar
    tbl.add_column(width=2, justify="center")  # spinner
    tbl.add_column(ratio=1)

    bar = Text("│", style=f"bold {color}")

    # Header line: agent name + emoji + status, with a tagline in dim
    # when no status is shown so the header is never empty/weak.
    name_text = Text()
    name_text.append(f" {agent.emoji} ", style=f"bold {color}")
    name_text.append(agent.display_name, style=f"bold {color}")
    if status_text:
        name_text.append("  ·  ", style="dim")
        name_text.append(status_text, style="dim italic")
    else:
        tagline = AGENT_TAGLINES.get(agent.name, "")
        if tagline:
            name_text.append("  ·  ", style="dim")
            name_text.append(tagline, style="dim italic")

    tbl.add_row(bar, Spinner(spinner_name, style=f"bold {color}"), name_text)
    if md_content:
        # Blank spacer row to separate header from body, then the body
        # rendered with the same color bar prefix so the visual spine
        # continues all the way down the response.
        tbl.add_row(bar, "", "")
        tbl.add_row(
            bar,
            "",
            Padding(
                Markdown(strip_hr(md_content), code_theme=CODE_THEME),
                pad=(0, 0, 0, 1),
            ),
        )

    return tbl


def build_response_panel(
    agent,
    content: str,
    elapsed: float,
    interrupted: bool = False,
    ctx: object | None = None,
) -> Panel:
    """Final response panel — rounded border, emoji-prefixed title with a
    dim tagline, body rendered as Markdown with syntax-highlighted code
    fences, footer with timing + context usage bar.

    Each agent's accent color is used for the border and title so multiple
    replies in a long session are visually distinguishable without having
    to read the name. Tagline gives a recognizable identity beat under
    the personality name.
    """
    color = AGENT_COLORS.get(agent.name, "white")
    timing = f"{elapsed:.1f}s" if not interrupted else f"{elapsed:.1f}s  ·  interrupted"

    # Title: emoji  display_name  ·  tagline (dim italic). The em-dash
    # spacer is rendered in dim so the eye lands on the name first.
    title_text = Text()
    title_text.append("  ")
    title_text.append(f"{agent.emoji}  ", style=f"bold {color}")
    title_text.append(agent.display_name, style=f"bold {color}")
    tagline = AGENT_TAGLINES.get(agent.name, "")
    if tagline:
        title_text.append("   ·   ", style="dim")
        title_text.append(tagline, style="dim italic")

    # Footer: timing + context usage bar (only when ctx is available).
    sub = f"[dim italic]{timing}[/]"
    if ctx is not None:
        try:
            cu = agent.context_usage()
            pct = cu["pct"]
            msgs = cu["message_count"]
            r = min(10, max(0, int(pct / 10)))
            bar_color = "green" if pct < 50 else "yellow" if pct < 80 else "red"
            sub += (
                f"   [bold {bar_color}]{'━' * r}[/][dim]{'╌' * (10 - r)}[/]"
                f"  [dim]{pct}%  ·  {msgs}msgs[/]"
            )
        except Exception:
            pass

    return Panel(
        Padding(Markdown(strip_hr(content), code_theme=CODE_THEME), pad=(0, 1, 0, 1)),
        title=title_text,
        title_align="left",
        subtitle=sub,
        subtitle_align="right",
        border_style=color,
        box=box.ROUNDED,
        padding=(1, 1),
    )

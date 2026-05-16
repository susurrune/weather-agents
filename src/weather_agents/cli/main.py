"""CLI interface for Weather Agents — terminal agent product."""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import sys
import time
import uuid
from typing import Any

import typer
from rich import box
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    import msvcrt as _msvcrt
else:
    import termios as _termios
    import tty as _tty

from weather_agents import __version__
from weather_agents.core.config import (
    USER_CONFIG_DIR,
    _sync_api_keys_to_env,
    delete_config,
    format_models_for_display,
    load_config,
    load_model_catalog,
    set_config,
)
from weather_agents.core.factory import (
    AGENT_CLASSES,
    AGENT_COLORS,
    create_system_context,
)
from weather_agents.core.icons import icon_text
from weather_agents.core.logger import set_request_id
from weather_agents.core.workspace import (
    detect_best_workspace_root,
    format_bytes,
    init_workspace,
    resolve_workspace_path,
)

# ── Slash commands registry (for popup) ──────────────────────────────────

_COMMANDS: list[tuple[str, str]] = [
    ("/help", "show all commands"),
    ("/clear", "clear screen"),
    ("/status", "agent overview"),
    ("/cost", "usage & cost (reset: /cost reset)"),
    ("/compact", "compress context"),
    ("/history", "event log"),
    ("/mcp", "MCP server status"),
    ("/skills", "list skills"),
    ("/use ", "activate a skill"),
    ("/deactivate", "deactivate skills"),
    ("/sessions", "list sessions"),
    ("/session new ", "start new session"),
    ("/session load ", "switch session"),
    ("/session delete ", "delete session"),
    ("/memory", "memory stats (clear: /memory clear)"),
    ("/workspace", "workspace info/set/auto"),
    ("/model", "view/change model  (all: /model all <m>)"),
    ("/apikey", "manage API keys  (set/del)"),
    ("/task ", "multi-agent orchestration"),
    ("/fog", "switch to Fog"),
    ("/rain", "switch to Rain"),
    ("/frost", "switch to Frost"),
    ("/snow", "switch to Snow"),
    ("/dew", "switch to Dew"),
    ("/sunshine", "switch to Sunshine (晴)"),
    ("/version", "version info"),
    ("/quit", "exit chat"),
    ("/exit", "exit chat"),
]

_COMMAND_LOOKUP: dict[str, str] = {c[0].split()[0].lstrip("/"): c[0] for c in _COMMANDS}

# Input-line history buffer (shared across all agents in the session)
_input_history: list[str] = []
_history_idx: int = 0

# ── Cross-platform key reader ─────────────────────────────────────────────


def _get_key() -> str:
    """Read a single keypress. Returns named tokens for special keys."""
    if sys.platform == "win32":
        ch = _msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            ch2 = _msvcrt.getwch()
            # Scan code Z (0x5A) = Shift+Tab on Windows console
            if ch2 == "Z":
                return "shift_tab"
            return {"H": "up", "P": "down", "K": "left", "M": "right"}.get(ch2, ch2)
        if ch == "\x03":
            raise KeyboardInterrupt
        if ch == "\x1b":
            if _msvcrt.kbhit():
                nxt = _msvcrt.getwch()
                if nxt == "[" and _msvcrt.kbhit():
                    nxt2 = _msvcrt.getwch()
                    if nxt2 == "Z":
                        return "shift_tab"
                return "esc"
            return "esc"
        if ch == "\r":
            return "enter"
        if ch == "\x08":
            return "backspace"
        if ch == "\t":
            # Some Windows terminals pass Shift+Tab as \t (same as Tab).
            # Use GetAsyncKeyState to check if Shift is held.
            try:
                import ctypes as _ct

                SHIFT_MASK = 0x8000
                if _ct.windll.user32.GetAsyncKeyState(0x10) & SHIFT_MASK:  # VK_SHIFT
                    return "shift_tab"
            except Exception:
                pass
            return "tab"
        return ch
    else:
        fd = sys.stdin.fileno()
        old = _termios.tcgetattr(fd)
        try:
            import select

            _tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                # Drain the full escape sequence with a short timeout
                seq = "\x1b"
                while True:
                    r, _, _ = select.select([fd], [], [], 0.05)
                    if not r:
                        break
                    more = sys.stdin.read(1)
                    if not more:
                        break
                    seq += more
                    # CSI sequences end with a byte in 0x40–0x7E
                    if ord(more) in range(0x40, 0x7F):
                        break
                if seq == "\x1b":
                    return "esc"
                if seq.startswith("\x1b["):
                    final = seq[-1]
                    if final == "A":
                        return "up"
                    if final == "B":
                        return "down"
                    if final == "C":
                        return "right"
                    if final == "D":
                        return "left"
                    if final == "Z":
                        return "shift_tab"
                    return "esc"
                if seq.startswith("\x1bO") and seq[-1] in "PQ":
                    return "tab"
                return "esc"
            if ch == "\x03":
                raise KeyboardInterrupt
            if ch in ("\r", "\n"):
                # Consume trailing \n from \r\n sent by some IMEs
                r, _, _ = select.select([fd], [], [], 0.01)
                if r:
                    nxt = sys.stdin.read(1)
                    if nxt == "\n":
                        pass  # consumed
                return "enter"
            if ch in ("\x7f", "\x08"):
                return "backspace"
            if ch == "\t":
                return "tab"
            return ch
        finally:
            _termios.tcsetattr(fd, _termios.TCSADRAIN, old)


def _poll_esc() -> bool:
    """Non-blocking check for Esc / Ctrl+C keypress.

    Called from the streaming loop as a secondary interrupt path so Esc is
    detected even when the background _esc_watcher executor thread can't read
    from the console (e.g. Windows Terminal limits on background threads).
    """
    if sys.platform == "win32":
        try:
            while _msvcrt.kbhit():
                ch = _msvcrt.getwch()
                if ch in ("\x1b", "\x03"):
                    return True
        except Exception:
            pass
    return False


app = typer.Typer(name="wacode", help="Weather Agents CLI", no_args_is_help=False)
console = Console()

# Per-agent animated spinner themes for streaming / status indicators
AGENT_SPINNERS: dict[str, str] = {
    "fog": "dots",
    "rain": "line",
    "frost": "star",
    "snow": "dots2",
    "dew": "bounce",
    "sunshine": "moon",
}


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"Weather Agents v{__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def _global_options(
    ctx: typer.Context,
    version: bool = typer.Option(  # noqa: B008
        False,
        "--version",
        "-V",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Top-level Typer callback hosting global flags like --version."""
    _ = version  # Consumed by callback above.
    if ctx.invoked_subcommand is None:
        chat(agent="fog", message=None)


def _strip_hr(markup: str) -> str:
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


def _build_stream_display(
    agent,
    status_text: str,
    md_content: str,
) -> Table:
    """Live renderable during streaming: agent header + content."""
    color = AGENT_COLORS.get(agent.name, "white")
    spinner_name = AGENT_SPINNERS.get(agent.name, "dots")

    tbl = Table(show_header=False, box=None, padding=0, expand=True)
    tbl.add_column(ratio=1)

    # Header: animated spinner · agent name · status
    name_text = Text()
    name_text.append(f" {agent.display_name}", style=f"bold {color}")
    if status_text:
        name_text.append("  ", style="dim")
        name_text.append(status_text, style="dim")

    header_row = Table(show_header=False, box=None, padding=0, expand=True)
    header_row.add_column(width=3, justify="center")
    header_row.add_column(ratio=1)
    header_row.add_row(
        Spinner(spinner_name, style=f"bold {color}"),
        name_text,
    )
    tbl.add_row(header_row)

    # Streamed content
    if md_content:
        tbl.add_row(Padding(Markdown(_strip_hr(md_content)), pad=(0, 2, 0, 2)))

    return tbl


def _build_response_panel(
    agent,
    content: str,
    elapsed: float,
    interrupted: bool = False,
) -> Panel:
    """Panel wrapping the final agent response."""
    color = AGENT_COLORS.get(agent.name, "white")
    title_text = Text()
    title_text.append("  ", style="dim")
    title_text.append(agent.display_name, style=f"bold {color}")

    sub = f"{elapsed:.1f}s" if not interrupted else f"{elapsed:.1f}s  interrupted"

    return Panel(
        Padding(Markdown(_strip_hr(content)), pad=(0, 1, 0, 1)),
        title=title_text,
        title_align="left",
        subtitle=f"[dim]{sub}[/dim]",
        subtitle_align="right",
        border_style=f"dim {color}",
        box=box.SIMPLE,
        padding=(0, 1),
    )


def _format_cost(cost: float) -> str:
    """Format cost adaptively: dollars or cents."""
    if cost >= 1.0:
        return f"${cost:.2f}"
    if cost >= 0.01:
        return f"${cost:.4f}"
    if cost > 0.0:
        cents = cost * 100
        return f"{cents:.2f}¢"
    return "$0"


def _build_status_line(agent, ctx) -> Text:
    """Build a compact status line: context bar · msgs · cost · model."""
    line = Text()
    try:
        cu = agent.context_usage()
        model = cu["model"]
        pct = cu["pct"]
        est = cu["estimated_tokens"]
        max_ctx = cu["max_tokens"]
        msgs = cu["message_count"]
        cost = ctx.llm.get_total_cost()

        filled = min(10, max(0, int(pct / 10)))
        bar_color = "green" if pct < 50 else "yellow" if pct < 80 else "red"
        bar = f"{'━' * filled}{'╌' * (10 - filled)}"
        ctx_str = f"{est // 1000}k/{max_ctx // 1000}k" if est > 1000 else f"{est}/{max_ctx}"

        line.append(bar, style=f"bold {bar_color}")
        line.append(f"  {pct}%  {ctx_str}", style="dim")
        line.append("  ·  ", style="dim")
        line.append(f"{msgs} msgs", style="dim")
        line.append("  ·  ", style="dim")
        line.append(_format_cost(cost), style="dim green" if cost < 0.01 else "dim yellow")
        line.append("  ·  ", style="dim")
        model_short = model if len(model) <= 30 else model[:27] + "…"
        line.append(model_short, style="dim")
    except Exception:
        line.append("", style="dim")
    return line


# -- Choice menu helpers (Claude Code-style interactive selection) ------------


def _numbered_blocks(text: str) -> list[str]:
    """Extract numbered or letter-prefixed lines from text.  2+ items or empty."""
    import re

    items: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        # Strip leading visual decorations (box-drawing, checkboxes, bullets)
        cleaned = re.sub(r"^[─-╿☐☑☒◦●○▪▸➔‣•★☆✦※*\-\s]+", "", stripped).strip()
        if re.match(r"^(?:\d+|[A-Za-z])[.、\)\s]\s*\S", cleaned):
            items.append(cleaned)
    return items if len(items) >= 2 else []


def _parse_simple_choices(text: str) -> list[str]:
    """Parse numbered or letter-prefixed OPTIONS (not questions) from AI response.

    Returns short choice strings (without the leading ``"N. "`` prefix)
    when the text contains 2+ concise items with NO question marks.
    Empty list otherwise.

    Example match::
        "1. 个人作品集\\n2. 产品官网\\n3. 博客首页"
        → ["个人作品集", "产品官网", "博客首页"]

    Also matches letter prefixes::
        "A. 功能测试\\nB. 性能测试\\nC. 安全测试"
        → ["功能测试", "性能测试", "安全测试"]
    """
    import re

    items: list[str] = []
    for line in text.split("\n"):
        # Strip leading visual decorations (box-drawing, checkboxes, bullets)
        cleaned = re.sub(r"^[─-╿☐☑☒◦●○▪▸➔‣•★☆✦※*\-–—\s]+", "", line).strip()
        m = re.match(r"^(?:\d+|[A-Za-z])[.、\)\s]\s*(.+)$", cleaned)
        if not m:
            continue
        content = m.group(1).strip()
        if "?" in content or "？" in content:
            continue  # questions, not choices
        if len(content) > 70:
            continue  # instructions, not choices
        items.append(content)
    return items if len(items) >= 2 else []


def _parse_questionnaire(text: str) -> list[dict] | None:
    r"""Parse a multi-question block into a structured questionnaire.

    Detects::

        1. 什么主题？ — 个人作品集？产品官网？博客首页？
        2. 为谁做？ — 你本人的品牌？某个项目？

    Returns ``[{"question": str, "options": [str, ...]}, ...]`` with at
    least one entry, or ``None`` when the pattern is not detected.
    """
    import re

    raw = _numbered_blocks(text)
    if not raw:
        return None

    questions: list[dict] = []
    for item in raw:
        stripped = re.sub(r"^(?:\d+|[A-Za-z])[.、\)\s]+", "", item).strip()

        # Split on dash separator: "Q? — A? B? C?"
        parts = re.split(r"\s*[—–-]\s*", stripped, maxsplit=1)
        if len(parts) < 2:
            continue

        q_text = parts[0].strip()
        if not any(c in q_text for c in "?？"):
            continue  # not a question

        # Parse sub-options separated by ？ ? /
        opts = re.split(r"[？?/]\s*", parts[1])
        opts = [o.strip().rstrip("？?)）") for o in opts if o.strip() and len(o.strip()) > 1]

        if q_text and len(opts) >= 2:
            questions.append({"question": q_text, "options": opts})

    return questions if questions else None


def _render_choice_menu(items: list[str], title: str = "") -> Table:
    """Build the Rich renderable for a choice-selection popup."""
    tbl = Table(show_header=False, box=None, padding=(0, 1), expand=False)
    tbl.add_column()
    for i, item in enumerate(items):
        if i == _render_choice_menu.selected:  # type: ignore[attr-defined]
            tbl.add_row(Text(f"❯ {item}", style="bold cyan"))
        else:
            tbl.add_row(Text(f"  {item}", style="default"))

    hint_tbl = Table(show_header=False, box=None, padding=(0, 1))
    hint_tbl.add_column()
    hint_tbl.add_row(Text("↑↓ navigate  ·  enter select  ·  esc cancel", style="dim"))

    inner = Table(show_header=False, box=None, padding=0)
    inner.add_column()
    inner.add_row(tbl)
    inner.add_row(hint_tbl)

    panel = Panel(
        inner,
        border_style="cyan",
        box=box.ROUNDED,
        padding=(0, 1),
        width=min(80, console.width - 4),
        title=Text(title, style="dim") if title else None,
        title_align="left",
    )

    outer = Table(show_header=False, box=None, padding=0, expand=True)
    outer.add_column(justify="center")
    outer.add_row(panel)
    return outer


def _show_choice_menu(
    items: list[str],
    title: str = "",
) -> str | None:
    """Interactive selection popup.  Returns selected item or ``None``."""
    _render_choice_menu.selected = 0  # type: ignore[attr-defined]

    with Live(
        Table(show_header=False, box=None, padding=0),
        console=console,
        refresh_per_second=10,
        transient=True,
    ) as live:
        while True:
            live.update(_render_choice_menu(items, title))

            key = _get_key()
            if key == "enter":
                return items[_render_choice_menu.selected]  # type: ignore[attr-defined]
            if key == "up":
                _render_choice_menu.selected = max(  # type: ignore[attr-defined]
                    0,
                    _render_choice_menu.selected - 1,  # type: ignore[attr-defined]
                )
            elif key == "down":
                _render_choice_menu.selected = min(  # type: ignore[attr-defined]
                    len(items) - 1,
                    _render_choice_menu.selected + 1,  # type: ignore[attr-defined]
                )
            elif key == "esc":
                return None


async def _run_questionnaire(questions: list[dict]) -> str | None:
    """Sequential multi-question selector.

    Shows each question with its options, one at a time.  Returns a
    combined answer string (``"Q1: A；Q2: B"``) or ``None`` on cancel.
    """
    answers: list[str] = []
    for qi, q in enumerate(questions, 1):
        choice = _show_choice_menu(q["options"], title=f"Q{qi}  {q['question']}")
        if choice is None:
            return None  # Esc → abort entire questionnaire
        answers.append(f"{q['question']} {choice}")

        # Brief confirmation of the selection
        console.print(f"  [dim]{q['question']}[/dim] [bold]{choice}[/bold]")
    return "；".join(answers) if answers else None


def _ime_cursor_col(display_name: str, buffer: list[str], cursor_pos: int) -> int:
    """Calculate the terminal column of the visual cursor in the input line."""
    from rich.cells import cell_len

    prefix = f"  {display_name} ❯ "
    return cell_len(prefix) + cell_len("".join(buffer[:cursor_pos]))


def _place_ime_cursor(col: int) -> None:
    """Move Windows console cursor to *col* (preserving current row).

    This tells the IME where to display the candidate window instead of
    defaulting to the far-right of the terminal after a Rich ``Live``
    update.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        import ctypes.wintypes
        import struct

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE

        csbi = ctypes.create_string_buffer(22)
        kernel32.GetConsoleScreenBufferInfo(handle, csbi)
        _, cur_y = struct.unpack_from("HH", csbi, 4)  # X, Y from dwCursorPosition

        coord = ctypes.wintypes._COORD(col, cur_y)
        kernel32.SetConsoleCursorPosition(handle, coord)
    except Exception:
        pass


# -- Interactive mode (plan / auto) ------------------------------------------

INTERACTIVE_MODE: str = "auto"  # "auto" or "plan"


# Combined auto-continue signal: CJK + English patterns that indicate
# the model plans more work.  Order matters for fast short-circuit.
_AUTO_CONTINUE = re.compile(
    r"(?:接下来|下一步|下面我|然后我|接着|还需要|继续|挨个|"
    r"next[,:]\s*(?:I'll|let me|step|we|I will)|"
    r"let me\s(?:now|continue|proceed|handle|take|work)|"
    r"I'll\s(?:now|start|begin|go|handle|take|work|need)|"
    r"接下来|剩下|现在[就我]|先[把给让]|"
    r"\b(?:continue|next)\b|"
    r"remaining|ongoing|further|additionally|subsequently|"
    r"now[,:]\s*(?:let|I'll|I will|we'll|the))"
)

# Auto-stop signal: explicit "done" markers trump auto-continue
_AUTO_STOP = re.compile(
    r"(?:完成了|全部完成|以上就|都做好了|已经完成|到此结束|"
    r"all done|task complete|finished|everything is done)",
    re.IGNORECASE,
)


def _should_auto_continue(text: str, had_tool_calls: bool = False) -> bool:
    """Check if the AI response signals more work — auto-continue.

    1. Explicit "done" markers → stop (strongest signal).
    2. Tool calls → continue (the model took action, likely more to come).
    3. Text pattern match → continue if planning/signalling more steps.
    """
    # Only inspect the tail — the last paragraph where forward-looking
    # language actually lives.
    last_lines = text.strip().split("\n")
    tail = "\n".join(last_lines[-6:]) if len(last_lines) > 6 else "\n".join(last_lines)

    if _AUTO_STOP.search(tail):
        return False
    if had_tool_calls:
        return True
    return bool(_AUTO_CONTINUE.search(tail))


# -- Chat -------------------------------------------------------------------


async def _chat_single(agent_name: str, message: str) -> None:
    set_request_id(uuid.uuid4().hex[:12])
    ctx = create_system_context()
    agent = ctx.agent_map.get(agent_name)
    if not agent:
        console.print(f"[red]Unknown agent: {agent_name}[/red]")
        return
    await _init_agent_lazy(agent, ctx)
    try:
        t0 = time.monotonic()
        spinner_style = AGENT_SPINNERS.get(agent.name, "dots")
        ict = icon_text(agent.name)
        status_handle = console.status(
            f"[dim]{ict} {agent.display_name} thinking...[/dim]",
            spinner=spinner_style,
        )
        status_handle.start()

        def _on_status(msg: str) -> None:
            status_handle.update(f"[dim]{ict} {msg}[/dim]")

        try:
            resp = await agent.chat(message, on_status=_on_status)
        finally:
            status_handle.stop()
        elapsed = time.monotonic() - t0
        console.print(_build_response_panel(agent, resp, elapsed))
    finally:
        await ctx.close_all()


async def _init_agent_lazy(agent, ctx) -> None:
    """Init an agent if not already initialized. Used for lazy startup."""
    if not agent._base_system_prompt:
        await agent.init()
        # Init MCP if configured (only on first agent init)
        if ctx.mcp is not None and ctx.mcp._server_configs and not ctx.mcp_status:
            with contextlib.suppress(Exception):
                ctx.mcp_status = await ctx.mcp.connect_all()


def _build_input_display(
    agent,
    ctx,
    buffer: str,
    cursor_pos: int,
    popup_visible: bool,
    selected_idx: int,
    filtered_commands: list[tuple[str, str]],
    mode: str = "auto",
) -> list:
    """Build renderables for the input area with optional command popup."""
    color = AGENT_COLORS.get(agent.name, "cyan")
    results: list = []
    w = console.width
    rule = "━" * max(0, w - 4)

    # ── Top solid line (agent color) ─────────────────────────────────────────
    results.append(Text(f"  {rule}", style=f"bold {color}"))

    # ── Prompt line ──────────────────────────────────────────────────────────
    prompt = Text()
    prompt.append("  ")
    if mode == "plan":
        prompt.append("[PLAN] ", style="bold magenta")
    elif mode == "auto":
        prompt.append("[AUTO] ", style="bold yellow")
    prompt.append(agent.display_name, style=f"bold {color}")
    prompt.append(" ❯ ", style=f"{color}")
    if buffer:
        pre = buffer[:cursor_pos]
        post = buffer[cursor_pos:]
        if buffer.startswith("/"):
            space_idx = pre.find(" ")
            if space_idx > 0:
                prompt.append(pre[:space_idx], style="bold cyan")
                prompt.append(pre[space_idx:])
                prompt.append("▌", style=f"bold {color}")
                prompt.append(post)
            elif space_idx < 0 and post:
                # /command followed by args — pre is the cmd, post is args
                prompt.append(pre, style="bold cyan")
                prompt.append("▌", style=f"bold {color}")
                prompt.append(post)
            else:
                prompt.append(pre, style="bold cyan")
                prompt.append("▌", style=f"bold {color}")
                if post:
                    prompt.append(post)
        else:
            prompt.append(pre)
            prompt.append("▌", style=f"bold {color}")
            if post:
                prompt.append(post)
    else:
        prompt.append("▌", style=f"bold {color}")
    results.append(prompt)

    # ── Hint line ────────────────────────────────────────────────────────────
    if popup_visible:
        hint = "↑↓ select  tab complete  esc dismiss  enter confirm"
    else:
        hint = "/ commands  ↑↓ history  esc clear"
    hint_line = Text()
    hint_line.append("  ")
    hint_line.append(hint, style="dim")
    results.append(hint_line)

    # ── Bottom solid line (subtle) ───────────────────────────────────────────
    results.append(Text(f"  {rule}", style="dim"))

    # ── Status bar ───────────────────────────────────────────────────────────
    status_text = _build_status_line(agent, ctx)
    status_bar = Text()
    status_bar.append("  ")
    status_bar.append(status_text)
    results.append(status_bar)

    # ── Command popup ────────────────────────────────────────────────────────
    if popup_visible and filtered_commands:
        tbl = Table(show_header=False, box=None, padding=(0, 1), expand=False)
        tbl.add_column(width=28)
        tbl.add_column(style="dim")

        start = max(0, min(selected_idx - 6, len(filtered_commands) - 14))
        end = min(len(filtered_commands), start + 14)

        if start > 0:
            tbl.add_row(Text("  ↑ more", style="dim"), "")

        for i in range(start, end):
            cmd, desc = filtered_commands[i]
            if i == selected_idx:
                cmd_text = Text()
                cmd_text.append("❯ ", style="bold cyan")
                cmd_text.append(cmd, style="bold cyan")
                tbl.add_row(cmd_text, Text(desc, style="default"))
            else:
                cmd_text = Text()
                cmd_text.append("  ")
                cmd_text.append(cmd, style="cyan")
                tbl.add_row(cmd_text, Text(desc, style="dim"))

        if end < len(filtered_commands):
            tbl.add_row(Text("  ↓ more", style="dim"), "")

        popup = Panel(
            tbl,
            title="[dim]commands[/dim]",
            title_align="left",
            border_style="dim cyan",
            box=box.ROUNDED,
            padding=(0, 0),
            width=min(60, w - 4),
        )
        results.append(popup)

    return results


def _read_line_with_popup(agent, ctx, mode: str = "auto") -> str:
    """Read a line of input with slash-command popup support.

    *mode* controls the indicator shown in the input bar ("auto" or "plan").
    """
    # Fall back to simple input when stdin is not a TTY (piped / test env)
    if not sys.stdin.isatty():
        color = AGENT_COLORS.get(agent.name, "cyan")
        prompt = Text()
        prompt.append(agent.display_name, style=f"bold {color}")
        prompt.append(" ❯ ", style=color)
        return console.input(prompt)

    global _history_idx
    buffer: list[str] = []
    cursor_pos = 0
    popup_visible = False
    selected_idx = 0

    result = ""
    with Live(
        Table(show_header=False, box=None, padding=0),
        console=console,
        refresh_per_second=10,
        transient=True,
    ) as live:
        while True:
            text = "".join(buffer)
            filtered = [c for c in _COMMANDS if c[0].startswith(text)] if popup_visible else []
            if filtered and selected_idx >= len(filtered):
                selected_idx = len(filtered) - 1

            tbl = Table(show_header=False, box=None, padding=0, expand=True)
            tbl.add_column(ratio=1)
            for item in _build_input_display(
                agent, ctx, text, cursor_pos, popup_visible, selected_idx, filtered, mode
            ):
                tbl.add_row(item)
            live.update(tbl)
            # Move the console cursor to match the visual ▌ position so
            # the IME candidate window appears in the right place.
            _place_ime_cursor(_ime_cursor_col(agent.display_name, buffer, cursor_pos))

            try:
                key = _get_key()
            except KeyboardInterrupt:
                raise

            if key == "enter":
                if popup_visible and filtered:
                    result = filtered[selected_idx][0]
                else:
                    result = "".join(buffer).strip()
                if result:
                    break
                continue

            if key == "esc":
                if popup_visible:
                    popup_visible = False
                    buffer.clear()
                    cursor_pos = 0
                    selected_idx = 0
                else:
                    buffer.clear()
                    cursor_pos = 0
                continue

            if key == "backspace":
                if buffer and cursor_pos > 0:
                    del buffer[cursor_pos - 1]
                    cursor_pos -= 1
                    if not buffer:
                        popup_visible = False
                        selected_idx = 0
                continue

            if key == "left":
                if popup_visible:
                    popup_visible = False
                if cursor_pos > 0:
                    cursor_pos -= 1
                continue

            if key == "right":
                if popup_visible:
                    popup_visible = False
                if cursor_pos < len(buffer):
                    cursor_pos += 1
                continue

            if key == "up":
                if popup_visible and filtered:
                    selected_idx = max(0, selected_idx - 1)
                elif _input_history:
                    if _history_idx > 0:
                        _history_idx -= 1
                    buffer[:] = list(_input_history[_history_idx])
                    cursor_pos = len(buffer)
                continue

            if key == "down":
                if popup_visible and filtered:
                    selected_idx = min(len(filtered) - 1, selected_idx + 1)
                elif _input_history:
                    if _history_idx < len(_input_history) - 1:
                        _history_idx += 1
                        buffer[:] = list(_input_history[_history_idx])
                    else:
                        _history_idx = len(_input_history)
                        buffer.clear()
                    cursor_pos = len(buffer)
                continue

            if key == "shift_tab":
                # Toggle between auto and plan mode
                global INTERACTIVE_MODE
                INTERACTIVE_MODE = "plan" if INTERACTIVE_MODE == "auto" else "auto"
                mode = INTERACTIVE_MODE  # sync local var so the display updates
                mode_tag = (
                    "[bold yellow]AUTO[/bold yellow]"
                    if INTERACTIVE_MODE == "auto"
                    else "[bold magenta]PLAN[/bold magenta]"
                )
                console.print(f"  {mode_tag}")
                continue

            if popup_visible and filtered:
                if key == "up":
                    selected_idx = max(0, selected_idx - 1)
                elif key == "down":
                    selected_idx = min(len(filtered) - 1, selected_idx + 1)
                elif key == "tab":
                    buffer[:] = list(filtered[selected_idx][0])
                    if not filtered[selected_idx][0].endswith(" "):
                        buffer.append(" ")
                    cursor_pos = len(buffer)
                    popup_visible = False
                elif isinstance(key, str) and len(key) == 1 and key.isprintable():
                    buffer.insert(cursor_pos, key)
                    cursor_pos += 1
                    selected_idx = 0
                continue

            if isinstance(key, str) and len(key) == 1:
                if key == "/" and not buffer:
                    buffer.insert(cursor_pos, key)
                    cursor_pos += 1
                    popup_visible = True
                    selected_idx = 0
                elif key.isprintable():
                    buffer.insert(cursor_pos, key)
                    cursor_pos += 1
                    if buffer == ["/"]:
                        popup_visible = True
                        selected_idx = 0
                    elif popup_visible:
                        selected_idx = 0

    if result:
        # Save to history (no duplicates, limit 50)
        if not _input_history or _input_history[-1] != result:
            _input_history.append(result)
            if len(_input_history) > 50:
                _input_history.pop(0)
        _history_idx = len(_input_history)
        color = AGENT_COLORS.get(agent.name, "cyan")
        echo = Text()
        echo.append(agent.display_name, style=f"bold {color}")
        echo.append(" ❯ ", style=color)
        echo.append(result, style="white")
        console.print(echo)
    return result


async def _interactive(agent_name: str | None = None) -> None:
    global INTERACTIVE_MODE
    set_request_id(uuid.uuid4().hex[:12])
    ctx = create_system_context()
    # Lazy init: only initialize current agent, not all 5
    current = agent_name or "fog"
    agent = ctx.agent_map[current]
    await _init_agent_lazy(agent, ctx)
    model = ctx.config.llm.default_model
    ws = getattr(ctx, "workspace_path", "")
    workspace_path = ws if isinstance(ws, str) else ""
    _print_welcome(model, workspace_path)

    try:
        agents = ctx.agent_map

        while True:
            try:
                inp: str | None = _read_line_with_popup(agent, ctx, INTERACTIVE_MODE)
            except (EOFError, KeyboardInterrupt):
                console.print()
                break
            if not inp:
                continue

            # Update session preview from the first user message
            with contextlib.suppress(Exception):
                await agent.memory.update_session_preview()
            cmd = inp.strip()
            cmd_lower = cmd.lower()

            # --- Slash commands ---
            if cmd_lower in ("/quit", "/exit", "/q"):
                break
            if cmd_lower in ("/help", "/?"):
                _print_help(ctx)
                continue
            if cmd_lower == "/clear":
                console.clear()
                _print_welcome(ctx.config.llm.default_model, workspace_path)
                continue
            if cmd_lower == "/status":
                _print_status(agents)
                continue
            if cmd_lower == "/cost":
                _print_cost(ctx)
                continue
            if cmd_lower == "/cost reset":
                ctx.llm.reset_usage_stats()
                console.print("  [dim]usage stats reset[/dim]")
                continue
            if cmd_lower == "/memory":
                await _print_memory_status(ctx)
                continue
            if cmd_lower == "/memory clear":
                for ag in ctx.agent_map.values():
                    removed = sum(1 for m in ag.memory.short_term if m.role != "system")
                    await ag.memory.clear_short_term()
                    console.print(
                        f"  [green]cleared {icon_text(ag.name)} {ag.display_name} "
                        f"({removed} messages)[/green]"
                    )
                continue
            if cmd_lower == "/compact":
                await _init_agent_lazy(agent, ctx)
                result = await agent.compact()
                console.print(f"  [green]✓ {result}[/green]")
                continue
            if cmd_lower == "/history":
                _print_history(ctx)
                continue
            if cmd_lower == "/mcp":
                _print_mcp_status(ctx)
                continue
            if cmd_lower == "/skills":
                _print_skills(agent)
                continue
            if cmd_lower.startswith("/use "):
                skill_name = cmd[5:].strip()
                if agent.activate_skill(skill_name):
                    console.print(f"  [green]+ {skill_name}[/green]")
                else:
                    console.print(
                        f"  [red]unknown skill: {skill_name}[/red] [dim](/skills to list)[/dim]"
                    )
                continue
            if cmd_lower == "/deactivate":
                agent.deactivate_all_skills()
                console.print("  [dim]skills deactivated[/dim]")
                continue
            if cmd_lower == "/workspace":
                _print_workspace(ctx)
                continue
            if cmd_lower.startswith("/workspace set "):
                _handle_workspace_set(cmd, ctx)
                continue
            if cmd_lower == "/workspace auto":
                _handle_workspace_auto(ctx)
                continue
            if cmd_lower == "/sessions":
                await _print_sessions(agent)
                continue
            if cmd_lower.startswith("/session"):
                await _handle_session_command(cmd, agent)
                continue
            if cmd_lower.startswith("/task "):
                goal = cmd[6:].strip()
                if goal:
                    # Lazy-init all agents for orchestration
                    for ag in agents.values():
                        await _init_agent_lazy(ag, ctx)
                    await _run_task(goal, agents)
                continue
            if cmd_lower.startswith("/model"):
                _handle_model_command(cmd, ctx)
                continue
            if cmd_lower.startswith("/apikey"):
                _handle_apikey_command(cmd, ctx)
                continue
            if cmd_lower == "/version":
                console.print(f"  Weather Agents [bold]v{__version__}[/bold]")
                continue
            if cmd_lower == "/auto":
                INTERACTIVE_MODE = "auto"
                console.print(
                    "  [bold yellow]AUTO[/bold yellow]  mode — autonomous reasoning & execution"
                )
                continue
            if cmd_lower == "/plan":
                INTERACTIVE_MODE = "plan"
                console.print(
                    "  [bold magenta]PLAN[/bold magenta]  mode — plan first, then confirm before execute"
                )
                continue
            if cmd_lower.lstrip("/") in AGENT_CLASSES:
                new_name = cmd_lower.lstrip("/")
                new_agent = agents[new_name]
                await _init_agent_lazy(new_agent, ctx)
                current = new_name
                agent = new_agent
                color = AGENT_COLORS.get(new_name, "white")
                switch_msg = Text()
                switch_msg.append("  ")
                switch_msg.append(agent.display_name, style=f"bold {color}")
                switch_msg.append("  ready", style="dim")
                console.print(switch_msg)
                continue
            if cmd_lower.startswith("/") and cmd.strip() != "/":
                _print_help(ctx)
                continue

            # --- Plan mode: show a plan before executing ---
            if INTERACTIVE_MODE == "plan":
                await _init_agent_lazy(agent, ctx)
                plan_t0 = time.monotonic()
                plan_content = ""
                plan_live = Live(
                    _build_stream_display(agent, "Planning...", ""),
                    console=console,
                    refresh_per_second=12,
                    transient=False,
                )
                plan_live.start()
                try:
                    async for event in agent.chat_stream(f"[PLAN] {inp}"):
                        if event["type"] == "content":
                            plan_content += event["text"]
                            plan_live.update(
                                _build_stream_display(agent, "Planning...", plan_content)
                            )
                        elif event["type"] == "done":
                            break
                except KeyboardInterrupt:
                    pass
                finally:
                    if plan_content.strip():
                        plan_live.update(
                            _build_response_panel(agent, plan_content, time.monotonic() - plan_t0)
                        )
                    plan_live.stop()

                if not plan_content.strip():
                    console.print("  [dim yellow]plan empty — skipping[/dim yellow]")
                    continue

                console.print()
                console.print(
                    "  [dim]Press [bold]Enter[/bold] to execute · [bold]Esc[/bold] to cancel[/dim]"
                )
                key = _get_key()
                if key != "enter":
                    console.print("  [dim]cancelled[/dim]")
                    agent._pop_last_user_message()
                    continue

                # Plan confirmed: remove the [PLAN] user message so it
                # doesn't appear twice when chat_stream adds inp again.
                agent._pop_last_user_message()

            # --- Streaming chat with tool-call support ---
            # Inner loop: allows choice-menu re-entry with a new input
            _auto_continue_count = 0
            while True:
                await _init_agent_lazy(agent, ctx)
                t0 = time.monotonic()
                interrupted = False
                md_content = ""
                status_text = "Thinking..."
                activities: list[dict] = []
                _empty_retried = False
                _esc_event = asyncio.Event()

                async def _esc_watcher(ev: asyncio.Event):
                    loop = asyncio.get_running_loop()
                    while not ev.is_set():
                        if sys.platform == "win32":
                            hit = await loop.run_in_executor(None, _msvcrt.kbhit)
                            if hit:
                                ch = await loop.run_in_executor(None, _msvcrt.getwch)
                                if ch == "\x1b":
                                    ev.set()
                                    break
                            await asyncio.sleep(0.05)
                        else:
                            key = await loop.run_in_executor(None, _get_key)
                            if key == "esc":
                                ev.set()
                                break

                async def _resize_watcher(lv: Live):
                    """Refresh Live display when terminal is resized (drag/resize)."""
                    loop = asyncio.get_running_loop()
                    last = None
                    while True:
                        cur = await loop.run_in_executor(
                            None, lambda: (console.width, console.height)
                        )
                        if last is not None and cur != last:
                            with contextlib.suppress(Exception):
                                lv.refresh()
                        last = cur
                        await asyncio.sleep(0.3)

                live = Live(
                    _build_stream_display(agent, "", ""),
                    console=console,
                    refresh_per_second=12,
                    transient=False,
                )
                live.start()

                esc_task = asyncio.create_task(_esc_watcher(_esc_event))
                resize_task = asyncio.create_task(_resize_watcher(live))

                try:
                    async for event in agent.chat_stream(inp):
                        if _esc_event.is_set() or _poll_esc():
                            if not _esc_event.is_set():
                                _esc_event.set()
                            interrupted = True
                            break
                        if event["type"] == "content":
                            md_content += event["text"]
                            live.update(_build_stream_display(agent, status_text, md_content))
                        elif event["type"] == "reasoning":
                            status_text = "Thinking..."
                            live.update(_build_stream_display(agent, status_text, md_content))
                        elif event["type"] == "tool_status":
                            status_text = event["label"]
                            is_dlg = event["label"].startswith("Delegating to ")
                            activities.append(
                                {
                                    "label": event["label"],
                                    "status": "running",
                                    "delegation": is_dlg,
                                }
                            )
                            live.update(_build_stream_display(agent, status_text, md_content))
                        elif event["type"] == "tool_done":
                            for a in activities:
                                if a["label"] == event["label"] and a["status"] == "running":
                                    a["status"] = "done" if event.get("success") else "error"
                                    break
                            live.update(_build_stream_display(agent, status_text, md_content))
                        elif event["type"] == "done":
                            break
                except KeyboardInterrupt:
                    interrupted = True
                finally:
                    _esc_event.set()
                    esc_task.cancel()
                    resize_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await esc_task
                    with contextlib.suppress(asyncio.CancelledError):
                        await resize_task
                    if md_content.strip():
                        live.update(
                            _build_response_panel(
                                agent, md_content, time.monotonic() - t0, interrupted
                            )
                        )
                    live.stop()

                if not md_content.strip():
                    if interrupted:
                        console.print("  [dim]interrupted[/dim]")
                    elif not _empty_retried:
                        _empty_retried = True
                        console.print("  [dim yellow]empty response, retrying...[/dim yellow]")
                        await asyncio.sleep(0.5)
                        # Clean up the user msg + empty assistant from the failed attempt
                        agent._pop_last_user_message()
                        if (
                            agent.memory.short_term
                            and agent.memory.short_term[-1].role == "assistant"
                            and not agent.memory.short_term[-1].content
                        ):
                            agent.memory.short_term.pop()
                        continue
                    else:
                        console.print("  [dim yellow]model returned empty response[/dim yellow]")
                    break  # Exit inner loop, back to input

                # — Auto mode: continue if the AI signals more work —
                had_tools = any(a["status"] == "done" for a in activities)
                if (
                    INTERACTIVE_MODE == "auto"
                    and not interrupted
                    and _should_auto_continue(md_content, had_tool_calls=had_tools)
                    and _auto_continue_count < 10
                ):
                    _auto_continue_count += 1
                    inp = "请继续完成"
                    console.print(
                        f"  [dim]⋯ auto-continue ({_auto_continue_count}/10)[/dim]"
                        + (" [tools]" if had_tools else "")
                    )
                    continue  # Restart streaming

                # — Choice menu: detect numbered options and show interactive popup —
                # Try questionnaire first, fall back to single-select
                questions = _parse_questionnaire(md_content)
                if questions:
                    inp = await _run_questionnaire(questions)
                    if inp is not None:
                        continue  # Restart streaming with combined answers
                else:
                    choices = _parse_simple_choices(md_content)
                    if choices:
                        choice = _show_choice_menu(choices)
                        if choice is not None:
                            inp = choice
                            continue  # Restart streaming with selected choice

                break  # No choices or user cancelled → back to main input loop

    finally:
        console.print()
        console.print(Rule(style="dim"))
        console.print("  [dim]Session ended[/dim]")
        await ctx.close_all()


# -- Welcome & Help --------------------------------------------------------


def _build_welcome_art() -> Text:
    """Build the ASCII-art welcome banner."""
    t = Text()
    t.append("        ·  ✦  ·       · ✦  ·  ✦       ✦  ·  ✦  ·\n", style="dim bright_white")
    t.append("     ✦        ✦    ✦         ✦    ·         ✦    \n", style="dim bright_white")
    t.append("   ", style="")
    t.append("≈", style="cyan bold")
    t.append("  W E A T H E R   A G E N T S  ", style="bold white")
    t.append("≈", style="cyan bold")
    t.append("\n")
    t.append("     ·        ·    ·         ·    ✦         ·    \n", style="dim bright_white")
    t.append("        ✦  ·  ✦       ✦ ·  ✦  ·       ·  ✦  ·    \n", style="dim bright_white")
    return t


def _print_welcome(model: str, workspace_path: str = "") -> None:
    console.print()

    agent_names = list(AGENT_CLASSES.keys())
    agent_display = {c.name: c.display_name for c in AGENT_CLASSES.values()}
    agent_role = {
        "fog": "research",
        "rain": "codegen",
        "frost": "review",
        "snow": "planning",
        "dew": "devops",
        "sunshine": "companion",
    }
    art = _build_welcome_art()

    # ── Agent row ──────────────────────────────────────────────────────
    agent_tbl = Table(show_header=False, box=None, padding=(0, 3), expand=True)
    for _ in agent_names:
        agent_tbl.add_column(ratio=1, justify="center")

    agent_rows: list[list[Text]] = [[], [], []]
    for idx, name in enumerate(agent_names):
        color = AGENT_COLORS.get(name, "white")
        active = idx == 0
        display = agent_display.get(name, name.title())
        role = agent_role.get(name, "")
        s = "●" if active else "○"
        s_style = f"bold {color}" if active else "dim"

        line1 = Text(justify="center")
        line1.append(display, style=f"bold {color}")

        line2 = Text(justify="center")
        line2.append(role, style="dim italic")

        line3 = Text(justify="center")
        line3.append(f"{s} ", style=s_style)
        line3.append("active" if active else "standby", style=s_style)

        agent_rows[0].append(line1)
        agent_rows[1].append(line2)
        agent_rows[2].append(line3)

    for row in agent_rows:
        agent_tbl.add_row(*row)

    # ── Meta ───────────────────────────────────────────────────────────
    meta = Text(justify="center")
    meta.append("model  ", style="dim")
    meta.append(model, style="cyan bold")
    meta.append("   ·   ", style="dim")
    meta.append("workspace  ", style="dim")
    if workspace_path:
        short_ws = workspace_path if len(workspace_path) <= 40 else "…" + workspace_path[-38:]
        meta.append(short_ws, style="white")
    else:
        meta.append("(none)", style="dim")

    tip = Text(justify="center")
    tip.append("Type  ", style="dim")
    tip.append("/", style="cyan bold")
    tip.append("  for commands  ·  ", style="dim")
    tip.append("/help", style="cyan bold")
    tip.append("  for reference", style="dim")

    # ── Assemble ───────────────────────────────────────────────────────
    content = Table(show_header=False, box=None, padding=0, expand=True)
    content.add_column(justify="center")

    content.add_row(art)
    content.add_row(Text(""))
    content.add_row(agent_tbl)
    content.add_row(Text(""))
    content.add_row(meta)
    content.add_row(Text(""))
    content.add_row(tip)

    console.print(
        Panel(
            content,
            border_style="dim white",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )
    console.print()


def _print_help(ctx) -> None:
    en = getattr(ctx.config.llm, "language", "zh") == "en"

    def _h(zh: str, en_text: str) -> str:
        return en_text if en else zh

    sections = [
        (
            _h("指令", "Commands"),
            [
                ("/help", _h("显示帮助", "show this help")),
                ("/clear", _h("清屏", "clear screen")),
            ],
        ),
        (
            _h("Agent 切换", "Agents"),
            [
                (
                    "/fog  /rain  /frost  /snow  /dew  /sunshine",
                    _h("切换当前 Agent", "switch active agent"),
                ),
                ("/task <goal>", _h("多 Agent 编排", "multi-agent orchestration")),
            ],
        ),
        (
            _h("设置", "Config"),
            [
                ("/model", _h("查看模型 (↑↓选择)", "view model (↑↓ select)")),
                ("/model <name>", _h("设置全局模型", "set default model")),
                ("/model <agent> <name>", _h("设置 Agent 模型", "override per-agent model")),
                ("/model all <name>", _h("批量设置全部 Agent", "set all agents model")),
                ("/apikey", _h("查看密钥", "list API keys")),
                ("/apikey set <prov> <key>", _h("添加密钥", "add / replace key")),
                ("/apikey del <prov>", _h("删除密钥", "remove key")),
                ("/workspace", _h("工作空间信息", "workspace info")),
                ("/workspace set <path>", _h("设置工作空间", "set custom workspace")),
                ("/workspace auto", _h("自动检测工作空间", "reset to auto-detect")),
            ],
        ),
        (
            _h("技能", "Skills"),
            [
                ("/skills", _h("列出技能", "list available skills")),
                ("/use <skill>", _h("激活技能", "activate a skill")),
                ("/deactivate", _h("停用所有技能", "deactivate all skills")),
            ],
        ),
        (
            _h("信息", "Info"),
            [
                ("/status", _h("Agent 概览", "agent overview")),
                ("/cost", _h("用量与费用", "usage & cost")),
                ("/cost reset", _h("重置计数", "reset counters")),
                ("/compact", _h("压缩上下文", "compress context")),
                ("/history", _h("事件日志", "event log")),
                ("/mcp", _h("MCP 服务器状态", "MCP status")),
                ("/memory", _h("记忆层状态", "memory stats")),
                ("/memory clear", _h("清除短期记忆", "clear short-term memory")),
                ("/version", _h("版本信息", "version info")),
            ],
        ),
        (
            _h("会话", "Session"),
            [
                ("/sessions", _h("列出会话", "list saved sessions")),
                ("/session new [name]", _h("新建会话", "start new session")),
                ("/session load <id>", _h("加载会话", "switch to session")),
                ("/session delete <id>", _h("删除会话", "delete session")),
                ("/quit", _h("退出", "exit")),
                ("/exit", _h("退出", "exit")),
            ],
        ),
    ]

    console.print()
    for title, items in sections:
        console.print(Rule(f"  {title}  ", align="left", style="dim"))
        tbl = Table(show_header=False, box=None, padding=(0, 2, 0, 2))
        tbl.add_column(width=34, no_wrap=True)
        tbl.add_column(style="dim")
        for cmd, desc in items:
            tbl.add_row(Text(cmd, style="cyan"), desc)
        console.print(tbl)
    console.print()


# -- Display helpers -------------------------------------------------------


def _print_status(agents: dict) -> None:
    console.print()
    console.print(Rule("  Agents  ", align="left", style="dim"))

    tbl = Table(
        show_header=True,
        box=box.SIMPLE_HEAD,
        padding=(0, 2, 0, 0),
        header_style="dim",
        show_edge=False,
    )
    tbl.add_column("Agent", width=18)
    tbl.add_column("State", width=8)
    tbl.add_column("Skills", style="dim", min_width=10)
    tbl.add_column("Calls", justify="right", width=6)
    tbl.add_column("In / Out tokens", justify="right", width=20)

    for a in agents.values():
        s = a.get_status()
        name = s["name"]
        color = AGENT_COLORS.get(name, "white")
        active_skills = [sk["name"] for sk in s.get("skills", []) if sk.get("active")]
        skills_str = ", ".join(active_skills) if active_skills else "—"
        state_color = "green" if s["state"] == "idle" else "yellow"
        tokens = f"{s['usage']['prompt_tokens']:,}  /  {s['usage']['completion_tokens']:,}"
        agent_cell = Text()
        agent_cell.append(s["display_name"], style=f"bold {color}")

        tbl.add_row(
            agent_cell,
            Text(s["state"], style=state_color),
            skills_str,
            str(s["usage"]["calls"]),
            Text(tokens, style="dim"),
        )
    console.print(tbl)


def _print_cost(ctx) -> None:
    console.print()
    console.print(Rule("  Usage & Cost  ", align="left", style="dim"))
    stats = ctx.llm.get_usage_stats()
    if not stats:
        console.print("  [dim]no usage recorded yet[/dim]")
        return

    tbl = Table(
        show_header=True,
        box=box.SIMPLE_HEAD,
        padding=(0, 2, 0, 0),
        header_style="dim",
        show_edge=False,
    )
    tbl.add_column("Agent", width=12)
    tbl.add_column("Calls", justify="right", width=6)
    tbl.add_column("In tokens", justify="right", width=12)
    tbl.add_column("Out tokens", justify="right", width=12)
    tbl.add_column("Cost", justify="right", width=10)

    total_cost = 0.0
    for name, s in stats.items():
        cost = s.get("cost", 0.0)
        total_cost += cost
        cost_style = "green" if cost < 0.01 else "yellow" if cost < 0.10 else "red"
        tbl.add_row(
            Text(name, style="cyan"),
            str(s.get("calls", 0)),
            f"{s.get('prompt_tokens', 0):,}",
            f"{s.get('completion_tokens', 0):,}",
            Text(_format_cost(cost), style=cost_style),
        )

    # Total row
    total_style = "green" if total_cost < 0.05 else "yellow" if total_cost < 0.50 else "red"
    tbl.add_row(
        Text("total", style="bold"),
        "",
        "",
        "",
        Text(_format_cost(total_cost), style=f"bold {total_style}"),
    )
    console.print(tbl)


def _print_history(ctx) -> None:
    events = ctx.bus.get_history(limit=20)
    if not events:
        console.print("  [dim]no events yet[/dim]")
        return

    console.print()
    console.print(Rule("  Event Log  ", align="left", style="dim"))

    tbl = Table(
        show_header=True,
        box=box.SIMPLE_HEAD,
        padding=(0, 2, 0, 0),
        header_style="dim",
        show_edge=False,
    )
    tbl.add_column("Time", width=10, style="dim")
    tbl.add_column("Type", width=18)
    tbl.add_column("Source", width=8)
    tbl.add_column("Detail", style="dim")

    for e in events[-15:]:
        ts = e.timestamp.strftime("%H:%M:%S")
        summary = _summarize_event(e.type.value, e.data)
        tbl.add_row(
            ts,
            Text(e.type.value, style="cyan"),
            Text(e.source, style="bold"),
            summary,
        )
    console.print(tbl)


def _summarize_event(event_type: str, data: dict | None) -> str:
    """Render an event's data field as a short, readable line — no truncated dicts."""
    if not data:
        return ""
    if event_type == "tool_call":
        tool = data.get("tool", "?")
        args = data.get("args", {})
        arg_bits = [f"{k}={_short(v)}" for k, v in list(args.items())[:2]]
        suffix = f"({', '.join(arg_bits)})" if arg_bits else ""
        return f"{tool}{suffix}"
    if event_type == "llm_call":
        usage = data.get("usage") or {}
        ptok = usage.get("prompt_tokens", 0)
        ctok = usage.get("completion_tokens", 0)
        return f"{data.get('model', '?')}  {ptok}→{ctok} tok"
    if event_type == "state_change":
        return f"{data.get('old_state', '?')} → {data.get('new_state', '?')}"
    # Fallback: comma-separated key=value pairs trimmed to width
    pairs = [f"{k}={_short(v)}" for k, v in list(data.items())[:3]]
    return ", ".join(pairs)


def _short(v) -> str:
    s = str(v)
    return s if len(s) <= 30 else s[:27] + "..."


def _print_mcp_status(ctx) -> None:
    mcp_servers = ctx.config.mcp.servers
    if not mcp_servers:
        console.print("  [dim]no MCP servers configured[/dim]")
        return

    connected: dict[str, str] = {}
    for line in ctx.mcp_status or []:
        if ":" in line:
            name, info = line.split(":", 1)
            connected[name.strip()] = info.strip()

    console.print()
    console.print(Rule("  MCP Servers  ", align="left", style="dim"))

    tbl = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    tbl.add_column(width=3)  # icon
    tbl.add_column(width=20)  # name
    tbl.add_column(width=6, style="dim")  # transport
    tbl.add_column()  # status

    for s in mcp_servers:
        name = s.get("name", "?")
        transport = "stdio" if s.get("command") else "sse"
        enabled = s.get("enabled", True)
        if not enabled:
            icon = Text("○", style="dim")
            status = Text("disabled", style="dim")
        elif name in connected:
            icon = Text("●", style="green")
            status = Text(connected[name], style="green")
        else:
            icon = Text("●", style="yellow")
            status = Text("not connected", style="yellow")
        tbl.add_row(icon, Text(name, style="cyan"), transport, status)

    console.print(tbl)


async def _print_memory_status(ctx) -> None:
    console.print()
    console.print(Rule("  Memory  ", align="left", style="dim"))

    tbl = Table(
        show_header=True,
        box=box.SIMPLE_HEAD,
        padding=(0, 2, 0, 0),
        header_style="dim",
        show_edge=False,
    )
    tbl.add_column("Agent", width=18)
    tbl.add_column("Short", justify="right", width=8)
    tbl.add_column("Working", justify="right", width=8)
    tbl.add_column("Long-term", justify="right", width=10)

    for ag in ctx.agent_map.values():
        color = AGENT_COLORS.get(ag.name, "white")
        short = len(ag.memory.short_term)
        working = len(ag.memory.working)
        long_term = await ag.memory.recall(limit=100)

        agent_cell = Text()
        agent_cell.append(ag.display_name, style=f"bold {color}")

        tbl.add_row(
            agent_cell,
            Text(str(short), style="dim" if short == 0 else "default"),
            Text(str(working), style="dim" if working == 0 else "default"),
            Text(str(len(long_term)), style="dim" if not long_term else "default"),
        )
    console.print(tbl)


async def _print_sessions(agent) -> None:
    sessions = await agent.memory.list_sessions()
    active_id = agent.memory.get_active_session()

    console.print()
    console.print(Rule("  Sessions  ", align="left", style="dim"))

    if not sessions:
        console.print("  [dim]no saved sessions[/dim]")
        console.print("  [dim]/session new [name]  — start a new one[/dim]")
        return

    tbl = Table(
        show_header=True,
        box=box.SIMPLE_HEAD,
        padding=(0, 2, 0, 0),
        header_style="dim",
        show_edge=False,
    )
    tbl.add_column("", width=2)  # active marker
    tbl.add_column("ID", width=20, style="cyan")
    tbl.add_column("Name / Preview", min_width=24)
    tbl.add_column("Msgs", justify="right", width=6, style="dim")

    for s in sessions:
        active = s["id"] == active_id
        marker = Text("●", style="green") if active else Text(" ")
        sid = s["id"]
        name = s["name"] or s["preview"] or "(empty)"
        if len(name) > 48:
            name = name[:45] + "…"
        count = s["message_count"]
        tbl.add_row(marker, sid, Text(name, style="bold" if active else "default"), str(count))

    console.print(tbl)
    console.print()
    console.print("  [dim]/session new [name]    start fresh session[/dim]")
    console.print("  [dim]/session load <id>     switch to session[/dim]")
    console.print("  [dim]/session delete <id>   delete session[/dim]")


async def _handle_session_command(cmd: str, agent) -> None:
    parts = cmd.strip().split(maxsplit=2)
    if len(parts) < 2:
        await _print_sessions(agent)
        return

    action = parts[1].lower()

    if action == "new":
        name = parts[2] if len(parts) > 2 else None
        sid = await agent.memory.create_session(name)
        console.print(f"  [green]+ new session [cyan]{sid}[/cyan][/green]")
        return

    if action == "load":
        if len(parts) < 3:
            console.print("  [red]usage: /session load <id>[/red]")
            return
        sid = parts[2]
        ok = await agent.memory.load_session(sid)
        if ok:
            console.print(f"  [green]loaded session [cyan]{sid}[/cyan][/green]")
        else:
            console.print(f"  [red]session not found: {sid}[/red]")
        return

    if action in ("delete", "del", "rm"):
        if len(parts) < 3:
            console.print("  [red]usage: /session delete <id>[/red]")
            return
        sid = parts[2]
        ok = await agent.memory.delete_session(sid)
        if ok:
            console.print(f"  [green]deleted session [cyan]{sid}[/cyan][/green]")
        else:
            console.print(f"  [red]session not found: {sid}[/red]")
        return

    console.print("  [red]usage: /session [new|load|delete] ...[/red]")


def _print_skills(agent) -> None:
    skills = agent.get_available_skills()
    color = AGENT_COLORS.get(agent.name, "white")

    console.print()
    console.print(
        Rule(
            f"  {icon_text(agent.name)} {agent.display_name} Skills  ",
            align="left",
            style="dim",
        )
    )

    if not skills:
        console.print("  [dim]no skills available[/dim]")
        return

    tbl = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    tbl.add_column(width=3)  # status dot
    tbl.add_column(width=22)  # name
    tbl.add_column(style="dim")  # description

    for sk in skills:
        dot = Text("●", style=f"bold {color}") if sk["active"] else Text("○", style="dim")
        tbl.add_row(
            dot, Text(sk["name"], style="cyan" if sk["active"] else "dim"), sk["description"]
        )

    console.print(tbl)
    console.print()
    console.print("  [dim]/use <skill>   activate  ·  /deactivate   deactivate all[/dim]")


# -- Workspace management ----------------------------------------------------


def _print_workspace_path() -> None:
    """Print workspace path, disk info, and subdirectories."""
    import shutil

    cfg = load_config()
    configured = cfg.workspace.path
    is_auto = configured.lower() == "auto"
    resolved = resolve_workspace_path(configured)
    resolved_str = str(resolved.resolve())
    exists = resolved.exists()

    console.print()
    console.print(Rule("  Workspace  ", align="left", style="dim"))

    # Key-value info table
    kv = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    kv.add_column(width=12, style="dim")
    kv.add_column()

    kv.add_row("mode", Text("auto", style="cyan") if is_auto else Text("custom", style="yellow"))
    if is_auto:
        kv.add_row("detected", str(detect_best_workspace_root()))
    else:
        kv.add_row("configured", configured)
    kv.add_row("resolved", resolved_str)
    kv.add_row(
        "status", Text("exists", style="green") if exists else Text("not created", style="yellow")
    )

    try:
        usage = shutil.disk_usage(resolved_str)
        ratio = usage.free / usage.total if usage.total else 0
        bar_filled = max(1, int(ratio * 10))
        disk_bar_color = "green" if ratio > 0.2 else "yellow" if ratio > 0.1 else "red"
        disk_bar = Text("█" * bar_filled + "─" * (10 - bar_filled), style=disk_bar_color)
        disk_info = Text()
        disk_info.append(format_bytes(usage.free), style="green")
        disk_info.append(f" free / {format_bytes(usage.total)}  ")
        disk_info.append(disk_bar)
        kv.add_row("disk", disk_info)
    except OSError:
        kv.add_row("disk", Text("unavailable", style="red"))

    console.print(kv)

    # Subdirectory listing
    if exists:
        subs = sorted(resolved.iterdir())
        if subs:
            console.print()
            console.print("  [dim]contents[/dim]")
            for child in subs:
                if child.is_dir():
                    console.print(f"    [dim]{child.name}/[/dim]")
                else:
                    console.print(f"    [dim]{child.name}[/dim]")

    # Hint
    console.print()
    console.print("  [dim]/workspace set <path>   set custom path[/dim]")
    if not is_auto:
        console.print("  [dim]/workspace auto         reset to auto-detect[/dim]")

    # Windows drive list
    if os.name == "nt":
        console.print()
        console.print(Rule("  Drives  ", align="left", style="dim"))
        from weather_agents.core.workspace import _get_drive_list

        drive_tbl = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
        drive_tbl.add_column(width=2)  # active marker
        drive_tbl.add_column(width=6)  # path
        drive_tbl.add_column(width=12)  # free
        drive_tbl.add_column(width=12, style="dim")  # total
        drive_tbl.add_column()  # bar

        for d in _get_drive_list():
            active = str(resolved).startswith(d.path)
            marker = Text("●", style="green") if active else Text(" ")
            ratio = d.free_bytes / d.total_bytes if d.total_bytes else 0
            bar_c = "green" if ratio > 0.2 else "yellow" if ratio > 0.1 else "red"
            filled = max(1, int(ratio * 10))
            bar = Text("█" * filled + "─" * (10 - filled), style=bar_c)
            drive_tbl.add_row(
                marker,
                Text(d.path, style="cyan"),
                Text(f"{format_bytes(d.free_bytes)} free", style="green"),
                f"/ {format_bytes(d.total_bytes)}",
                bar,
            )
        console.print(drive_tbl)


def _free_bar(free: int, total: int) -> str:
    """Draw a minimal 10-char usage bar."""
    if total <= 0:
        return "[dim][----------][/dim]"
    ratio = free / total
    filled = max(1, int(ratio * 10))
    bar = "█" * filled + "─" * (10 - filled)
    color = "green" if ratio > 0.2 else "yellow" if ratio > 0.1 else "red"
    return f"[{color}]{bar}[/{color}]"


def _print_workspace(ctx) -> None:
    _print_workspace_path()


def _handle_workspace_set(cmd: str, ctx) -> None:
    path_str = cmd[len("/workspace set ") :].strip()
    if not path_str:
        console.print("  [red]usage: /workspace set <absolute-path>[/red]")
        return

    # Resolve and validate
    from pathlib import Path

    resolved = Path(os.path.expanduser(path_str)).resolve()
    if not resolved.is_absolute():
        console.print("  [red]path must be absolute[/red]")
        return

    ok, msg = set_config("workspace.path", str(resolved))
    color = "green" if ok else "red"
    console.print(f"  [{color}]{msg}[/{color}]")

    if ok:
        # Immediately create the new workspace
        try:
            init_workspace(resolved)
            console.print(f"  [green]workspace created at {resolved}[/green]")
        except OSError as e:
            console.print(f"  [yellow]Warning: could not create workspace: {e}[/yellow]")


def _handle_workspace_auto(ctx) -> None:
    ok, msg = delete_config("workspace.path")
    color = "green" if ok else "red"
    console.print(f"  [{color}]{msg}[/{color}]")

    # Detect and create the auto workspace for display
    root = detect_best_workspace_root()
    try:
        init_workspace(root)
        console.print(f"  [green]workspace -> {root}[/green]")
    except OSError as e:
        console.print(f"  [yellow]Warning: {e}[/yellow]")


# -- Model & API key management --------------------------------------------


def _interactive_model_select(prompt: str = "Select model") -> str | None:
    """Show available models and let the user pick with ↑↓ / enter / esc."""
    catalog = load_model_catalog()
    if not catalog:
        console.print("  [red]No models found in catalog[/red]")
        return None

    # Flatten to ordered list — group by provider, with a header row
    entries: list[dict] = []  # {name, provider, context, max_output, is_header?}
    for provider, models in catalog.items():
        entries.append({"name": provider, "is_header": True})
        for m in models:
            m["is_header"] = False
            entries.append(m)

    selected_idx = 0
    # Move to first non-header
    for i, e in enumerate(entries):
        if not e.get("is_header"):
            selected_idx = i
            break

    # Show current configuration above the selection list
    with Live(
        Table(show_header=False, box=None, padding=0),
        console=console,
        refresh_per_second=10,
        transient=True,
    ) as live:
        while True:
            tbl = Table(show_header=False, box=None, padding=0, expand=True)
            tbl.add_column(ratio=1)

            # Prompt line
            prompt_line = Text()
            prompt_line.append(f"\n  {prompt}", style="bold")
            prompt_line.append("  (↑↓ select  enter confirm  esc cancel)", style="dim")
            tbl.add_row(prompt_line)
            tbl.add_row(Text())

            # Model list
            for i, e in enumerate(entries):
                if e.get("is_header"):
                    tbl.add_row(Text(f"  [{e['name'].upper()}]", style="bold dim"))
                    continue

                line = Text()
                marker = "❯" if i == selected_idx else " "
                style = "bold cyan" if i == selected_idx else ""
                line.append(f" {marker} ", style=style)
                line.append(f"  {e['name']}", style=style)

                ctx_str = f"ctx={e.get('context_window', '?')}"
                if i == selected_idx:
                    line.append(f"  ({ctx_str}, max={e.get('max_output', '?')})", style="dim")
                else:
                    line.append(f"  ({ctx_str})", style="dim")

                tbl.add_row(line)

            tbl.add_row(Text())
            hint = Text()
            hint.append("  [dim]Tip: /model <agent> </dim>", style="dim")
            hint.append("<name>", style="cyan dim")
            hint.append(" [dim]for per-agent,  [/dim]", style="dim")
            hint.append("/model all <name>", style="cyan dim")
            hint.append(" [dim]for all[/dim]", style="dim")
            tbl.add_row(hint)
            live.update(tbl)

            try:
                key = _get_key()
            except KeyboardInterrupt:
                return None

            if key == "enter":
                return entries[selected_idx].get("name", "")
            if key == "esc":
                return None
            if key == "up":
                for j in range(selected_idx - 1, -1, -1):
                    if not entries[j].get("is_header"):
                        selected_idx = j
                        break
            if key == "down":
                for j in range(selected_idx + 1, len(entries)):
                    if not entries[j].get("is_header"):
                        selected_idx = j
                        break
            if key == "left":
                selected_idx = 0
                for j, e in enumerate(entries):
                    if not e.get("is_header"):
                        selected_idx = j
                        break


def _handle_model_command(cmd: str, ctx) -> None:
    parts = cmd.strip().split(maxsplit=1)

    # ── /model (no args) — show status + interactive select ──────────────
    if len(parts) == 1:
        current = ctx.config.llm.default_model
        console.print(f"\n  [bold]default:[/bold] [cyan]{current}[/cyan]\n")
        for name in AGENT_CLASSES:
            agent_cfg = getattr(ctx.config.agents, name, None)
            m = agent_cfg.model if agent_cfg and agent_cfg.model else current
            marker = "" if agent_cfg and agent_cfg.model else " [dim](default)[/dim]"
            console.print(f"  {icon_text(name)} {name:<6}  {m}{marker}")
        console.print(
            "\n  [dim]/model <name>           set default model\n"
            "  /model <agent> <name>    set agent model\n"
            "  /model all <name>        set for all agents\n"
            "  /model <agent> default   reset to default[/dim]"
        )

        model = _interactive_model_select("Select model")
        if model:
            ok, msg = set_config("default_model", model)
            if ok:
                ctx.config.llm.default_model = model
                console.print(f"\n  [green]model -> {model}[/green]")
            else:
                console.print(f"\n  [red]{msg}[/red]")
        return

    arg = parts[1].strip()
    tokens = arg.split(maxsplit=1)

    # ── /model all [model] — bulk set all agents ─────────────────────────
    if tokens[0] == "all":
        if len(tokens) == 2:
            model_name = tokens[1]
            for name in AGENT_CLASSES:
                set_config(f"model.{name}", model_name)
                agent_cfg = getattr(ctx.config.agents, name)
                agent_cfg.model = model_name
            console.print(f"  [green]all agents -> {model_name}[/green]")
        else:
            model = _interactive_model_select("Select model for all agents")
            if model:
                for name in AGENT_CLASSES:
                    set_config(f"model.{name}", model)
                    agent_cfg = getattr(ctx.config.agents, name)
                    agent_cfg.model = model
                console.print(f"\n  [green]all agents -> {model}[/green]")
        return

    # ── /model <agent> (no model name) — interactive select for that agent
    if len(tokens) == 1 and tokens[0] in AGENT_CLASSES:
        agent_name = tokens[0]
        model = _interactive_model_select(f"Select model for {icon_text(agent_name)} {agent_name}")
        if model:
            set_config(f"model.{agent_name}", model)
            agent_cfg = getattr(ctx.config.agents, agent_name)
            agent_cfg.model = model
            console.print(f"\n  [green]{icon_text(agent_name)} {agent_name} -> {model}[/green]")
        return

    # ── /model <agent> <model> — direct set ──────────────────────────────
    if len(tokens) == 2 and tokens[0] in AGENT_CLASSES:
        agent_name, model_name = tokens
        if model_name.lower() == "default":
            delete_config(f"model.{agent_name}")
            agent_cfg = getattr(ctx.config.agents, agent_name)
            agent_cfg.model = ""
            console.print(f"  [green]{icon_text(agent_name)} {agent_name} -> default[/green]")
        else:
            set_config(f"model.{agent_name}", model_name)
            agent_cfg = getattr(ctx.config.agents, agent_name)
            agent_cfg.model = model_name
            console.print(f"  [green]{icon_text(agent_name)} {agent_name} -> {model_name}[/green]")
        return

    # ── /model <model> — direct set default ─────────────────────────────
    model_name = arg
    ok, msg = set_config("default_model", model_name)
    if ok:
        ctx.config.llm.default_model = model_name
        console.print(f"  [green]model -> {model_name}[/green]")
    else:
        console.print(f"  [red]{msg}[/red]")


def _handle_apikey_command(cmd: str, ctx) -> None:
    parts = cmd.strip().split(maxsplit=2)

    if len(parts) == 1:
        keys = ctx.config.llm.api_keys
        if not keys:
            console.print("  [dim]no API keys configured[/dim]")
        else:
            console.print()
            for provider, key in keys.items():
                masked = key[:8] + "****" + key[-4:] if len(key) > 16 else key[:4] + "****"
                console.print(
                    f"  [green]●[/green]  [cyan]{provider:<12}[/cyan]  [dim]{masked}[/dim]"
                )
        console.print(
            "\n  [dim]/apikey set <provider> <key>    add or replace\n"
            "  /apikey del <provider>             remove[/dim]"
        )
        return

    action = parts[1].lower()

    if action in ("set", "add") and len(parts) == 3:
        tokens = parts[2].strip().split(maxsplit=1)
        if len(tokens) != 2:
            console.print("  [red]usage: /apikey set <provider> <key>[/red]")
            return
        provider, key = tokens
        provider = provider.lower()
        ok, msg = set_config(f"api_key.{provider}", key)
        if ok:
            ctx.config.llm.api_keys[provider] = key
            _sync_api_keys_to_env({provider: key})
            console.print(f"  [green]+ {provider} key saved[/green]")
        else:
            console.print(f"  [red]{msg}[/red]")
        return

    if action in ("del", "delete", "rm", "remove"):
        if len(parts) < 3:
            console.print("  [red]usage: /apikey del <provider>[/red]")
            return
        provider = parts[2].strip().lower()
        ok, msg = delete_config(f"api_key.{provider}")
        if ok:
            ctx.config.llm.api_keys.pop(provider, None)
            from weather_agents.core.config import _ENV_KEY_MAP

            env_var = _ENV_KEY_MAP.get(provider, f"{provider.upper()}_API_KEY")
            os.environ.pop(env_var, None)
            console.print(f"  [green]- {provider} key removed[/green]")
        else:
            console.print(f"  [red]{msg}[/red]")
        return

    console.print("  [red]usage: /apikey [set <provider> <key> | del <provider>][/red]")


# -- Task orchestration ----------------------------------------------------


async def _run_task(goal: str, agents=None) -> None:
    own_ctx = None
    if agents is None:
        own_ctx = create_system_context()
        await own_ctx.init_all()
        agents = own_ctx.agent_map

    try:
        from weather_agents.core.factory import orchestrate_task

        status_handles: dict[str, Any] = {}

        async def _on_start(t):
            ict = icon_text(t.assigned_to or "")
            sp = AGENT_SPINNERS.get(t.assigned_to or "", "dots")
            sh = console.status(f"[dim]{ict} {t.description}...[/dim]", spinner=sp)
            sh.start()
            status_handles[t.id] = sh

        async def _on_done(t, r):
            sh = status_handles.pop(t.id, None)
            if sh:
                sh.stop()
            ict = icon_text(r.agent)
            icon = "[green]✓[/green]" if r.success else "[red]✗[/red]"
            console.print(f"  {icon} {ict} {r.description}")

        console.print()
        console.print(Rule("  Task  ", align="left", style="dim"))
        console.print(f"  [bold]{goal}[/bold]")
        console.print()

        with console.status("  [dim]planning…[/dim]", spinner="dots"):
            tasks, results, summary = await orchestrate_task(
                goal,
                agents,
                on_task_start=_on_start,
                on_task_done=_on_done,
                result_truncate=500,
            )

        if not tasks:
            console.print("  [dim]no tasks generated[/dim]")
            return

        # Task plan
        console.print(Rule("  Plan  ", align="left", style="dim"))
        plan_tbl = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
        plan_tbl.add_column(width=4, style="dim")  # id
        plan_tbl.add_column(width=4)  # emoji
        plan_tbl.add_column()  # description
        plan_tbl.add_column(width=12, style="dim")  # dep

        for t in tasks:
            dep = f"← {t.parent_id}" if t.parent_id else ""
            plan_tbl.add_row(f"{t.id}.", t.assigned_to or "", t.description, dep)
        console.print(plan_tbl)

        # Results
        console.print()
        ok = sum(1 for r in results if r.success)
        total = len(results)
        result_color = "green" if ok == total else "yellow" if ok > 0 else "red"
        console.print(
            f"  [{result_color}]{'✓' if ok == total else '!'} {ok}/{total} tasks completed[/{result_color}]"
        )

        if summary:
            console.print()
            console.print(Rule("  Summary  ", align="left", style="dim"))
            console.print(Padding(Markdown(_strip_hr(summary)), pad=(0, 2, 0, 2)))

    finally:
        # Clean up any lingering status handles
        for sh in status_handles.values():
            sh.stop()
        if own_ctx:
            await own_ctx.close_all()


# -- CLI commands ----------------------------------------------------------


@app.command()
def chat(
    agent: str = typer.Argument("fog", help="Agent name (fog/rain/frost/snow/dew/sunshine)"),
    message: str | None = typer.Argument(None, help="Message (omit for interactive mode)"),
) -> None:
    """Chat with an agent. Omit message for interactive mode."""
    if agent not in AGENT_CLASSES:
        console.print(f"[red]Unknown agent: {agent}. Use: {', '.join(AGENT_CLASSES)}[/red]")
        raise typer.Exit(1)

    # First-run: nothing is configured yet. Walk the user through the wizard,
    # then drop straight into chat — no separate `wacode init` step required.
    if not _is_configured():
        console.print("\n  [yellow]No API key configured yet — running first-run setup.[/yellow]")
        _run_setup_wizard()
        if not _is_configured():
            console.print(
                "\n  [yellow]Skipped without entering a key. "
                "Run [cyan]wacode init[/cyan] later when ready.[/yellow]\n"
            )
            raise typer.Exit(0)

    if message:
        asyncio.run(_chat_single(agent, message))
    else:
        asyncio.run(_interactive(agent))


@app.command()
def task(goal: str = typer.Argument(..., help="Task goal for multi-agent orchestration")) -> None:
    """Multi-agent orchestration: Snow decomposes and coordinates agents."""
    asyncio.run(_run_task(goal))


@app.command()
def status() -> None:
    """Show all agent status and model configuration."""
    ctx = create_system_context()
    console.print()
    console.print(Rule("  Agent Configuration  ", align="left", style="dim"))

    tbl = Table(
        show_header=True,
        box=box.SIMPLE_HEAD,
        padding=(0, 2, 0, 0),
        header_style="dim",
        show_edge=False,
    )
    tbl.add_column("Agent", width=20)
    tbl.add_column("Specialty", style="dim", width=14)
    tbl.add_column("Model", width=30)
    tbl.add_column("Skills", style="dim")

    for name, cls in AGENT_CLASSES.items():
        color = AGENT_COLORS.get(name, "white")
        model = getattr(ctx.config.agents, name).model or ctx.config.llm.default_model
        skills = ", ".join(cls.skill_names) if cls.skill_names else "—"
        agent_cell = Text()
        agent_cell.append(cls.display_name, style=f"bold {color}")
        tbl.add_row(agent_cell, cls.specialty, Text(model, style="cyan"), skills)
    console.print(tbl)


# -- Config ----------------------------------------------------------------


@app.command()
def config(
    action: str = typer.Argument("list", help="list / set / delete / models"),
    key: str = typer.Argument(None, help="Config key"),
    value: str = typer.Argument(None, help="Config value (for set)"),
) -> None:
    """Manage configuration."""
    if action == "list":
        cfg = load_config()
        console.print()
        console.print(Rule("  Configuration  ", align="left", style="dim"))

        kv = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
        kv.add_column(width=14, style="dim")
        kv.add_column()
        kv.add_row("default model", Text(cfg.llm.default_model, style="cyan"))
        kv.add_row("temperature", str(cfg.llm.temperature))
        kv.add_row("max tokens", str(cfg.llm.max_tokens))
        kv.add_row("timeout", f"{cfg.llm.timeout}s")
        console.print(kv)

        console.print()
        console.print(Rule("  Per-agent Models  ", align="left", style="dim"))
        agent_tbl = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
        agent_tbl.add_column(width=20)
        agent_tbl.add_column()
        for name in AGENT_CLASSES:
            color = AGENT_COLORS.get(name, "white")
            attr = getattr(cfg.agents, name)
            m = attr.model or ""
            model_cell = Text(m, style="cyan") if m else Text("(default)", style="dim")
            agent_cell = Text()
            agent_cell.append(name, style=f"bold {color}")
            agent_tbl.add_row(agent_cell, model_cell)
        console.print(agent_tbl)

        if cfg.llm.api_keys:
            console.print()
            console.print(Rule("  API Keys  ", align="left", style="dim"))
            key_tbl = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
            key_tbl.add_column(width=3)
            key_tbl.add_column(width=14)
            key_tbl.add_column(style="dim")
            for p, v in cfg.llm.api_keys.items():
                masked = v[:8] + "…" + v[-4:] if len(v) > 16 else v[:4] + "…"
                key_tbl.add_row(Text("●", style="green"), Text(p, style="cyan"), masked)
            console.print(key_tbl)

        console.print()
        console.print(f"  [dim]{USER_CONFIG_DIR / 'config.yaml'}[/dim]")

    elif action == "set":
        if not key or value is None:
            console.print("  [red]usage: wacode config set <key> <value>[/red]")
            raise typer.Exit(1)
        ok, msg = set_config(key, value)
        color = "green" if ok else "red"
        console.print(f"  [{color}]{msg}[/{color}]")

    elif action == "delete":
        if not key:
            console.print("  [red]usage: wacode config delete <key>[/red]")
            raise typer.Exit(1)
        ok, msg = delete_config(key)
        color = "green" if ok else "red"
        console.print(f"  [{color}]{msg}[/{color}]")

    elif action == "models":
        catalog = load_model_catalog()
        if not catalog:
            console.print("  [yellow]no models.yaml found[/yellow]")
            return
        console.print(format_models_for_display(catalog))

    else:
        console.print(f"  [red]unknown action: {action} (list / set / delete / models)[/red]")


# -- Memory ----------------------------------------------------------------


@app.command()
def memory(
    action: str = typer.Argument("status", help="status / clear"),
    agent_name: str = typer.Argument(None, help="Agent name or omit for all"),
) -> None:
    """Manage agent memory."""

    async def _run() -> None:
        ctx = create_system_context()
        await ctx.init_all()
        try:
            if action == "clear":
                targets = [agent_name] if agent_name else list(ctx.agent_map.keys())
                for name in targets:
                    agent = ctx.agent_map.get(name)
                    if not agent:
                        console.print(f"  [red]unknown agent: {name}[/red]")
                        continue
                    # Count only non-system messages — those are what clear_short_term removes.
                    removed = sum(1 for m in agent.memory.short_term if m.role != "system")
                    await agent.memory.clear_short_term()
                    console.print(
                        f"  [green]cleared {icon_text(agent.name)} {agent.display_name} "
                        f"({removed} messages)[/green]"
                    )
            else:
                for _name, agent in ctx.agent_map.items():
                    short = len(agent.memory.short_term)
                    working = len(agent.memory.working)
                    long_term = await agent.memory.recall(limit=100)
                    console.print(
                        f"  {icon_text(agent.name)} {agent.display_name}  "
                        f"[dim]{short} short / {working} working / "
                        f"{len(long_term)} long-term[/dim]"
                    )
        finally:
            await ctx.close_all()

    asyncio.run(_run())


# -- Init / Setup Wizard --------------------------------------------------


def _is_configured() -> bool:
    """Has the user supplied at least one API key (config file or env)?"""
    cfg = load_config()
    if any(v for v in cfg.llm.api_keys.values()):
        return True
    for env_var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY"):
        if os.environ.get(env_var):
            return True
    return False


def _provider_for_model(model: str) -> str:
    """Infer the provider responsible for a model id."""
    m = model.lower()
    if m.startswith("ollama/"):
        return "local"
    if "deepseek" in m:
        return "deepseek"
    if m.startswith(("claude", "anthropic/")):
        return "anthropic"
    if m.startswith(("gpt", "openai/", "o1", "o3", "o4")):
        return "openai"
    return "openai"


def _flatten_catalog(catalog: dict) -> list[tuple[str, str]]:
    """Return [(provider, model_name), ...] preserving provider order."""
    out = []
    for prov, models in catalog.items():
        for m in models:
            out.append((prov, m["name"]))
    return out


def _print_catalog(flat: list[tuple[str, str]]) -> None:
    """Print numbered model menu grouped by provider."""
    last_prov = None
    for i, (prov, name) in enumerate(flat, 1):
        if prov != last_prov:
            console.print(f"\n    [bold dim]{prov.upper()}[/bold dim]")
            last_prov = prov
        console.print(f"      [dim]{i:>2}.[/dim] [cyan]{name}[/cyan]")


def _pick_from_catalog(
    flat: list[tuple[str, str]],
    prompt: str,
    default_idx: int | None = None,
) -> tuple[str, str] | None:
    """Loop until the user types a valid number or hits Enter for default."""
    hint = f" [dim](Enter for {default_idx})[/dim]" if default_idx else ""
    while True:
        raw = console.input(f"  {prompt}{hint}: ").strip()
        if not raw and default_idx is not None:
            return flat[default_idx - 1]
        if raw.isdigit() and 1 <= int(raw) <= len(flat):
            return flat[int(raw) - 1]
        console.print(f"    [red]pick a number 1-{len(flat)}[/red]")


def _collect_keys(providers: set[str]) -> None:
    """Prompt for one API key per cloud provider in the set."""
    cloud = sorted(p for p in providers if p != "local")
    if not cloud:
        console.print("  [dim]All chosen models run locally — no API keys needed.[/dim]")
        return
    console.print(f"\n  [bold]API keys for:[/bold] [cyan]{', '.join(cloud)}[/cyan]")
    console.print("  [dim](pasted keys are hidden in transit but stored in plain YAML)[/dim]\n")
    for provider in cloud:
        cfg = load_config()
        current = cfg.llm.api_keys.get(provider, "")
        suffix = " [dim](Enter to keep current)[/dim]" if current else ""
        key = console.input(f"  {provider:<10} key{suffix}: ").strip()
        if key:
            ok, msg = set_config(f"api_key.{provider}", key)
            color = "green" if ok else "red"
            console.print(f"    [{color}]{msg}[/{color}]")


def _run_setup_wizard() -> None:
    """Walk the user through choosing a model strategy and storing API keys.

    Does NOT enter chat — the caller decides whether to launch _interactive().
    """
    console.print()
    console.print(
        Panel(
            "[bold]Weather Agents Setup[/bold]\n[dim]Configure your agents in 3 steps[/dim]",
            border_style="dim cyan",
            box=box.ROUNDED,
            padding=(1, 2),
            width=44,
        )
    )

    catalog = load_model_catalog()
    if not catalog:
        console.print("\n  [red]No model catalog found. Reinstall and try again.[/red]")
        return
    flat = _flatten_catalog(catalog)

    # Step 1: choose mode
    console.print()
    console.print(Rule("  Step 1 — Agent mode  ", align="left", style="dim"))
    step1_tbl = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    step1_tbl.add_column(width=3, style="cyan bold")
    step1_tbl.add_column(width=12, style="bold")
    step1_tbl.add_column(style="dim")
    step1_tbl.add_row("1.", "Unified", "one model + one API key for all agents  (recommended)")
    step1_tbl.add_row("2.", "Per-agent", "a different model for each agent  (advanced)")
    console.print(step1_tbl)

    mode = ""
    while mode not in ("1", "2"):
        mode = console.input("\n  Choice [1/2] — Enter for 1: ").strip() or "1"
        if mode not in ("1", "2"):
            console.print("  [red]please enter 1 or 2[/red]")

    # Step 2: pick models
    providers_needed: set[str] = set()
    console.print()
    console.print(Rule("  Step 2 — Model selection  ", align="left", style="dim"))

    if mode == "1":
        _print_catalog(flat)
        default_idx = next((i + 1 for i, (p, _) in enumerate(flat) if p == "deepseek"), 1)
        picked = _pick_from_catalog(flat, "\n  Model #", default_idx=default_idx)
        if not picked:
            return
        provider, model_name = picked
        set_config("default_model", model_name)
        for ag in AGENT_CLASSES:
            delete_config(f"model.{ag}")
        console.print(f"  [green]✓ default → {model_name}[/green]")
        providers_needed.add(provider)
    else:
        _print_catalog(flat)
        console.print()
        default_idx = next((i + 1 for i, (p, _) in enumerate(flat) if p == "deepseek"), 1)
        for agent_name, cls in AGENT_CLASSES.items():
            label = f"{icon_text(agent_name)} {cls.display_name} model #"
            picked = _pick_from_catalog(flat, label, default_idx=default_idx)
            if not picked:
                continue
            prov, model_name = picked
            set_config(f"model.{agent_name}", model_name)
            providers_needed.add(prov)
            console.print(f"  [green]✓ {icon_text(agent_name)} {agent_name} → {model_name}[/green]")

    # Step 3: collect API keys
    console.print()
    console.print(Rule("  Step 3 — API keys  ", align="left", style="dim"))
    _collect_keys(providers_needed)

    console.print()
    console.print("  [green]✓ Setup complete[/green]")
    cfg_path = USER_CONFIG_DIR / "config.yaml"
    console.print(f"  [dim]config saved to {cfg_path}[/dim]")


@app.command()
def init() -> None:
    """Run the setup wizard, then optionally drop into chat."""
    _run_setup_wizard()
    answer = console.input("  Enter chat now? [Y/n]: ").strip().lower()
    if answer in ("", "y", "yes"):
        asyncio.run(_interactive())
    else:
        console.print("\n  [dim]Run `wacode` when ready.[/dim]\n")


# -- Version ---------------------------------------------------------------


@app.command()
def version() -> None:
    """Show version information."""
    console.print(f"  Weather Agents [bold]v{__version__}[/bold]")

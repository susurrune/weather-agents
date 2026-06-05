"""Live orchestration dashboard for ``sky task``.

A self-contained Rich ``Live`` panel showing real-time multi-agent execution:
a progress bar, a per-task status table with spinners, and the current tool
activity. Extracted from ``cli/main.py`` so the orchestration UI is a single,
testable unit rather than buried in the 5k-line CLI module.

The orchestrator drives it through callbacks: ``on_start`` / ``on_done`` per
task, ``on_tool_status`` / ``on_tool_done`` per tool, and ``merge_tasks`` when
the judge loop re-plans mid-run.
"""

from __future__ import annotations

import time
from typing import Any

from rich import box
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from weather_agents.cli.console import console
from weather_agents.core.agent import TaskState
from weather_agents.core.icons import AGENT_COLORS, icon_text

# Per-state row colour in the task table.
_TASK_STATE_COLOR: dict[TaskState, str] = {
    TaskState.PENDING: "dim",
    TaskState.RUNNING: "bold cyan",
    TaskState.COMPLETED: "green",
    TaskState.FAILED: "red",
    TaskState.SKIPPED: "dim",
}


class TaskDashboard:
    """Rich Live dashboard for real-time orchestration execution.

    Shows a live-updating panel with a progress bar, task status table,
    and current activity during ``sky task`` execution. Transient — the
    dashboard disappears once orchestration completes, leaving only the
    final summary on screen.
    """

    def __init__(self, goal: str, tasks: list[Any]) -> None:
        self._goal = goal
        self._task_states: dict[str, TaskState] = {}
        self._task_agents: dict[str, str] = {}
        self._task_descs: dict[str, str] = {}
        self._task_deps: dict[str, list[str]] = {}
        self._done: int = 0
        self._total: int = 0
        self._current_id: str = ""
        self._current_label: str = ""
        self._current_agent: str = ""
        self._tool_label: str = ""
        self._tool_result: str = ""
        self._start_ts: float = 0.0
        self._live: Live | None = None

        self._ingest_tasks(tasks)

    def _ingest_tasks(self, tasks: list[Any]) -> None:
        """Register or refresh task metadata. Used both at construction
        and when the orchestrator re-plans mid-run: the dashboard needs
        to know about the new tasks' agents/descriptions/deps, otherwise
        the table renders them with "?" placeholders and the progress
        bar sees a stale denominator (e.g. "3/2  150%")."""
        for t in tasks:
            tid = t.id
            # Existing tasks: keep their (already-tracked) state; refresh
            # metadata in case the planner overwrote description / deps.
            if tid not in self._task_states:
                self._task_states[tid] = t.status
                if t.status in (
                    TaskState.COMPLETED,
                    TaskState.FAILED,
                    TaskState.SKIPPED,
                ):
                    self._done += 1
            self._task_agents[tid] = t.assigned_to or "?"
            self._task_descs[tid] = t.description
            self._task_deps[tid] = t.all_deps
        self._total = len(self._task_states)

    def merge_tasks(self, tasks: list[Any]) -> None:
        """Public hook for the orchestrator's replan path."""
        self._ingest_tasks(tasks)
        self._refresh()

    @property
    def _elapsed(self) -> str:
        if not self._start_ts:
            return "0:00"
        s = int(time.monotonic() - self._start_ts)
        return f"{s // 60:02d}:{s % 60:02d}"

    def start(self) -> None:
        self._start_ts = time.monotonic()
        self._live = Live(
            self._render(),
            console=console,
            refresh_per_second=4,
            transient=True,
        )
        self._live.start()

    def stop(self) -> None:
        if self._live:
            self._live.stop()
            self._live = None

    def _refresh(self) -> None:
        if self._live:
            self._live.update(self._render())

    def on_start(self, task_id: str, description: str, agent: str) -> None:
        self._task_states[task_id] = TaskState.RUNNING
        self._current_id = task_id
        self._current_label = description
        self._current_agent = agent
        self._tool_label = ""
        self._tool_result = ""
        self._refresh()

    def on_tool_status(self, label: str) -> None:
        self._tool_label = label
        self._tool_result = ""
        self._refresh()

    def on_tool_done(self, label: str, success: bool) -> None:
        self._tool_label = label
        self._tool_result = "✓" if success else "✗"
        self._refresh()

    def on_done(self, task_id: str, success: bool) -> None:
        self._task_states[task_id] = TaskState.COMPLETED if success else TaskState.FAILED
        self._done += 1
        self._current_id = ""
        self._current_label = ""
        self._current_agent = ""
        self._tool_label = ""
        self._tool_result = ""
        self._refresh()

    def update_task_state(self, task_id: str, state: TaskState) -> None:
        """Direct state update for cycle-detected / pre-completed tasks."""
        self._task_states[task_id] = state
        if state in (TaskState.COMPLETED, TaskState.FAILED, TaskState.SKIPPED):
            self._done += 1
        self._refresh()

    def _render(self) -> Panel:
        grid = Table(show_header=False, box=None, padding=(0, 0), expand=True)
        grid.add_column(ratio=1)

        # ── Progress row ──
        pct = (self._done / self._total * 100) if self._total else 0
        bar_filled = int(pct / 5)
        bar_empty = 20 - bar_filled
        bar_color = "green" if self._done == self._total else "yellow"
        # Text.append(str, style=...) does NOT parse rich markup — it
        # appends the literal characters. That's why "[yellow]...[/]"
        # was showing verbatim in the panel. Build the colored segments
        # by appending with the style applied, then concatenate.
        progress = Text()
        progress.append("  ")
        progress.append(f"{self._done}/{self._total}", style=f"bold {bar_color}")
        progress.append("  ")
        progress.append("━" * bar_filled, style=f"bold {bar_color}")
        progress.append("╌" * bar_empty, style="dim")
        progress.append("  ")
        progress.append(f"{int(pct)}%  ", style="dim")
        progress.append(self._elapsed, style="dim")
        grid.add_row(progress)
        grid.add_row(Text(""))

        # ── Task table ──
        if self._total:
            tbody = Table(show_header=False, box=None, padding=(0, 1), expand=True)
            tbody.add_column(width=4)  # sequence number / status
            tbody.add_column(width=4)  # agent icon (now wider for richer glyphs)
            tbody.add_column(ratio=1)
            tbody.add_column(width=14)

            for idx, (tid, state) in enumerate(self._task_states.items(), 1):
                color = _TASK_STATE_COLOR.get(state, "dim")
                agent_name = self._task_agents.get(tid, "?")
                desc = self._task_descs.get(tid, "")[:55]
                deps = self._task_deps.get(tid, [])
                dep_str = f"← {','.join(deps[:3])}" if deps else ""
                ag_icon = icon_text(agent_name)
                ag_color = AGENT_COLORS.get(agent_name, "white")

                # Position column: live spinner for the running task,
                # static number for everything else. The spinner makes
                # "which task is the agent on right now" obvious without
                # the user reading colors.
                pos_cell: Any
                if state == TaskState.RUNNING:
                    pos_cell = Spinner(
                        "dots", text=Text(f"{idx}", style=f"bold {color}"), style=color
                    )
                else:
                    glyph = (
                        "✓"
                        if state == TaskState.COMPLETED
                        else (
                            "✗"
                            if state == TaskState.FAILED
                            else ("–" if state == TaskState.SKIPPED else "·")
                        )
                    )
                    pos_cell = Text(f"{idx}{glyph}", style=color)

                tbody.add_row(
                    pos_cell,
                    Text(ag_icon, style=f"bold {ag_color}"),
                    Text(desc, style=color),
                    Text(dep_str, style="dim"),
                )
            grid.add_row(tbody)
            grid.add_row(Text(""))

        # ── Current activity ──
        if self._current_label:
            sep = Rule(style="dim", characters="─")
            grid.add_row(sep)
            agent_color = AGENT_COLORS.get(self._current_agent, "white")
            cur = Text()
            cur.append(f"  {icon_text(self._current_agent)} ", style=agent_color)
            cur.append(self._current_label, style=agent_color)
            grid.add_row(cur)
        if self._tool_label:
            # Use a 2-col mini-table so the in-flight tool row gets a
            # live spinner (refreshes via Live's 4Hz tick). Completed /
            # failed rows show the static glyph instead. Without this
            # users see a static dim "·" while the tool runs and have
            # no signal that the system is still working.
            tool_row = Table(show_header=False, box=None, padding=(0, 0), expand=False)
            tool_row.add_column(width=2)
            tool_row.add_column(ratio=1)
            label_text = Text(f" {self._tool_label}", style="dim")
            if self._tool_result == "✓":
                tool_row.add_row(Text(" ✓", style="green"), label_text)
            elif self._tool_result == "✗":
                tool_row.add_row(Text(" ✗", style="red"), label_text)
            else:
                tool_row.add_row(
                    Spinner("dots", style="dim"),
                    label_text,
                )
            grid.add_row(tool_row)

        return Panel(
            grid,
            title="[bold]Task Orchestration[/bold]",
            border_style="dim",
            box=box.ROUNDED,
            padding=(1, 2),
        )

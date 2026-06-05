"""Claude Code-style tool-call status display for the streaming REPL.

Renders ``IN`` (tool starting) and ``OUT`` (tool finished) lines, collapsing
consecutive calls on the same (category, tool, path) into a single indented
group — so ten reads of the same file show as one header with a tidy tree
rather than ten near-identical lines.

The grouping buffer is module-private state here (was a pair of ``global``
buffers floating in the 4.7k-line ``cli/main.py``). Keeping it encapsulated in
its own module is the whole point: the REPL just calls ``print_tool_in`` /
``print_tool_out`` / ``flush_tool_group`` and never touches the buffer.
"""

from __future__ import annotations

from weather_agents.cli.console import console

# Tool name → display category (Bash / File / Git / Search / Web / ...).
_TOOL_CATEGORIES: dict[str, str] = {
    "read_file": "File",
    "write_file": "File",
    "edit_file": "File",
    "move_file": "File",
    "copy_file": "File",
    "delete_file": "File",
    "list_directory": "File",
    "file_search": "File",
    "tree": "File",
    "grep": "Search",
    "code_search": "Search",
    "web_search": "Search",
    "fetch_page": "Web",
    "http_get": "Web",
    "http_post": "Web",
    "shell_exec": "Bash",
    "get_cwd": "Bash",
    "task_done": "Done",
    "lint_file": "Lint",
    "scan_deps": "Scan",
    "delegate_to": "Delegate",
    "use_skill": "Skill",
    "list_skills": "Skill",
    "git_status": "Git",
    "git_diff": "Git",
    "git_log": "Git",
    "git_add": "Git",
    "git_commit": "Git",
    "git_checkout": "Git",
}


def tool_category(tool_name: str) -> str:
    """Map tool name to display category (Bash / File / Git / Search / Web / ...)."""
    return _TOOL_CATEGORIES.get(tool_name, "Tool")


# Per-category accent color used for the tool block's left rule. Falls
# back to dim cyan for unknown categories so unfamiliar tools still get
# a visual frame.
_TOOL_CAT_COLORS: dict[str, str] = {
    "Bash": "yellow",
    "File": "cyan",
    "Git": "magenta",
    "Search": "blue",
    "Web": "green",
    "Lint": "bright_magenta",
    "Scan": "bright_yellow",
    "Delegate": "bright_cyan",
    "Skill": "bright_green",
    "Tool": "dim cyan",
}


# ── Tool-group buffer — consecutive same-(cat, path) calls collapse into a tree ──

_tool_group_key: tuple | None = None
_tool_group_entries: list[dict] = []


def normalize_tool_path(tool_name: str, args: dict) -> str:
    """Grouping key: the file/directory/url/pattern a tool operates on."""
    for key in ("path", "dst", "src", "directory", "pattern", "url", "query"):
        v = args.get(key)
        if v and isinstance(v, str):
            return v
    return ""


def tool_args_str(args: dict, label: str, max_v: int = 80) -> str:
    """Build a compact args suffix, skipping values already in the label."""
    if not args:
        return ""
    parts: list[str] = []
    for k, v in args.items():
        if v is None or v == "":
            continue
        v_str = str(v).replace("\n", " ")
        if len(v_str) > max_v:
            v_str = v_str[: max_v - 3] + "..."
        if v_str and v_str in label:
            continue
        parts.append(f"{k}={v_str}")
    return "  " + " ".join(parts) if parts else ""


def _print_tool_out_flat(label: str, success: bool, result: str, cat: str, cat_color: str) -> None:
    """Print a single-line tool-finished marker (used only for single-entry groups)."""
    status_icon = "[green]✓[/]" if success else "[red]✗[/]"
    label_short = label if len(label) < 60 else label[:57] + "..."
    line = f"  [{cat_color}]│[/] {status_icon} [{cat_color}]{cat}[/] [bold]{label_short}[/]"
    if result:
        result_flat = result.replace("\n", " ").strip()
        if len(result_flat) > 120:
            result_flat = result_flat[:117] + "..."
        line += f"  [dim]{result_flat}[/]"
    console.print(line)


def flush_tool_group() -> None:
    """Render the buffered tool group and reset it."""
    global _tool_group_key, _tool_group_entries
    if not _tool_group_entries:
        return

    cat = tool_category(_tool_group_entries[0]["tool_name"])
    cat_color = _TOOL_CAT_COLORS.get(cat, "dim cyan")
    icon = "●" if cat == "Bash" else "·"

    if len(_tool_group_entries) == 1:
        # Single entry — unchanged format.
        e = _tool_group_entries[0]
        label_short = e["label"] if len(e["label"]) < 60 else e["label"][:57] + "..."
        arg_str = tool_args_str(e.get("args", {}), e["label"])
        console.print(
            f"  [{cat_color}]│[/] [{cat_color}]{icon}[/] [{cat_color}]{cat}[/]"
            f" [bold]{label_short}[/]{arg_str}"
        )
        if e["result"] is not None:
            _print_tool_out_flat(e["label"], e["success"], e["result"], cat, cat_color)
    else:
        # Multiple consecutive same-path calls — indent under a single header.
        first = _tool_group_entries[0]
        first_label = first["label"] if len(first["label"]) < 60 else first["label"][:57] + "..."
        console.print(
            f"  [{cat_color}]│[/] [{cat_color}]{icon}[/] [{cat_color}]{cat}[/]"
            f" [bold]{first_label}[/]"
        )
        for i, e in enumerate(_tool_group_entries):
            status = "[green]✓[/]" if e["success"] else "[red]✗[/]"
            detail = tool_args_str(e.get("args", {}), "", max_v=50)
            marker = "└" if i == len(_tool_group_entries) - 1 else "├"
            line = f"  [{cat_color}]│[/]   [{cat_color}]{marker}[/] "
            if e["result"] is None:
                line += f"[dim]· {detail}[/]" if detail else "[dim]·[/]"
            elif detail:
                line += f"{detail}  {status}"
            else:
                line += f"{status}"
            console.print(line)

    _tool_group_key = None
    _tool_group_entries = []


def print_tool_in(label: str, tool_name: str, args: dict) -> None:
    """Buffer or flush-print a tool-starting line, grouping same-path calls."""
    cat = tool_category(tool_name)
    path = normalize_tool_path(tool_name, args)
    new_key = (cat, tool_name, path)

    global _tool_group_key, _tool_group_entries
    if _tool_group_key is not None and new_key != _tool_group_key:
        flush_tool_group()

    if _tool_group_key is None:
        _tool_group_key = new_key

    _tool_group_entries.append(
        {"label": label, "tool_name": tool_name, "args": args, "success": True, "result": None}
    )


def print_tool_out(label: str, tool_name: str, success: bool, result: str = "") -> None:
    """Mark the last buffered tool entry as complete."""
    global _tool_group_entries
    if _tool_group_entries:
        _tool_group_entries[-1]["success"] = success
        _tool_group_entries[-1]["result"] = result

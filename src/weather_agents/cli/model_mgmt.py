"""Model & API-Key management helpers for the CLI REPL."""

from __future__ import annotations

import os
import sys

from rich.live import Live
from rich.table import Table
from rich.text import Text

from weather_agents.cli.console import console
from weather_agents.cli.keys import get_key as _get_key
from weather_agents.cli.pickers import arrow_pick_from_list as _arrow_pick_from_list
from weather_agents.core.agent import TaskState
from weather_agents.core.config import (
    delete_config,
    load_model_catalog,
    set_config,
)
from weather_agents.core.factory import AGENT_CLASSES
from weather_agents.core.icons import icon_text

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
                return str(entries[selected_idx].get("name", ""))
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


def _refresh_agent_identity(ctx) -> None:
    """Re-render every agent's system prompt so the runtime-identity
    block (current model id) matches the just-updated config. Without
    this, switching models via ``/model`` left the agent claiming the
    OLD model id in its system prompt — leading to "I'm Claude" replies
    after the user already switched to DeepSeek. Cheap: only mutates
    the existing system message in short_term, no LLM call.
    """
    for agent in ctx.agent_map.values():
        rebuild = getattr(agent, "_rebuild_system_prompt", None)
        if callable(rebuild):
            try:
                rebuild()
            except Exception:
                # Re-rendering is best-effort cosmetic; never let a
                # transient failure block the user's /model change.
                continue


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
                _refresh_agent_identity(ctx)
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
            _refresh_agent_identity(ctx)
            console.print(f"  [green]all agents -> {model_name}[/green]")
        else:
            model = _interactive_model_select("Select model for all agents")
            if model:
                for name in AGENT_CLASSES:
                    set_config(f"model.{name}", model)
                    agent_cfg = getattr(ctx.config.agents, name)
                    agent_cfg.model = model
                _refresh_agent_identity(ctx)
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
            _refresh_agent_identity(ctx)
            console.print(f"\n  [green]{icon_text(agent_name)} {agent_name} -> {model}[/green]")
        return

    # ── /model <agent> <model> — direct set ──────────────────────────────
    if len(tokens) == 2 and tokens[0] in AGENT_CLASSES:
        agent_name, model_name = tokens
        if model_name.lower() == "default":
            delete_config(f"model.{agent_name}")
            agent_cfg = getattr(ctx.config.agents, agent_name)
            agent_cfg.model = ""
            _refresh_agent_identity(ctx)
            console.print(f"  [green]{icon_text(agent_name)} {agent_name} -> default[/green]")
        else:
            set_config(f"model.{agent_name}", model_name)
            agent_cfg = getattr(ctx.config.agents, agent_name)
            agent_cfg.model = model_name
            _refresh_agent_identity(ctx)
            console.print(f"  [green]{icon_text(agent_name)} {agent_name} -> {model_name}[/green]")
        return

    # ── /model <model> — direct set default ─────────────────────────────
    model_name = arg
    ok, msg = set_config("default_model", model_name)
    if ok:
        ctx.config.llm.default_model = model_name
        _refresh_agent_identity(ctx)
        console.print(f"  [green]model -> {model_name}[/green]")
    else:
        console.print(f"  [red]{msg}[/red]")


def _print_provider_status(ctx) -> None:
    """Render every catalog provider grouped by region with key status.

    The catalog is the single source of truth — every provider sky knows
    how to route to (35+ entries as of round 11). This view answers
    "which providers can I use right now?" at a glance: ● means a key
    is configured (in config OR an env var), ○ means missing.
    """
    from weather_agents.core.config import load_provider_catalog

    catalog = load_provider_catalog()
    if not catalog:
        console.print("  [dim]no provider catalog loaded[/dim]")
        return

    # Group by region. Display order — US first, then CN, then aggregator,
    # then EU, then Local — chosen by historical user demand.
    region_order = ["US", "CN", "Aggregator", "EU", "Local"]
    groups: dict[str, list[tuple[str, dict]]] = {r: [] for r in region_order}
    for prov_id, entry in catalog.items():
        region = entry.get("region", "Other")
        groups.setdefault(region, []).append((prov_id, entry))

    configured = set(ctx.config.llm.api_keys.keys())

    console.print()
    for region in region_order + [r for r in groups if r not in region_order]:
        provs = groups.get(region) or []
        if not provs:
            continue
        console.print(f"  [bold dim]{region}[/bold dim]")
        for prov_id, entry in sorted(provs):
            env_var = entry.get("env_var", "")
            has_config_key = prov_id in configured
            has_env = bool(env_var and os.environ.get(env_var))
            ready = has_config_key or has_env
            dot = "[green]●[/green]" if ready else "[dim]○[/dim]"
            note = (entry.get("notes") or "")[:60]
            source = (
                ""
                if not ready
                else (" [dim](config)[/dim]" if has_config_key else " [dim](env)[/dim]")
            )
            console.print(f"  {dot}  [cyan]{prov_id:<22}[/cyan]  [dim]{note}[/dim]{source}")
        console.print()
    console.print(
        "  [dim]/apikey set                  pick a provider and enter the key\n"
        "  /apikey del                  pick a provider to remove\n"
        "  /apikey set <provider> <key> direct set (still works)[/dim]"
    )


def _provider_picker_items(catalog: dict[str, dict], configured: set[str]) -> list[tuple[str, str]]:
    """Build (key, label) tuples for the arrow picker. Region prefix
    sorts US/CN/EU/Local/Aggregator together; the label embeds the
    region tag, env var name, and a configured/missing marker so the
    user can decide in one glance."""
    items: list[tuple[str, str]] = []
    region_rank = {"US": 0, "CN": 1, "Aggregator": 2, "EU": 3, "Local": 4}
    for prov_id, entry in sorted(
        catalog.items(),
        key=lambda kv: (region_rank.get(kv[1].get("region", ""), 99), kv[0]),
    ):
        region = entry.get("region", "?")
        env_var = entry.get("env_var", "")
        note = (entry.get("notes") or "")[:40]
        mark = "●" if prov_id in configured or os.environ.get(env_var) else "○"
        label = f"[{region:<10}] {mark} {prov_id:<22} {note}"
        items.append((prov_id, label))
    return items


def _read_api_key_input(provider: str) -> str | None:
    """Securely prompt for an API key. Uses getpass when stdin is a TTY
    so the key isn't echoed. Cancels (returns None) on Ctrl-C / empty."""
    import getpass

    try:
        if sys.stdin.isatty():
            raw = getpass.getpass(f"  {provider} key (hidden): ").strip()
        else:
            raw = console.input(f"  {provider} key: ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print()
        return None
    return raw or None


def _handle_apikey_command(cmd: str, ctx) -> None:
    parts = cmd.strip().split(maxsplit=2)

    if len(parts) == 1:
        _print_provider_status(ctx)
        return

    action = parts[1].lower()

    # ── /apikey set ──────────────────────────────────────────────────
    if action in ("set", "add"):
        from weather_agents.core.config import load_provider_catalog

        # Direct form: `/apikey set <provider> <key>` — unchanged.
        if len(parts) == 3:
            tokens = parts[2].strip().split(maxsplit=1)
            if len(tokens) == 2:
                provider, key = tokens
                _apikey_save(ctx, provider.lower(), key)
                return
            # Fall through to picker when only the provider was given.
            provider = parts[2].strip().lower()
            if provider and sys.stdin.isatty():
                entered = _read_api_key_input(provider)
                if entered:
                    _apikey_save(ctx, provider, entered)
                return
            console.print("  [red]usage: /apikey set <provider> <key>[/red]")
            return

        # No args → interactive: pick provider, then prompt for key.
        if not sys.stdin.isatty():
            console.print("  [red]usage: /apikey set <provider> <key>[/red]")
            return
        catalog = load_provider_catalog()
        configured = set(ctx.config.llm.api_keys.keys())
        items = _provider_picker_items(catalog, configured)
        picked = _arrow_pick_from_list(
            items,
            title=f"  Set API key  ({len(items)} providers)",
            active_keys=configured,
            viewport=18,
        )
        if not picked:
            return
        entered = _read_api_key_input(picked)
        if entered:
            _apikey_save(ctx, picked, entered)
        return

    # ── /apikey del ──────────────────────────────────────────────────
    if action in ("del", "delete", "rm", "remove"):
        from weather_agents.core.config import load_provider_catalog

        if len(parts) >= 3:
            provider = parts[2].strip().lower()
            _apikey_remove(ctx, provider)
            return

        # No args → pick from CONFIGURED providers only.
        configured_list: list[str] = list(ctx.config.llm.api_keys.keys())
        if not configured_list:
            console.print("  [dim]no API keys configured[/dim]")
            return
        if not sys.stdin.isatty():
            console.print("  [red]usage: /apikey del <provider>[/red]")
            return
        catalog = load_provider_catalog()
        items = [
            (
                p,
                f"{p:<22} {(catalog.get(p, {}).get('notes') or '')[:50]}",
            )
            for p in sorted(configured_list)
        ]
        picked = _arrow_pick_from_list(
            items,
            title=f"  Remove API key  ({len(items)} configured)",
        )
        if picked:
            _apikey_remove(ctx, picked)
        return

    console.print("  [red]usage: /apikey [set [<provider> [<key>]] | del [<provider>]][/red]")


def _apikey_save(ctx, provider: str, key: str) -> None:
    """Persist an API key + sync to env vars + give visible feedback."""
    ok, msg = set_config(f"api_key.{provider}", key)
    if not ok:
        console.print(f"  [red]{msg}[/red]")
        return
    ctx.config.llm.api_keys[provider] = key
    _sync_api_keys_to_env({provider: key})
    console.print(f"  [green]+ {provider} key saved[/green]")


def _apikey_remove(ctx, provider: str) -> None:
    """Delete an API key from config + clear the env var."""
    from weather_agents.core.config import get_provider_env_var

    ok, msg = delete_config(f"api_key.{provider}")
    if not ok:
        console.print(f"  [red]{msg}[/red]")
        return
    ctx.config.llm.api_keys.pop(provider, None)
    env_var = get_provider_env_var(provider)
    os.environ.pop(env_var, None)
    console.print(f"  [green]- {provider} key removed[/green]")


# -- Task orchestration ----------------------------------------------------


TASK_STATE_ICONS: dict[TaskState, tuple[str, str]] = {
    TaskState.PENDING: ("◌", "dim"),
    TaskState.RUNNING: ("●", "cyan"),
    TaskState.COMPLETED: ("✓", "green"),
    TaskState.FAILED: ("✗", "red"),
    TaskState.SKIPPED: ("–", "dim"),
}
def _provider_for_model(model: str) -> str:
    """Resolve a model id back to its catalog provider key.

    Walks ``models.yaml`` for an exact match first (so ``gpt-5`` resolves
    to ``openai`` via the ``provider:`` field), then falls back to a
    LiteLLM-style prefix check (``<provider>/<rest>``). Last-resort
    keyword heuristics catch ids that aren't in the catalog yet. Using
    the catalog instead of the old hard-coded if/elif chain means
    adding a new provider in providers.yaml just works — no Python
    edits needed in the wizard.
    """
    if not model:
        return "openai"
    m = model.strip()
    m_lower = m.lower()

    # 1. Exact lookup in the model catalog.
    try:
        catalog = load_model_catalog()
        for _provider_group, models in catalog.items():
            for entry in models:
                if entry.get("name") == m:
                    p = entry.get("provider")
                    if isinstance(p, str) and p:
                        from weather_agents.core.config import resolve_provider_alias

                        return resolve_provider_alias(p)
    except Exception:
        pass

    # 2. <provider>/<rest> prefix — try the canonical id, then aliases.
    if "/" in m_lower:
        prefix = m_lower.split("/", 1)[0]
        from weather_agents.core.config import (
            load_provider_catalog,
            resolve_provider_alias,
        )

        provider_ids = set(load_provider_catalog().keys())
        if prefix in provider_ids:
            return prefix
        resolved = resolve_provider_alias(prefix)
        if resolved in provider_ids:
            return resolved

    # 3. Fallback keyword heuristics for unrecognised ids.
    if m_lower.startswith(("claude", "anthropic")):
        return "anthropic"
    if m_lower.startswith(("gpt", "openai", "o1", "o3", "o4")):
        return "openai"
    if "deepseek" in m_lower:
        return "deepseek"
    if "gemini" in m_lower:
        return "google_gemini"
    return "openai"



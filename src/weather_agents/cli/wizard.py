"""First-run setup wizard for the CLI.

Walks a new user through: picking a model strategy (unified vs per-agent),
selecting models from the catalog via the arrow picker, entering API keys
(hidden), and optionally a couple of profile facts so agents feel personal from
the first message. Invoked by ``sky init`` and on first run of ``sky chat`` when
nothing is configured yet.

Lives in its own module (it depends only on cli.pickers / cli.console + core
config) so the 4k-line main module isn't carrying ~250 lines of one-time
onboarding flow.
"""

from __future__ import annotations

import os
import sys

from rich import box
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from weather_agents.cli.console import console
from weather_agents.cli.pickers import arrow_pick_from_list, flatten_catalog
from weather_agents.core.config import (
    USER_CONFIG_DIR,
    delete_config,
    load_config,
    load_model_catalog,
    set_config,
)
from weather_agents.core.factory import AGENT_CLASSES
from weather_agents.core.icons import icon_text


def collect_keys(providers: set[str]) -> None:
    """Prompt for one API key per cloud provider in the set.

    Uses the catalog for env-var resolution + docs URL, hidden ``getpass``
    input on TTY so pasted keys aren't echoed, and shows the user where
    to grab the key for each provider. Skips local-only providers
    (Ollama / vLLM / LM Studio / llama.cpp) since they don't need keys.
    """
    from weather_agents.core.config import get_provider_env_var, load_provider_catalog

    LOCAL = {"ollama", "lm_studio", "vllm", "llamacpp", "local"}
    cloud = sorted(p for p in providers if p not in LOCAL)
    if not cloud:
        console.print("  [dim]All chosen models run locally — no API keys needed.[/dim]")
        return

    catalog = load_provider_catalog()
    console.print(f"\n  [bold]API keys needed for:[/bold] [cyan]{', '.join(cloud)}[/cyan]")
    console.print(
        "  [dim](key is hidden as you type; stored in plain YAML at ~/.skyloom/config.yaml)[/dim]\n"
    )

    import getpass

    for provider in cloud:
        entry = catalog.get(provider, {})
        env_var = entry.get("env_var") or get_provider_env_var(provider)
        docs = entry.get("docs_url") or ""
        notes = entry.get("notes") or ""

        cfg = load_config()
        current = cfg.llm.api_keys.get(provider) or os.environ.get(env_var) or ""
        if current:
            console.print(
                f"  [green]●[/green] [cyan]{provider}[/cyan]  "
                f"[dim]already configured ({'env' if not cfg.llm.api_keys.get(provider) else 'config'}) — "
                f"Enter to keep[/dim]"
            )
        else:
            console.print(f"  [dim]○[/dim] [cyan]{provider}[/cyan]  [dim]{notes}[/dim]")
            if docs:
                console.print(f"     [dim]get a key: {docs}[/dim]")

        try:
            if sys.stdin.isatty():
                key = getpass.getpass(f"     {env_var} (hidden): ").strip()
            else:
                key = console.input(f"     {env_var}: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n  [yellow]setup cancelled[/yellow]")
            return

        if key:
            ok, msg = set_config(f"api_key.{provider}", key)
            color = "green" if ok else "red"
            console.print(f"     [{color}]{msg}[/{color}]\n")
        elif not current:
            console.print(
                f"     [dim]skipped — you can set it later with /apikey set {provider}[/dim]\n"
            )
        else:
            console.print("     [dim](kept)[/dim]\n")


def wizard_pick_model(prompt_title: str, default_id: str | None = None) -> tuple[str, str] | None:
    """Arrow-pick a model from the catalog (117 entries across 34
    providers). Returns ``(provider_id, model_name)`` or None on cancel.

    Items render as ``<provider>  <model_id>`` so the user can filter
    by typing either side. Non-TTY callers fall back to a numeric
    prompt the same way the legacy wizard worked.
    """
    catalog = load_model_catalog()
    if not catalog:
        console.print("\n  [red]No model catalog found. Reinstall and try again.[/red]")
        return None

    # Build picker items with a stable order — provider group, then
    # the order yaml defined the models in (typically newest first).
    # ``default_id`` is informational — the picker doesn't take a
    # starting-cursor arg yet; users see the current default in the
    # title bar and can press Esc to keep it.
    flat = flatten_catalog(catalog)
    items: list[tuple[str, str]] = []
    for prov, name in flat:
        label = f"{prov:<22} {name}"
        items.append((name, label))

    if not sys.stdin.isatty():
        # Numeric fallback for piped / non-interactive runs.
        for i, (prov, name) in enumerate(flat, 1):
            console.print(f"  {i:>3}. [cyan]{name}[/cyan]  [dim]{prov}[/dim]")
        raw = console.input(f"\n  {prompt_title} #: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(flat):
            return flat[int(raw) - 1]
        return None

    title = f"  {prompt_title}"
    if default_id:
        title += f"  [Esc keeps: {default_id}]"
    picked = arrow_pick_from_list(
        items,
        title=title,
        viewport=20,
    )
    if not picked:
        return None
    for prov, name in flat:
        if name == picked:
            return prov, name
    # Shouldn't reach — picked came from items[][0] which is the name.
    return None


def collect_profile_basics() -> None:
    """Ask for a name + how to be addressed so agents feel personal from turn 1.

    Entirely optional — blank answers skip the field. Stored in the same local
    profile (用户画像) the agents read each turn. We keep this to two questions:
    a long form on first run is friction; the agents fill in the rest over time
    via set_user_profile.
    """
    from weather_agents.core.profile import load_profile, set_profile_field

    existing = load_profile()
    if existing:
        # Re-running init with a profile already present — don't nag.
        console.print("  [dim]已有画像，跳过。用 `sky profile` 查看或修改。[/dim]")
        return

    console.print("  [dim]两个小问题，直接回车可跳过。智能体会在对话中慢慢补全其余。[/dim]\n")
    try:
        name = console.input("  你的名字 / 我该怎么称呼你？ ").strip()
        if name:
            set_profile_field("称呼", name)
        addr = console.input("  想让智能体用什么语气？(如：朋友 / 正式 / 随意) ").strip()
        if addr:
            set_profile_field("相处语气", addr)
    except (EOFError, KeyboardInterrupt):
        console.print("\n  [dim]跳过。[/dim]")
        return
    if name or addr:
        console.print("  [green]✓ 记下了[/green]")
    else:
        console.print("  [dim]跳过，之后随时可以 `sky profile set` 添加。[/dim]")


def run_setup_wizard() -> None:
    """Walk the user through choosing a model strategy and storing API keys.

    Modernized in round 13:
      - Step 2 uses the arrow picker (↑↓ / type to filter / Enter) over
        the full 117-model catalog instead of a static numbered scroll.
      - Step 3 reads the env var name + docs URL from providers.yaml
        and uses hidden ``getpass`` input on TTY so keys aren't echoed.

    Does NOT enter chat — the caller decides whether to launch _interactive().
    """
    console.print()
    console.print(
        Panel(
            "[bold]Skyloom Setup[/bold]\n[dim]3 steps · 34 providers · 117 models[/dim]",
            border_style="dim cyan",
            box=box.ROUNDED,
            padding=(1, 2),
            width=50,
        )
    )

    catalog = load_model_catalog()
    if not catalog:
        console.print("\n  [red]No model catalog found. Reinstall and try again.[/red]")
        return

    # Sensible default — DeepSeek flash is fast + cheap and a good
    # starting point for users who don't know what to pick. Any model
    # already configured wins, so re-running init keeps the current
    # default at the top of the picker.
    current_default = load_config().llm.default_model
    fallback_default = "deepseek/deepseek-v4-flash"
    default_id = current_default or fallback_default

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
    console.print(
        Rule(
            "  Step 2 — Model selection  (↑↓ to move · type to filter · Enter to pick · Esc to skip)  ",
            align="left",
            style="dim",
        )
    )

    if mode == "1":
        picked = wizard_pick_model("Pick default model", default_id=default_id)
        if not picked:
            console.print("  [yellow]setup cancelled[/yellow]")
            return
        provider, model_name = picked
        set_config("default_model", model_name)
        for ag in AGENT_CLASSES:
            delete_config(f"model.{ag}")
        console.print(f"  [green]✓ default → {model_name}[/green]")
        providers_needed.add(provider)
    else:
        # Per-agent: pick for each. Esc on any one keeps the existing
        # config for that agent (or default if none set yet).
        console.print(
            "  [dim]Esc on any agent keeps its current model (or the global default).[/dim]\n"
        )
        for agent_name, cls in AGENT_CLASSES.items():
            label = f"{cls.display_name} ({agent_name})"
            picked = wizard_pick_model(f"Model for {label}", default_id=default_id)
            if not picked:
                console.print(f"  [dim]  {icon_text(agent_name)} {agent_name} → keep current[/dim]")
                continue
            prov, model_name = picked
            set_config(f"model.{agent_name}", model_name)
            providers_needed.add(prov)
            console.print(f"  [green]✓ {icon_text(agent_name)} {agent_name} → {model_name}[/green]")

    # Step 3: collect API keys
    console.print()
    console.print(Rule("  Step 3 — API keys  ", align="left", style="dim"))
    collect_keys(providers_needed)

    # Step 4: optional — a couple of facts so agents personalize from message 1.
    console.print()
    console.print(Rule("  Step 4 — 关于你 (可选)  ", align="left", style="dim"))
    collect_profile_basics()

    console.print()
    console.print("  [green]✓ Setup complete[/green]")
    cfg_path = USER_CONFIG_DIR / "config.yaml"
    console.print(f"  [dim]config saved to {cfg_path}[/dim]")
    console.print(
        "  [dim]tip: /apikey to inspect or change keys, /model to switch models later[/dim]"
    )

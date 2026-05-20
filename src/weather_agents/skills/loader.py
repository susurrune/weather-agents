"""Skill loader — discovers skills from .py modules and .md files (Anthropic format).

Skill sources, in registration order (earlier sources win on name conflicts):

  1. Python-coded skills bundled with wa (weather_agents.skills.*)
  2. Markdown skills bundled inside the wa package
     (weather_agents/config/skills/*.md)
  3. User-installed Anthropic-format skills at
     ``~/.weather-agents/skills/<name>/SKILL.md``
  4. Plugin-bundled skills at
     ``~/.weather-agents/plugins/marketplaces/<m>/{plugins,external_plugins}/<plugin>/skills/<name>/SKILL.md``
     — registered with the namespaced name ``<plugin>:<name>``.

Prior versions scanned Claude Code's ``~/.claude/skills/`` and
``~/.claude/plugins/`` directly. That coupled wa's skill set to Claude
Code's install, so reinstalling wa or uninstalling Claude Code silently
broke skills. wa now owns the directory under its own config root; a
one-time migration helper copies user-level skills from the legacy
``~/.claude/skills/`` location into the new wa-owned tree on demand
(``wa skill migrate`` / the ``/skills migrate`` REPL command).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from weather_agents.core.config import USER_CONFIG_DIR
from weather_agents.core.skill import Skill, SkillRegistry

# Canonical wa-owned locations (under ~/.weather-agents/). Created on
# demand by the migration helper; absence is fine — the loader returns
# an empty list and the agent proceeds with built-in skills only.
WA_SKILLS_DIR: Path = USER_CONFIG_DIR / "skills"
WA_PLUGINS_DIR: Path = USER_CONFIG_DIR / "plugins" / "marketplaces"

# Legacy Claude Code paths. Read ONLY by the migration helper — never
# scanned by the default loader. Keeping the constants here so tests
# and the migrate command share one source of truth.
LEGACY_CLAUDE_SKILLS_DIR: Path = Path(os.path.expanduser("~/.claude/skills"))
LEGACY_CLAUDE_PLUGINS_DIR: Path = Path(os.path.expanduser("~/.claude/plugins/marketplaces"))


def register_all_skills(registry: SkillRegistry | None = None) -> None:
    """Discover and register all skills wa knows about.

    Registration order (earlier sources win on name conflicts):

      1. Built-in SKILL.md files shipped inside the wa package
         (weather_agents/assets/builtin_skills/<name>/SKILL.md). These
         were Python-coded skill modules before the unification
         refactor; converting to SKILL.md means every skill in wa now
         uses the same on-disk format as Claude Code.
      2. User-installed Anthropic-format skills under
         ``~/.weather-agents/skills/<name>/SKILL.md``. A user file with
         the same name as a built-in shadows it — that's the official
         override path for customising a default.
      3. Plugin-bundled skills under
         ``~/.weather-agents/plugins/marketplaces/<m>/...``. Namespaced
         as ``<plugin>:<skill_name>`` so they coexist with user-level
         and built-in skills of the same bare name.
    """
    reg = registry or SkillRegistry()
    for skill in _get_builtin_md_skills():
        reg.register(skill)
    for skill in _get_wa_user_skills():
        # User SKILL.md files override built-ins by name. The override
        # path is intentional: a user dropping a customised
        # ``code_reviewer/SKILL.md`` into ~/.weather-agents/skills/
        # wins over the bundled default.
        reg.register(skill)
    for skill in _get_wa_plugin_skills():
        if skill.name not in reg.list_names():
            reg.register(skill)


def _get_builtin_md_skills() -> list[Skill]:
    """Load the SKILL.md files shipped inside the wa package.

    These cover the agents' default specialties (code_reviewer,
    web_research, task_planner, etc.) that used to be Python-coded
    skill modules. Storing them as SKILL.md makes the format consistent
    with Claude Code skills and lets users edit them with a text
    editor without touching Python source.
    """
    try:
        import importlib.resources

        ref = importlib.resources.files("weather_agents") / "assets" / "builtin_skills"
        path = Path(str(ref))
    except Exception:
        return []
    return _load_skills_from_skills_root(path)


def _load_skills_from_skills_root(base_path: Path) -> list[Skill]:
    """Scan a ``<base>/<name>/SKILL.md`` layout and return parsed skills.

    Shared by both user-level and plugin discovery — the only difference
    between them is the parent directory and whether names get a
    namespace prefix.
    """
    if not base_path.is_dir():
        return []

    skills: list[Skill] = []
    for entry in sorted(base_path.iterdir()):
        if not entry.is_dir() or entry.name.startswith(("_", ".")):
            continue
        skill_file = entry / "SKILL.md"
        if not skill_file.is_file():
            continue
        try:
            skill = Skill.from_markdown(skill_file)
            if skill:
                skill.resource_dir = str(entry)
                skills.append(skill)
        except Exception:
            continue
    return skills


def _get_wa_user_skills() -> list[Skill]:
    """User-installed Anthropic-format skills under wa's own folder.

    Path: ``~/.weather-agents/skills/<name>/SKILL.md``.
    Names are kept bare (no prefix) so a user-installed ``pptx`` shadows
    the plugin-namespaced ``anthropic-skills:pptx`` if both exist.
    """
    return _load_skills_from_skills_root(WA_SKILLS_DIR)


def _get_wa_plugin_skills() -> list[Skill]:
    """Plugin-bundled skills under wa's marketplaces tree.

    Layout (mirroring Claude Code's marketplace structure but rooted at
    wa's config dir):

      ~/.weather-agents/plugins/marketplaces/<marketplace>/plugins/<plugin>/skills/<skill>/SKILL.md
      ~/.weather-agents/plugins/marketplaces/<marketplace>/external_plugins/<plugin>/skills/<skill>/SKILL.md

    Each loaded skill is namespaced as ``<plugin>:<skill_name>`` so it
    coexists with any user-level skill of the same bare name.
    """
    if not WA_PLUGINS_DIR.is_dir():
        return []

    skills: list[Skill] = []
    for marketplace in sorted(WA_PLUGINS_DIR.iterdir()):
        if not marketplace.is_dir() or marketplace.name.startswith("."):
            continue
        for plugins_root_name in ("plugins", "external_plugins"):
            plugins_root = marketplace / plugins_root_name
            if not plugins_root.is_dir():
                continue
            for plugin_dir in sorted(plugins_root.iterdir()):
                if not plugin_dir.is_dir() or plugin_dir.name.startswith("."):
                    continue
                skills_dir = plugin_dir / "skills"
                if not skills_dir.is_dir():
                    continue
                for skill_entry in sorted(skills_dir.iterdir()):
                    if not skill_entry.is_dir() or skill_entry.name.startswith(("_", ".")):
                        continue
                    skill_file = skill_entry / "SKILL.md"
                    if not skill_file.is_file():
                        continue
                    try:
                        skill = Skill.from_markdown(skill_file)
                    except Exception:
                        continue
                    if not skill:
                        continue
                    bare_name = skill.name
                    skill.name = f"{plugin_dir.name}:{bare_name}"
                    skill.resource_dir = str(skill_entry)
                    skills.append(skill)
    return skills


# ── Migration ────────────────────────────────────────────────────────


def has_legacy_claude_skills() -> bool:
    """True if the legacy Claude Code skills directory has content.

    The REPL uses this to decide whether to surface a one-line hint
    suggesting ``/skills migrate`` when wa's own skills directory is
    empty. Cheap call — only counts entries, no file reads.
    """
    if not LEGACY_CLAUDE_SKILLS_DIR.is_dir():
        return False
    return any(
        e.is_dir() and not e.name.startswith(("_", ".")) for e in LEGACY_CLAUDE_SKILLS_DIR.iterdir()
    )


def wa_skills_dir_empty() -> bool:
    """True when wa's own skills directory is missing or has no skills."""
    if not WA_SKILLS_DIR.is_dir():
        return True
    return not any(
        e.is_dir() and not e.name.startswith(("_", ".")) for e in WA_SKILLS_DIR.iterdir()
    )


def migrate_from_claude(*, dry_run: bool = False) -> dict:
    """Copy user-level skills + plugins from ``~/.claude/`` into wa's own
    config tree. Returns a summary dict for the caller to display.

    Behaviour:
      - Each existing skill directory in ``~/.claude/skills/<name>/`` is
        copied to ``~/.weather-agents/skills/<name>/`` (full tree copy
        so reference files like ``editing.md`` come along).
      - Each plugin tree under ``~/.claude/plugins/marketplaces/`` is
        copied to the matching path under ``~/.weather-agents/``.
      - Existing destination dirs are SKIPPED (not overwritten) so a
        repeated migrate is idempotent and never clobbers customised
        local copies.
      - ``dry_run=True`` returns the plan without touching the disk.

    Summary dict keys: ``skills_copied``, ``skills_skipped``,
    ``plugins_copied``, ``plugins_skipped``, ``dry_run``.
    """
    summary: dict = {
        "skills_copied": [],
        "skills_skipped": [],
        "plugins_copied": [],
        "plugins_skipped": [],
        "dry_run": dry_run,
    }

    # User-level skills.
    if LEGACY_CLAUDE_SKILLS_DIR.is_dir():
        for entry in sorted(LEGACY_CLAUDE_SKILLS_DIR.iterdir()):
            if not entry.is_dir() or entry.name.startswith(("_", ".")):
                continue
            dst = WA_SKILLS_DIR / entry.name
            if dst.exists():
                summary["skills_skipped"].append(entry.name)
                continue
            summary["skills_copied"].append(entry.name)
            if not dry_run:
                WA_SKILLS_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copytree(entry, dst, dirs_exist_ok=False)

    # Plugin trees. The structure is deeper, so we copy whole plugin
    # dirs (not individual skill entries) to preserve everything the
    # plugin author shipped (skills, commands, agents, etc.).
    if LEGACY_CLAUDE_PLUGINS_DIR.is_dir():
        for marketplace in sorted(LEGACY_CLAUDE_PLUGINS_DIR.iterdir()):
            if not marketplace.is_dir() or marketplace.name.startswith("."):
                continue
            for root_name in ("plugins", "external_plugins"):
                src_root = marketplace / root_name
                if not src_root.is_dir():
                    continue
                for plugin_dir in sorted(src_root.iterdir()):
                    if not plugin_dir.is_dir() or plugin_dir.name.startswith("."):
                        continue
                    dst = WA_PLUGINS_DIR / marketplace.name / root_name / plugin_dir.name
                    plugin_id = f"{marketplace.name}/{root_name}/{plugin_dir.name}"
                    if dst.exists():
                        summary["plugins_skipped"].append(plugin_id)
                        continue
                    summary["plugins_copied"].append(plugin_id)
                    if not dry_run:
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copytree(plugin_dir, dst, dirs_exist_ok=False)

    return summary

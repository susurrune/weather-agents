"""Skill loader — discovers skills from .py modules and .md files (Anthropic format)."""

from __future__ import annotations

import os
from pathlib import Path

from weather_agents.core.skill import Skill, SkillRegistry


def register_all_skills(registry: SkillRegistry | None = None) -> None:
    """Discover and register all built-in skills.

    Registration order (later sources are deduplicated against earlier
    ones by name):
    1. Python-coded skills bundled with wa
    2. Markdown skills bundled with wa (wa/config/skills/)
    3. User-level Claude Code skills at ~/.claude/skills/<name>/SKILL.md
       — registered with their bare name
    4. Plugin-bundled Claude Code skills at ~/.claude/plugins/marketplaces/
       <m>/{plugins,external_plugins}/<plugin>/skills/<name>/SKILL.md —
       registered with namespaced name "<plugin>:<name>" so they coexist
       with user-level skills of the same bare name (matches Claude Code's
       UI which shows both ``pptx`` and ``anthropic-skills:pptx`` when
       both exist).

    When *registry* is None a new SkillRegistry is created. Kept for
    backward compatibility; factory.py always passes the per-agent registry.
    """
    reg = registry or SkillRegistry()
    for skill in _get_python_skills():
        reg.register(skill)
    for skill in _get_markdown_skills(reg):
        if skill.name not in reg.list_names():
            reg.register(skill)
    for skill in _get_claude_skills():
        if skill.name not in reg.list_names():
            reg.register(skill)
    # Plugins last: they get a namespace prefix so same-name skills don't
    # collide with the user-level ones. Both versions remain available.
    for skill in _get_plugin_skills():
        if skill.name not in reg.list_names():
            reg.register(skill)


def _get_python_skills() -> list[Skill]:
    """Import each Python skill module and collect Skill instances."""
    from weather_agents.skills.api_integrator import create_skill as _api_integrator
    from weather_agents.skills.arch_designer import create_skill as _arch_designer
    from weather_agents.skills.ci_cd_manager import create_skill as _ci_cd_manager
    from weather_agents.skills.code_analysis import create_skill as _code_analysis
    from weather_agents.skills.code_generator import create_skill as _code_generator
    from weather_agents.skills.code_reviewer import create_skill as _code_reviewer
    from weather_agents.skills.content_writer import create_skill as _content_writer
    from weather_agents.skills.data_transformer import create_skill as _data_transformer
    from weather_agents.skills.document_analysis import create_skill as _document_analysis
    from weather_agents.skills.emotional_companion import create_skill as _emotional_companion
    from weather_agents.skills.performance_checker import create_skill as _performance_checker
    from weather_agents.skills.security_auditor import create_skill as _security_auditor
    from weather_agents.skills.self_evolve import create_skill as _self_evolve
    from weather_agents.skills.sys_operator import create_skill as _sys_operator
    from weather_agents.skills.task_planner import create_skill as _task_planner
    from weather_agents.skills.web_research import create_skill as _web_research
    from weather_agents.skills.workflow_designer import create_skill as _workflow_designer

    return [
        _web_research(),
        _code_analysis(),
        _document_analysis(),
        _code_generator(),
        _content_writer(),
        _data_transformer(),
        _code_reviewer(),
        _security_auditor(),
        _performance_checker(),
        _task_planner(),
        _arch_designer(),
        _workflow_designer(),
        _self_evolve(),
        _sys_operator(),
        _ci_cd_manager(),
        _api_integrator(),
        _emotional_companion(),
    ]


def _get_markdown_skills(registry: SkillRegistry) -> list[Skill]:
    """Load skills from .md files in the skills config directory.

    These complement the Python-defined skills and follow the
    Anthropic/Claude Code skill format (YAML frontmatter + markdown body).
    """
    import importlib.resources

    try:
        ref = importlib.resources.files("weather_agents") / "config" / "skills"
        path = Path(str(ref))
        if path.is_dir():
            return registry.load_skills_from_directory(path)
    except Exception:
        pass
    return []


def _get_claude_skills() -> list[Skill]:
    """Load skills from Claude Code's user-level skill directory.

    Scans ~/.claude/skills/<name>/SKILL.md and parses each via the
    standard YAML-frontmatter format. Names are kept as-is (no prefix).
    Skills whose names are already registered (built-in Python skills)
    are skipped by the caller.
    """
    base_path = Path(os.path.expanduser("~/.claude/skills"))
    if not base_path.is_dir():
        return []

    skills: list[Skill] = []
    for entry in sorted(base_path.iterdir()):
        if not entry.is_dir() or entry.name.startswith("_") or entry.name.startswith("."):
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


def _get_plugin_skills() -> list[Skill]:
    """Load skills bundled inside Claude Code plugins.

    Layout (mirroring Claude Code's marketplace structure):
      ~/.claude/plugins/marketplaces/<marketplace>/plugins/<plugin>/skills/<skill>/SKILL.md
      ~/.claude/plugins/marketplaces/<marketplace>/external_plugins/<plugin>/skills/<skill>/SKILL.md

    Each loaded skill is namespaced as ``<plugin>:<skill_name>`` so it
    coexists with any user-level skill of the same bare name. This
    matches Claude Code's display where ``anthropic-skills:pptx`` and
    a separate user-installed ``pptx`` can both be available.
    """
    base = Path(os.path.expanduser("~/.claude/plugins/marketplaces"))
    if not base.is_dir():
        return []

    skills: list[Skill] = []
    for marketplace in sorted(base.iterdir()):
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
                    if (
                        not skill_entry.is_dir()
                        or skill_entry.name.startswith("_")
                        or skill_entry.name.startswith(".")
                    ):
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
                    # Namespace by plugin name. e.g. plugin="anthropic-skills"
                    # + skill name="pptx" -> "anthropic-skills:pptx"
                    bare_name = skill.name
                    skill.name = f"{plugin_dir.name}:{bare_name}"
                    skill.resource_dir = str(skill_entry)
                    skills.append(skill)
    return skills

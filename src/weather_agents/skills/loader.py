"""Skill loader — discovers skills from .py modules and .md files (Anthropic format)."""

from __future__ import annotations

import os
from pathlib import Path

from weather_agents.core.skill import Skill, SkillRegistry


def register_all_skills(registry: SkillRegistry | None = None) -> None:
    """Discover and register all built-in skills.

    When *registry* is None a new SkillRegistry is created. Kept for backward
    compatibility; factory.py always passes the per-agent registry.
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
    """Load skills from Claude Code's skill directory.

    Scans ~/.claude/skills/ for SKILL.md files and parses them using the
    standard YAML-frontmatter format. Skills with names already registered
    (e.g. built-in Python skills) are skipped by the caller.
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
                skills.append(skill)
        except Exception:
            continue
    return skills

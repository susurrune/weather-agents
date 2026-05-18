"""Skill system — Anthropic-compatible composable capability modules.

Skills use Markdown + YAML frontmatter format, matching the Claude Code
skill specification. When activated, a skill injects its system prompt
and can register custom handler tools into the agent.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Skill:
    """A composable capability module for an agent.

    Compatible with Anthropic/Claude Code skill format:
    - Markdown files with YAML frontmatter
    - name, description, tools (required_tools)
    - Optional handler for custom tool injection
    - Optional config overrides: model, temperature, max_tokens

    Attributes:
        name: Unique identifier (e.g. "code_reviewer").
        description: Human-readable summary of what the skill does.
        system_prompt: System prompt text injected when the skill is active.
        required_tools: Tool names this skill needs available.
        tools: Additional tools this skill registers via its handler.
        handler: Optional callable that receives (agent, tool_registry) and
                 registers custom tools when the skill is activated.
        model: Optional model override when this skill is active.
        temperature: Optional temperature override (0.0-2.0).
        max_tokens: Optional max output tokens override.
    """

    name: str
    description: str
    system_prompt: str = ""
    required_tools: list[str] = field(default_factory=list)
    tools: list[Any] = field(default_factory=list)
    handler: Callable[..., Any] | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    # Keyword/phrase substrings — when any appears in the user's message
    # (case-insensitive substring match) the agent auto-activates this
    # skill on entry to chat_stream. Cuts the LLM round-trip cost of
    # ``list_skills + use_skill`` for common patterns. Empty list = no
    # auto-trigger (skill is only activatable explicitly).
    triggers: list[str] = field(default_factory=list)

    @classmethod
    def from_markdown(cls, path: Path) -> Skill | None:
        """Load a skill from a Markdown file with YAML frontmatter.

        Expected format (matching Anthropic/Claude Code skill spec):
        ```markdown
        ---
        name: my_skill
        description: What this skill does
        tools:
          - required_tool_a
          - required_tool_b
        model: claude-opus-4-7  # optional override
        temperature: 0.3        # optional override
        max_tokens: 32000       # optional override
        ---

        ## Skill: My Skill
        ...system prompt body...
        ```
        """
        text = path.read_text(encoding="utf-8")
        fm, body = _parse_frontmatter(text)
        if not fm:
            return None

        name = fm.get("name", path.stem)
        description = fm.get("description", "")
        tools = fm.get("tools", [])
        required_tools = [t for t in tools if isinstance(t, str)] if isinstance(tools, list) else []

        # Config overrides
        model = fm.get("model")
        temperature = fm.get("temperature")
        max_tokens = fm.get("max_tokens")

        triggers_raw = fm.get("triggers", [])
        triggers = (
            [t for t in triggers_raw if isinstance(t, str)]
            if isinstance(triggers_raw, list)
            else []
        )

        return cls(
            name=name,
            description=description,
            system_prompt=body.strip(),
            required_tools=required_tools,
            model=model if isinstance(model, str) else None,
            temperature=temperature if isinstance(temperature, (int, float)) else None,
            max_tokens=max_tokens if isinstance(max_tokens, int) else None,
            triggers=triggers,
        )


def _parse_frontmatter(text: str) -> tuple[dict | None, str]:
    """Parse YAML frontmatter from markdown text.

    Returns (frontmatter_dict_or_None, body_text).
    """
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)", text, re.DOTALL)
    if not match:
        return None, text
    try:
        fm = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return None, text
    return fm, match.group(2)


class SkillRegistry:
    """Central registry for all available skills."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def get_skills(self, names: list[str] | None = None) -> list[Skill]:
        if names is None:
            return list(self._skills.values())
        return [self._skills[n] for n in names if n in self._skills]

    def list_names(self) -> list[str]:
        return list(self._skills.keys())

    def merge(self, other: SkillRegistry) -> None:
        self._skills.update(other._skills)

    def load_skills_from_directory(self, directory: str | Path) -> list[Skill]:
        """Load all .md skill files from a directory (Anthropic format).

        Skips files starting with _ or . (private/disabled skills).
        """
        loaded: list[Skill] = []
        dir_path = Path(directory).expanduser()
        if not dir_path.is_dir():
            return loaded

        for md_file in sorted(dir_path.glob("*.md")):
            if md_file.name.startswith(("_", ".")):
                continue
            skill = Skill.from_markdown(md_file)
            if skill:
                self.register(skill)
                loaded.append(skill)

        return loaded


# Global skill registry
global_skill_registry = SkillRegistry()

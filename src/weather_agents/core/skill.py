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
    # Directory path this skill was loaded from (set by loader). Used by the
    # agent to tell the LLM where to find the skill's templates and assets.
    resource_dir: str | None = None
    # Anthropic skill-spec fields (matches ~/.claude/skills/<name>/SKILL.md).
    # license: display-only attribution / restriction notice.
    license: str | None = None
    # allowed_tools: when set, restricts the agent's active tool set to
    # these names while this skill is active. None = no restriction
    # (skill works alongside all other tools). Used by the tool router
    # to honor Claude Code skill-spec security boundaries.
    allowed_tools: list[str] | None = None

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
        # Auto-derive triggers from the description when the YAML
        # frontmatter doesn't specify any. Anthropic's skill descriptions
        # follow a stable pattern — quoted tokens like `"deck,"` and
        # `"slides,"` plus file extensions like `.pptx` / `.pdf` ARE the
        # triggers; not extracting them meant pptx / pdf / etc. never
        # auto-activated, costing a full list_skills + use_skill round
        # trip on every "make a deck" request. Manual `triggers:` always
        # wins so authored intent isn't overridden.
        if not triggers and isinstance(description, str) and description:
            triggers = _derive_triggers_from_description(description)

        # Anthropic skill-spec extras: license (display-only) and
        # allowed-tools (restriction list). YAML uses kebab-case for the
        # latter; accept both kebab and snake forms.
        license_raw = fm.get("license")
        license = (
            license_raw.strip() if isinstance(license_raw, str) and license_raw.strip() else None
        )
        allowed_raw = fm.get("allowed-tools", fm.get("allowed_tools"))
        allowed_tools: list[str] | None = None
        if isinstance(allowed_raw, list):
            allowed_tools = [t for t in allowed_raw if isinstance(t, str)] or None
        elif isinstance(allowed_raw, str) and allowed_raw.strip():
            # Some YAMLs use comma-separated strings; tolerate both.
            allowed_tools = [t.strip() for t in allowed_raw.split(",") if t.strip()] or None

        return cls(
            name=name,
            description=description,
            system_prompt=body.strip(),
            required_tools=required_tools,
            model=model if isinstance(model, str) else None,
            temperature=temperature if isinstance(temperature, (int, float)) else None,
            max_tokens=max_tokens if isinstance(max_tokens, int) else None,
            triggers=triggers,
            license=license,
            allowed_tools=allowed_tools,
        )


# Patterns for auto-deriving triggers from skill descriptions. Each
# pattern targets a stable shape Anthropic / community skill authors
# use to embed keyword hints in prose.
_TRIGGER_QUOTED = re.compile(r"[\"'“‘]([^\"'”’\n]{1,40})[\"'”’]")
# File extension references like .pptx / .pdf / .ipynb. The leading dot
# is preserved so a substring match against "file.pdf" still hits.
_TRIGGER_EXT = re.compile(r"(?<![A-Za-z0-9])\.[A-Za-z0-9]{2,6}\b")
# Punctuation/whitespace to strip from extracted candidates so
# `"deck,"` extracts to `deck` (trailing comma removed).
_TRIGGER_STRIP = " \t,.;:!?，。、；：！？、。"


def _derive_triggers_from_description(description: str) -> list[str]:
    """Pull candidate trigger phrases out of a skill description.

    The rules are conservative — we only emit substrings that appeared
    *quoted* in the description (the author's explicit hint) or look
    like file-extension tokens. Free-prose words are NOT extracted: the
    auto-activator already uses case-insensitive substring matching, and
    matching against common short prose words ('use', 'create', 'when')
    would over-trigger and activate every skill on every message.

    Deduplicated, case-preserving, max 12 entries (enough for the
    typical skill that lists 5-8 keywords + 1-2 extensions).
    """
    raw: list[str] = []
    for m in _TRIGGER_QUOTED.finditer(description):
        raw.append(m.group(1))
    for m in _TRIGGER_EXT.finditer(description):
        raw.append(m.group(0))

    seen: set[str] = set()
    out: list[str] = []
    for token in raw:
        # Only strip TRAILING punctuation. Leading dots are meaningful:
        # `.pptx` is a file-extension trigger that must keep the dot,
        # otherwise a user message that wrote ".pptx" (with the dot)
        # wouldn't substring-match against a stripped "pptx" candidate.
        cleaned = token.rstrip(_TRIGGER_STRIP).lstrip(" \t")
        if not cleaned or len(cleaned) < 2:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
        if len(out) >= 12:
            break
    return out


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
                skill.resource_dir = str(dir_path)
                self.register(skill)
                loaded.append(skill)

        return loaded


# Global skill registry
global_skill_registry = SkillRegistry()

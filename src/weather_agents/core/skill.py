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
    # Absolute path to the source SKILL.md (set by the loader). The agent
    # tells the LLM about this path when the body was truncated, so the
    # model knows where to `read_file` for the full guide.
    source_path: str | None = None
    # True when the loaded body exceeded the inline budget and only a
    # head-only summary was placed in system_prompt. _rebuild_system_prompt
    # injects a "read this file for the full guide" hint on activation.
    # 17-30KB skills like beautiful-webpage / canvas-design used to
    # blow up first-token latency by 30-60s; the lazy mode keeps the
    # active prompt under ~2KB per skill while preserving the full
    # guide on disk where the LLM can fetch it on demand.
    body_truncated: bool = False
    # Frontmatter keys that aren't part of wa's first-class schema but
    # appear in real-world Claude Code / community skill files (``version``,
    # ``homepage``, ``slug``, ``changelog``, ``metadata``, ``compatibility``,
    # ``argument-hint``, ``user-invocable``, etc.). Preserved verbatim so
    # the data survives a round-trip through the loader; nothing in wa
    # depends on the contents, but ``/skill info`` and future tools can
    # surface them.
    metadata: dict[str, Any] = field(default_factory=dict)

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
        # Normalize Claude Code tool names (PascalCase + optional
        # ``Tool(scope pattern)``) into wa's snake_case equivalents.
        # Real-world Anthropic skills look like:
        #   allowed-tools:
        #     - Bash(ls *)
        #     - Read
        #     - Write
        # Without this mapping wa would restrict the agent to literal
        # ``Read`` / ``Write`` which don't exist in our registry, leaving
        # the agent with zero usable tools. Unknown names pass through
        # so non-Claude skills with custom tool names still work.
        if allowed_tools:
            allowed_tools = [_normalize_claude_tool_name(t) for t in allowed_tools]
            # Dedupe preserving order — Bash(ls *) + Bash(rm *) both
            # map to run_bash; keeping one is correct since wa has no
            # per-command Bash scoping.
            seen: set[str] = set()
            deduped: list[str] = []
            for t in allowed_tools:
                if t not in seen:
                    seen.add(t)
                    deduped.append(t)
            allowed_tools = deduped

        # Preserve metadata fields the loader doesn't natively support
        # (``version``, ``homepage``, ``slug``, ``changelog``, ``metadata``,
        # ``compatibility``, ``argument-hint``, ``user-invocable``, etc.)
        # so they're inspectable later without dropping authored data on
        # the floor. Claude Code skills + community plugins use these
        # routinely and wa silently ignoring them used to make /skill
        # info return a stub.
        known_keys = {
            "name",
            "description",
            "tools",
            "model",
            "temperature",
            "max_tokens",
            "triggers",
            "license",
            "allowed-tools",
            "allowed_tools",
        }
        extra_metadata = {
            k: v for k, v in fm.items() if k not in known_keys and not k.startswith("_")
        }

        # Lazy-load very large skill bodies. Anthropic's skill spec puts
        # detailed instructions IN SKILL.md (beautiful-webpage at 17KB,
        # canvas-design at 12KB, xlsx at 11KB). Injecting the full body
        # into the system prompt costs 4-5K tokens per skill per LLM call
        # — first-token latency on a cold prefix cache can hit 30-90s on
        # DeepSeek for the larger ones. The lite mode injects only the
        # head (title + first H2 section, capped at LITE_MAX_CHARS) and
        # tells the agent where to read the full file when it needs the
        # rest. Small skills (≤ LITE_THRESHOLD) keep the legacy behavior.
        body_stripped = body.strip()
        body_truncated = False
        if len(body_stripped) > _SKILL_BODY_LITE_THRESHOLD:
            body_stripped = _extract_skill_head(body_stripped, _SKILL_BODY_LITE_MAX_CHARS)
            body_truncated = True

        return cls(
            name=name,
            description=description,
            system_prompt=body_stripped,
            required_tools=required_tools,
            model=model if isinstance(model, str) else None,
            temperature=temperature if isinstance(temperature, (int, float)) else None,
            max_tokens=max_tokens if isinstance(max_tokens, int) else None,
            triggers=triggers,
            license=license,
            allowed_tools=allowed_tools,
            source_path=str(path),
            body_truncated=body_truncated,
            metadata=extra_metadata,
        )


# Tuning for lazy skill loading. Bodies up to LITE_THRESHOLD chars get
# inlined as-is (small skills lose nothing). Larger bodies are
# summarized down to LITE_MAX_CHARS — empirically 1500 chars covers the
# title + first H2 section, which is where SKILL.md authors put the
# "quick reference" / "core principles" content.
_SKILL_BODY_LITE_THRESHOLD = 2000
_SKILL_BODY_LITE_MAX_CHARS = 1500


# Claude Code → wa tool-name aliases. Real Anthropic skill files write
# ``allowed-tools`` with their built-in PascalCase tool names (``Read``,
# ``Write``, ``Bash``, etc.), often with a permission-scoping suffix
# like ``Bash(ls *)``. wa's registry uses snake_case names. The aliases
# below map every Claude name we've seen in the wild to its wa
# equivalent; names not in the map pass through untouched so custom /
# plugin-defined tools keep working. Lowercase aliases are also accepted
# (some authors write ``read`` instead of ``Read``).
_CLAUDE_TOOL_ALIASES: dict[str, str] = {
    # File I/O
    "read": "read_file",
    "write": "write_file",
    "edit": "edit_file",
    "multiedit": "edit_file",
    "delete": "delete_file",
    # Shell / execution
    "bash": "run_bash",
    "shell": "run_bash",
    # Search
    "grep": "grep",
    "glob": "file_search",
    "search": "code_search",
    "websearch": "web_search",
    "webfetch": "fetch_page",
    # Filesystem listing
    "ls": "list_directory",
    "tree": "tree",
    # HTTP
    "fetch": "http_get",
    # Misc
    "taskdone": "task_done",
}


def _normalize_claude_tool_name(raw: str) -> str:
    """Translate a Claude Code tool name into wa's registry name.

    Handles:
      - PascalCase identifiers (``Read`` → ``read_file``)
      - Permission scoping (``Bash(ls *)`` → ``Bash`` → ``run_bash``)
      - Snake_case already-wa names (``read_file`` → ``read_file``)
      - Unknown names (pass through untouched so custom plugins work)
    """
    s = raw.strip()
    if not s:
        return s
    # Strip permission scoping: ``Bash(ls *)`` → ``Bash``. wa has no
    # per-command shell scoping so the inside is informational only.
    paren = s.find("(")
    if paren > 0:
        s = s[:paren].strip()
    # Try exact match first (preserves snake_case names users / plugins
    # may already have written), then case-insensitive alias lookup.
    if s in _CLAUDE_TOOL_ALIASES.values():
        return s
    return _CLAUDE_TOOL_ALIASES.get(s.lower(), s)


def _extract_skill_head(body: str, max_chars: int) -> str:
    """Return the head of a SKILL.md body — title plus its first major
    section, truncated at ``max_chars``. The rule:

      1. Keep all lines until the SECOND H2 (``## ``) heading.
      2. Stop early if the accumulated char count crosses ``max_chars``.
      3. Always preserve the very first heading even if max_chars is small.

    The shape of Anthropic / community SKILL.md files is consistent: H1
    title → H2 'Quick Reference' / 'Core' / 'Overview' → further H2
    sections with details. Keeping just through the first H2 section
    captures the highest-signal summary while dropping the long-form
    examples that an LLM can fetch on demand via read_file.
    """
    out: list[str] = []
    char_count = 0
    h2_count = 0
    for line in body.splitlines():
        is_h2 = line.startswith("## ") and not line.startswith("### ")
        if is_h2:
            h2_count += 1
            if h2_count > 1:
                break
        # Always keep the first heading; otherwise enforce the char cap.
        if out and char_count + len(line) + 1 > max_chars:
            break
        out.append(line)
        char_count += len(line) + 1
    return "\n".join(out).rstrip()


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

"""Regression tests for Anthropic skill-spec fields (license, allowed-tools).

Kept in a separate file from test_skill.py because the YAML frontmatter
strings here use real newlines and tools-list indentation that the
existing test fixture builders weren't designed for.
"""

from __future__ import annotations


class TestClaudeToolNameNormalization:
    """Real Anthropic skill files write ``allowed-tools`` with the
    PascalCase Claude Code names (``Read``, ``Write``, ``Bash``), often
    with permission-scoping suffixes like ``Bash(ls *)``. sky's registry
    uses snake_case names. The loader must translate so a restricted-
    tool skill installed straight from Claude Code actually works."""

    def test_pascalcase_names_mapped_to_snake_case(self, tmp_path):
        from weather_agents.core.skill import Skill

        md = tmp_path / "SKILL.md"
        md.write_text(
            "---\n"
            "name: claude_style\n"
            "description: x\n"
            "allowed-tools:\n"
            "  - Read\n"
            "  - Write\n"
            "  - Edit\n"
            "  - Bash\n"
            "---\n\nBody.",
            encoding="utf-8",
        )
        s = Skill.from_markdown(md)
        assert s is not None
        # The mapped sky registry names — what _active_tool_names will
        # filter against. Without the mapping the agent would see
        # ``[Read, Write, Edit, Bash]`` and have no usable tools.
        assert s.allowed_tools == ["read_file", "write_file", "edit_file", "run_bash"]

    def test_bash_with_scope_strips_to_run_bash(self, tmp_path):
        """``Bash(ls *)`` strips the scope (sky has no per-command Bash
        permissioning) and maps to ``run_bash``. Multiple ``Bash(...)``
        entries dedupe to a single ``run_bash`` so sky doesn't see a
        redundant restriction list."""
        from weather_agents.core.skill import Skill

        md = tmp_path / "SKILL.md"
        md.write_text(
            "---\n"
            "name: scoped\n"
            "description: x\n"
            "allowed-tools:\n"
            "  - Bash(ls *)\n"
            "  - Bash(mkdir *)\n"
            "  - Read\n"
            "---\n\nBody.",
            encoding="utf-8",
        )
        s = Skill.from_markdown(md)
        assert s is not None
        assert s.allowed_tools == ["run_bash", "read_file"]

    def test_snake_case_already_wa_name_passes_through(self, tmp_path):
        """sky-native skill files use snake_case directly. Those must
        not be re-mapped (would corrupt the list)."""
        from weather_agents.core.skill import Skill

        md = tmp_path / "SKILL.md"
        md.write_text(
            "---\n"
            "name: native\n"
            "description: x\n"
            "allowed-tools:\n"
            "  - read_file\n"
            "  - write_file\n"
            "  - run_bash\n"
            "---\n\nBody.",
            encoding="utf-8",
        )
        s = Skill.from_markdown(md)
        assert s is not None
        assert s.allowed_tools == ["read_file", "write_file", "run_bash"]

    def test_unknown_tool_name_passes_through(self, tmp_path):
        """A plugin can ship its own custom tool name. The loader must
        NOT mangle it just because we don't have an alias — it has to
        survive verbatim so the plugin's handler can register a tool
        of the same name."""
        from weather_agents.core.skill import Skill

        md = tmp_path / "SKILL.md"
        md.write_text(
            "---\n"
            "name: custom\n"
            "description: x\n"
            "allowed-tools:\n"
            "  - my_custom_tool\n"
            "  - AnotherTool\n"
            "---\n\nBody.",
            encoding="utf-8",
        )
        s = Skill.from_markdown(md)
        assert s is not None
        assert "my_custom_tool" in s.allowed_tools
        assert "AnotherTool" in s.allowed_tools

    def test_normalizer_helper_directly(self):
        from weather_agents.core.skill import _normalize_claude_tool_name

        assert _normalize_claude_tool_name("Read") == "read_file"
        assert _normalize_claude_tool_name("Bash(ls *)") == "run_bash"
        assert _normalize_claude_tool_name("read_file") == "read_file"
        assert _normalize_claude_tool_name("CustomThing") == "CustomThing"
        assert _normalize_claude_tool_name("") == ""


class TestMetadataPreserved:
    """Real-world Claude Code skills include version / homepage / slug /
    metadata / compatibility / changelog blocks in their frontmatter.
    sky used to silently drop these — round 8 keeps them in skill.metadata
    so /skill info and future tooling can display them, and so the data
    survives a round-trip load."""

    def test_known_metadata_keys_captured(self, tmp_path):
        from weather_agents.core.skill import Skill

        md = tmp_path / "SKILL.md"
        md.write_text(
            "---\n"
            "name: rich\n"
            "description: x\n"
            "version: 1.2.3\n"
            "homepage: https://example.com\n"
            "license: MIT\n"
            "compatibility: Node 18+\n"
            "metadata:\n"
            "  author: someone\n"
            "  category: ai/image\n"
            "---\n\nBody.",
            encoding="utf-8",
        )
        s = Skill.from_markdown(md)
        assert s is not None
        # First-class field still works.
        assert s.license == "MIT"
        # Unknown-to-sky fields preserved.
        assert s.metadata["version"] == "1.2.3"
        assert s.metadata["homepage"] == "https://example.com"
        assert s.metadata["compatibility"] == "Node 18+"
        assert s.metadata["metadata"]["author"] == "someone"

    def test_first_class_fields_not_duplicated_into_metadata(self, tmp_path):
        """``name`` / ``description`` / ``triggers`` / ``allowed-tools``
        belong to the dedicated Skill attributes; they must NOT also
        show up in ``metadata`` (would be a confusing double source of
        truth)."""
        from weather_agents.core.skill import Skill

        md = tmp_path / "SKILL.md"
        md.write_text(
            "---\nname: a\ndescription: b\ntriggers:\n  - t1\n---\n\nBody.",
            encoding="utf-8",
        )
        s = Skill.from_markdown(md)
        assert s is not None
        for key in ("name", "description", "triggers", "allowed-tools", "allowed_tools"):
            assert key not in s.metadata

    def test_no_extras_means_empty_metadata(self, tmp_path):
        from weather_agents.core.skill import Skill

        md = tmp_path / "SKILL.md"
        md.write_text(
            "---\nname: minimal\ndescription: x\n---\n\nBody.",
            encoding="utf-8",
        )
        s = Skill.from_markdown(md)
        assert s is not None
        assert s.metadata == {}


class TestAnthropicSpec:
    """Skill must parse Anthropic skill-spec fields: license and
    allowed-tools (kebab-case) per Claude Code's SKILL.md format."""

    def test_license_field_parsed(self, tmp_path):
        from weather_agents.core.skill import Skill

        md = tmp_path / "SKILL.md"
        md.write_text(
            "---\n"
            "name: licensed_skill\n"
            "description: A test\n"
            "license: Proprietary. LICENSE.txt has complete terms\n"
            "---\n"
            "\n"
            "Body.",
            encoding="utf-8",
        )
        s = Skill.from_markdown(md)
        assert s is not None
        assert s.license == "Proprietary. LICENSE.txt has complete terms"

    def test_allowed_tools_kebab_case(self, tmp_path):
        from weather_agents.core.skill import Skill

        md = tmp_path / "SKILL.md"
        md.write_text(
            "---\n"
            "name: restricted\n"
            "description: x\n"
            "allowed-tools:\n"
            "  - read_file\n"
            "  - write_file\n"
            "---\n"
            "\n"
            "Body.",
            encoding="utf-8",
        )
        s = Skill.from_markdown(md)
        assert s is not None
        assert s.allowed_tools == ["read_file", "write_file"]

    def test_allowed_tools_snake_case_accepted(self, tmp_path):
        from weather_agents.core.skill import Skill

        md = tmp_path / "SKILL.md"
        md.write_text(
            "---\nname: snake\ndescription: x\nallowed_tools:\n  - tool_a\n---\n\nBody.",
            encoding="utf-8",
        )
        s = Skill.from_markdown(md)
        assert s is not None
        assert s.allowed_tools == ["tool_a"]

    def test_no_allowed_tools_means_unrestricted(self, tmp_path):
        from weather_agents.core.skill import Skill

        md = tmp_path / "SKILL.md"
        md.write_text(
            "---\nname: open\ndescription: x\n---\n\nBody.",
            encoding="utf-8",
        )
        s = Skill.from_markdown(md)
        assert s is not None
        assert s.allowed_tools is None


class TestAllowedToolsRestriction:
    """BaseAgent._active_tool_names must filter by union-of-allowed_tools
    of active skills, while keeping wildcard semantics: any active skill
    without allowed_tools lifts the restriction."""

    def test_restriction_filters_to_allowed_set(self, app_config, bus, mock_llm):
        from weather_agents.agents.fog import FogAgent
        from weather_agents.core.skill import Skill, SkillRegistry
        from weather_agents.core.tool import Tool, ToolRegistry

        reg = ToolRegistry()
        for n in ("read_file", "write_file", "shell_exec", "web_search"):
            reg.register(Tool(name=n, description=n))
        sreg = SkillRegistry()
        sreg.register(
            Skill(
                name="restricted",
                description="x",
                allowed_tools=["read_file", "shell_exec"],
            )
        )
        agent = FogAgent(
            config=app_config,
            llm=mock_llm,
            bus=bus,
            tool_registry=reg,
            skill_registry=sreg,
        )
        agent._skills = sreg.get_skills()
        agent._active_skills.add("restricted")
        active = agent._active_tool_names()
        assert set(active) == {"read_file", "shell_exec"}

    def test_wildcard_skill_lifts_restriction(self, app_config, bus, mock_llm):
        from weather_agents.agents.fog import FogAgent
        from weather_agents.core.skill import Skill, SkillRegistry
        from weather_agents.core.tool import Tool, ToolRegistry

        reg = ToolRegistry()
        for n in ("read_file", "write_file", "shell_exec"):
            reg.register(Tool(name=n, description=n))
        sreg = SkillRegistry()
        sreg.register(
            Skill(
                name="restricted",
                description="x",
                allowed_tools=["read_file"],
            )
        )
        sreg.register(Skill(name="wildcard", description="x", allowed_tools=None))
        agent = FogAgent(
            config=app_config,
            llm=mock_llm,
            bus=bus,
            tool_registry=reg,
            skill_registry=sreg,
        )
        agent._skills = sreg.get_skills()
        agent._active_skills.update(["restricted", "wildcard"])
        active = set(agent._active_tool_names())
        # wildcard lifts the restriction; ALL registered tools available
        assert active == {"read_file", "write_file", "shell_exec"}

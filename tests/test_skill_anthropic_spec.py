"""Regression tests for Anthropic skill-spec fields (license, allowed-tools).

Kept in a separate file from test_skill.py because the YAML frontmatter
strings here use real newlines and tools-list indentation that the
existing test fixture builders weren't designed for.
"""

from __future__ import annotations


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

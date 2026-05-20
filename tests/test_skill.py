"""Tests for skill system including config overrides."""

import pytest


class TestSkillSmoke:
    """Verify every registered skill can be created and has valid metadata."""

    SKILL_NAMES = [
        "api_integrator",
        "arch_designer",
        "ci_cd_manager",
        "code_analysis",
        "code_generator",
        "code_reviewer",
        "content_writer",
        "data_transformer",
        "document_analysis",
        "emotional_companion",
        "performance_checker",
        "security_auditor",
        "self_evolve",
        "sys_operator",
        "task_planner",
        "web_research",
        "workflow_designer",
    ]

    @pytest.mark.parametrize("skill_name", SKILL_NAMES)
    def test_skill_creates(self, skill_name):
        """Each skill module must export a valid create_skill() function."""
        mod = __import__(f"weather_agents.skills.{skill_name}", fromlist=["create_skill"])
        skill = mod.create_skill()

        from weather_agents.core.skill import Skill

        assert isinstance(skill, Skill), f"{skill_name} should return Skill"
        assert skill.name, f"{skill_name} should have a name"
        assert skill.description, f"{skill_name} should have a description"
        assert isinstance(skill.required_tools, list), f"{skill_name} required_tools must be list"
        assert skill.system_prompt, f"{skill_name} should have system_prompt"

    @pytest.mark.parametrize("skill_name", SKILL_NAMES)
    def test_skill_handler_injects_tools(self, skill_name):
        """Each skill handler (if present) must return a list of Tool."""
        from unittest.mock import MagicMock

        from weather_agents.core.tool import ToolRegistry

        mod = __import__(f"weather_agents.skills.{skill_name}", fromlist=["create_skill"])
        skill = mod.create_skill()

        if skill.handler is None:
            pytest.skip(f"{skill_name} has no handler")

        registry = ToolRegistry()
        result = skill.handler(MagicMock(), registry)
        if result is not None:
            assert isinstance(result, list), f"{skill_name} handler should return list or None"


class TestSkillConfigOverrides:
    def test_skill_supports_model_override(self):
        from weather_agents.core.skill import Skill

        skill = Skill(
            name="test_skill",
            description="A test skill",
            model="claude-opus-4-7",
        )
        assert skill.model == "claude-opus-4-7"
        assert skill.temperature is None
        assert skill.max_tokens is None

    def test_skill_supports_temperature_override(self):
        from weather_agents.core.skill import Skill

        skill = Skill(
            name="test_skill",
            description="A test skill",
            temperature=0.3,
        )
        assert skill.temperature == 0.3

    def test_skill_supports_max_tokens_override(self):
        from weather_agents.core.skill import Skill

        skill = Skill(
            name="test_skill",
            description="A test skill",
            max_tokens=32000,
        )
        assert skill.max_tokens == 32000

    def test_skill_all_overrides(self):
        from weather_agents.core.skill import Skill

        skill = Skill(
            name="test_skill",
            description="A test skill",
            model="gpt-4.1",
            temperature=0.7,
            max_tokens=16000,
        )
        assert skill.model == "gpt-4.1"
        assert skill.temperature == 0.7
        assert skill.max_tokens == 16000

    def test_skill_default_no_overrides(self):
        from weather_agents.core.skill import Skill

        skill = Skill(name="test", description="test")
        assert skill.model is None
        assert skill.temperature is None
        assert skill.max_tokens is None

    def test_skill_from_markdown_with_overrides(self, tmp_path):
        from weather_agents.core.skill import Skill

        md = tmp_path / "test.md"
        md.write_text(
            """---
name: test_skill
description: A test
model: claude-sonnet-4-6
temperature: 0.5
max_tokens: 8192
---

## Test Skill
This is a test skill.
""",
            encoding="utf-8",
        )
        skill = Skill.from_markdown(md)
        assert skill is not None
        assert skill.model == "claude-sonnet-4-6"
        assert skill.temperature == 0.5
        assert skill.max_tokens == 8192
        assert "This is a test skill" in skill.system_prompt


class TestAutoDerivedTriggers:
    """When a SKILL.md doesn't specify `triggers:` in its frontmatter,
    derive candidates from quoted tokens and file extensions in the
    description. Anthropic skill descriptions follow this shape (the
    pptx skill literally says: trigger on \"deck,\" \"slides,\"
    \"presentation,\" .pptx). Without auto-derivation those skills cost
    a full list_skills + use_skill round-trip on every relevant message."""

    def test_quoted_tokens_extracted(self, tmp_path):
        from weather_agents.core.skill import Skill

        # YAML block scalar (|-) lets us embed double-quotes literally
        # inside the description without escaping or YAML re-parsing.
        # Anthropic SKILL.md files in the wild use exactly this shape.
        md = tmp_path / "pptx.md"
        md.write_text(
            "---\n"
            "name: pptx\n"
            "description: |-\n"
            "  Use this skill any time a .pptx file is involved.\n"
            '  Trigger whenever the user mentions "deck," "slides," or "presentation."\n'
            "  If a .pptx file needs to be opened, use it.\n"
            "---\n\nBody.",
            encoding="utf-8",
        )
        s = Skill.from_markdown(md)
        assert s is not None
        triggers_lower = [t.lower() for t in s.triggers]
        assert "deck" in triggers_lower
        assert "slides" in triggers_lower
        assert "presentation" in triggers_lower
        assert ".pptx" in triggers_lower

    def test_explicit_triggers_field_wins(self, tmp_path):
        """Authored `triggers:` overrides auto-derivation — author intent
        must always win, otherwise edits to the description silently
        break auto-activation."""
        from weather_agents.core.skill import Skill

        md = tmp_path / "x.md"
        md.write_text(
            "---\n"
            "name: x\n"
            "description: |-\n"
            '  "trigger A" and "trigger B"\n'
            "triggers:\n"
            "  - exclusive_only\n"
            "---\n\nBody.",
            encoding="utf-8",
        )
        s = Skill.from_markdown(md)
        assert s is not None
        assert s.triggers == ["exclusive_only"]

    def test_no_triggers_when_description_has_no_hints(self, tmp_path):
        from weather_agents.core.skill import Skill

        md = tmp_path / "y.md"
        md.write_text(
            "---\nname: y\ndescription: A generic skill with no quoted hints.\n---\n\nBody.",
            encoding="utf-8",
        )
        s = Skill.from_markdown(md)
        assert s is not None
        assert s.triggers == []

    def test_helper_strips_trailing_punctuation(self):
        from weather_agents.core.skill import _derive_triggers_from_description

        triggers = _derive_triggers_from_description('Trigger on "deck," "slides," "presentation."')
        # Trailing commas / periods inside the quotes should not bleed
        # into the trigger — substring matching against "deck," would
        # miss a user message that wrote "make me a deck" with no comma.
        assert "deck" in triggers
        assert not any(t.endswith(",") or t.endswith(".") for t in triggers)

    def test_helper_dedupes_case_insensitively(self):
        from weather_agents.core.skill import _derive_triggers_from_description

        triggers = _derive_triggers_from_description('"Deck" "deck" "DECK"')
        # First-seen wins; subsequent case variations are dropped.
        assert len(triggers) == 1


class TestListSkillsSuppression:
    """When a trigger auto-activates a skill, the LLM has zero reason
    to call list_skills — that's a redundant ~1k tokens + round-trip.
    Verify the suppression hint enters the message stream so the model
    sees both the policy AND knows which skills were chosen."""

    def test_auto_activated_skills_returned_from_chat_stream(self, app_config, bus, mock_llm):
        # Direct unit on the trigger return: chat_stream wires this list
        # into _chat_stream_impl as auto_activated. The integration with
        # tool-set suppression has its own coverage in agent internals.
        import asyncio

        from weather_agents.agents.fog import FogAgent
        from weather_agents.core.skill import Skill, SkillRegistry
        from weather_agents.core.tool import ToolRegistry

        sreg = SkillRegistry()
        sreg.register(
            Skill(
                name="pptx",
                description="x",
                triggers=["pptx", "ppt", "presentation"],
            )
        )
        agent = FogAgent(
            config=app_config,
            llm=mock_llm,
            bus=bus,
            tool_registry=ToolRegistry(),
            skill_registry=sreg,
        )

        async def _consume():
            async for _ in agent.chat_stream("make me a ppt"):
                pass

        asyncio.run(_consume())
        # The skill should now be active AND short_term should carry
        # the suppression hint as a system message.
        assert "pptx" in agent._active_skills
        sys_msgs = [m.content for m in agent.memory.short_term if m.role == "system"]
        assert any("Auto-activated skills" in c for c in sys_msgs)
        assert any("Do NOT" in c and "list_skills" in c for c in sys_msgs)

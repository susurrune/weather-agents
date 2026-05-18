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

"""Tests for skill system including config overrides."""

import pytest


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

"""Tests for skill system including config overrides."""



class TestSkillSmoke:
    """Verify every bundled built-in SKILL.md loads cleanly.

    The 17 default skills used to be Python modules with ``create_skill()``
    factories; the unification refactor converted them to SKILL.md files
    shipped under ``weather_agents/assets/builtin_skills/<name>/SKILL.md``.
    This test walks the on-disk directory so the test never goes stale
    against the actual ship list.
    """

    # The names every release should ship — pinned here so a bundled
    # skill being silently dropped from the package gets caught even if
    # the directory exists but has fewer than expected entries.
    EXPECTED_SKILLS = {
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
    }

    @staticmethod
    def _builtin_skill_paths():
        import importlib.resources
        from pathlib import Path

        ref = importlib.resources.files("weather_agents") / "assets" / "builtin_skills"
        base = Path(str(ref))
        assert base.is_dir(), f"bundled skills dir missing: {base}"
        return sorted(d for d in base.iterdir() if d.is_dir() and not d.name.startswith(("_", ".")))

    def test_all_expected_skills_shipped(self):
        names = {d.name for d in self._builtin_skill_paths()}
        missing = self.EXPECTED_SKILLS - names
        assert not missing, f"bundled skill(s) missing: {sorted(missing)}"

    def test_each_skill_md_loads(self):
        """Every bundled skill must parse without errors and produce
        a Skill with all required fields populated."""
        from weather_agents.core.skill import Skill

        for skill_dir in self._builtin_skill_paths():
            skill_md = skill_dir / "SKILL.md"
            assert skill_md.is_file(), f"{skill_dir.name} missing SKILL.md"
            skill = Skill.from_markdown(skill_md)
            assert skill is not None, f"{skill_dir.name} failed to parse"
            assert skill.name, f"{skill_dir.name} has empty name"
            assert skill.description, f"{skill_dir.name} has empty description"
            assert skill.system_prompt, f"{skill_dir.name} has empty body"
            assert isinstance(skill.required_tools, list)

    def test_register_all_skills_loads_bundled_set(self):
        """The loader plumbing must surface the bundled skills via
        register_all_skills — without this, an installed package would
        ship with the directory present but the loader unable to find
        it (would point at a stale config layout)."""
        from weather_agents.core.skill import SkillRegistry
        from weather_agents.skills.loader import register_all_skills

        reg = SkillRegistry()
        register_all_skills(reg)
        names = set(reg.list_names())
        missing = self.EXPECTED_SKILLS - names
        assert not missing, f"loader didn't surface: {sorted(missing)}"


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


class TestLazySkillLoad:
    """Round 5: very large SKILL.md bodies blow up first-token latency
    when injected as-is on every LLM call. The loader now truncates to
    the head section (title + first H2) for bodies over the threshold,
    sets body_truncated=True, and records source_path so the agent can
    tell the LLM where to read_file for the full guide."""

    def test_small_body_loaded_in_full(self, tmp_path):
        from weather_agents.core.skill import Skill

        md = tmp_path / "small.md"
        md.write_text(
            "---\nname: small\ndescription: x\n---\n\n# Small Skill\n\nThis fits inline easily.",
            encoding="utf-8",
        )
        s = Skill.from_markdown(md)
        assert s is not None
        assert s.body_truncated is False
        assert "This fits inline easily" in s.system_prompt

    def test_large_body_truncated_to_head(self, tmp_path):
        from weather_agents.core.skill import Skill

        body = "# Big Skill\n\n## Quick Reference\nFirst section content.\n\n"
        body += "## Detailed Guide\n"
        body += "X" * 5000  # push well past LITE_THRESHOLD
        md = tmp_path / "big.md"
        md.write_text(
            f"---\nname: big\ndescription: x\n---\n\n{body}",
            encoding="utf-8",
        )
        s = Skill.from_markdown(md)
        assert s is not None
        assert s.body_truncated is True
        # Head kept: title + first H2 section.
        assert "Big Skill" in s.system_prompt
        assert "Quick Reference" in s.system_prompt
        # Detail section dropped (the LLM should read_file for it).
        assert "Detailed Guide" not in s.system_prompt
        # source_path recorded so the agent can name the file in the
        # lazy-load hint it injects on activation.
        assert s.source_path and "big.md" in s.source_path

    def test_extract_head_stops_at_second_h2(self):
        from weather_agents.core.skill import _extract_skill_head

        body = (
            "# Title\n\n"
            "## First Section\n"
            "First section body.\n\n"
            "## Second Section\n"
            "Should NOT be included.\n"
        )
        head = _extract_skill_head(body, max_chars=10000)
        assert "First section body" in head
        assert "Second Section" not in head
        assert "Should NOT" not in head

    def test_extract_head_respects_char_cap(self):
        from weather_agents.core.skill import _extract_skill_head

        body = "# Title\n\n## Big Section\n" + ("X" * 5000)
        head = _extract_skill_head(body, max_chars=200)
        # Cap is enforced even within a single section.
        assert len(head) <= 250  # small slack for the line we accept before cap
        assert "Title" in head

    def test_extract_head_no_h2_returns_capped_body(self):
        from weather_agents.core.skill import _extract_skill_head

        # Some SKILL.md files have no H2 at all — should still produce
        # a sensible head bounded by char cap.
        body = "# Title only\n\nBody text without any second-level heading."
        head = _extract_skill_head(body, max_chars=10000)
        assert "Body text" in head

    def test_rebuild_system_prompt_injects_lazy_hint(self, app_config, bus, mock_llm, tmp_path):
        """The agent's rebuild path must mention source_path so the LLM
        knows it can read_file the full guide. Without the hint the
        model sees a truncated prompt with no signal that more exists."""
        from weather_agents.agents.fog import FogAgent
        from weather_agents.core.skill import Skill, SkillRegistry
        from weather_agents.core.tool import ToolRegistry

        # Build a skill that simulates the truncated state directly so
        # we don't depend on the threshold internals.
        sk = Skill(
            name="big_skill",
            description="x",
            system_prompt="head only",
            source_path=str(tmp_path / "big.md"),
            body_truncated=True,
        )
        reg = SkillRegistry()
        reg.register(sk)

        agent = FogAgent(
            config=app_config,
            llm=mock_llm,
            bus=bus,
            tool_registry=ToolRegistry(),
            skill_registry=reg,
        )
        agent._skills = reg.get_skills()
        agent._active_skills.add("big_skill")
        agent._rebuild_system_prompt()

        sys_msgs = [m.content for m in agent.memory.short_term if m.role == "system"]
        joined = "\n".join(sys_msgs)
        # Either zh or en wording is acceptable; the marker is the path.
        assert sk.source_path is not None and sk.source_path in joined
        assert "read_file" in joined


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

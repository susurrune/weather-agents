"""Tests for the sky-owned skills directory + Claude-code → sky migration.

Round 6 split skills out of ``~/.claude/skills/`` (where Claude Code
keeps its install-shared set) into ``~/.skyloom/skills/`` so
sky owns the directory and Claude Code's lifecycle no longer affects
ours. A migration helper copies the legacy content on demand.
"""

from __future__ import annotations

from pathlib import Path


def _make_skill_md(path: Path, name: str, body: str = "Body.") -> None:
    """Create a minimal SKILL.md at <path>/SKILL.md."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: test\n---\n\n{body}",
        encoding="utf-8",
    )


class TestWaSkillsLoader:
    def test_returns_empty_when_dir_missing(self, tmp_path, monkeypatch):
        from weather_agents.skills import loader

        monkeypatch.setattr(loader, "WA_SKILLS_DIR", tmp_path / "nope")
        assert loader._get_wa_user_skills() == []

    def test_loads_skill_from_wa_dir(self, tmp_path, monkeypatch):
        from weather_agents.skills import loader

        skills_root = tmp_path / "skills"
        _make_skill_md(skills_root / "alpha", "alpha")

        monkeypatch.setattr(loader, "WA_SKILLS_DIR", skills_root)
        result = loader._get_wa_user_skills()
        assert len(result) == 1
        assert result[0].name == "alpha"
        # resource_dir must point at the skill's own subdir so the
        # agent can find related files (editing.md, examples/, etc.).
        assert result[0].resource_dir == str(skills_root / "alpha")

    def test_skips_underscore_and_dot_prefixed(self, tmp_path, monkeypatch):
        """``_disabled`` and ``.hidden`` directories must not be loaded —
        same convention as Claude Code so users can stage drafts."""
        from weather_agents.skills import loader

        skills_root = tmp_path / "skills"
        _make_skill_md(skills_root / "active", "active")
        _make_skill_md(skills_root / "_disabled", "disabled")
        _make_skill_md(skills_root / ".scratch", "scratch")

        monkeypatch.setattr(loader, "WA_SKILLS_DIR", skills_root)
        names = {s.name for s in loader._get_wa_user_skills()}
        assert names == {"active"}

    def test_plugin_skills_namespaced(self, tmp_path, monkeypatch):
        from weather_agents.skills import loader

        plugins_root = tmp_path / "plugins" / "marketplaces"
        _make_skill_md(
            plugins_root / "anthropic-skills" / "plugins" / "anthropic-skills" / "skills" / "pptx",
            "pptx",
        )

        monkeypatch.setattr(loader, "WA_PLUGINS_DIR", plugins_root)
        skills = loader._get_wa_plugin_skills()
        assert len(skills) == 1
        # Namespace prefix: <plugin>:<bare-name>. Allows a user-level
        # ``pptx`` to coexist with ``anthropic-skills:pptx``.
        assert skills[0].name == "anthropic-skills:pptx"


class TestLegacyDetection:
    def test_has_legacy_claude_skills_false_when_missing(self, tmp_path, monkeypatch):
        from weather_agents.skills import loader

        monkeypatch.setattr(loader, "LEGACY_CLAUDE_SKILLS_DIR", tmp_path / "nope")
        assert loader.has_legacy_claude_skills() is False

    def test_has_legacy_claude_skills_true_when_has_skill(self, tmp_path, monkeypatch):
        from weather_agents.skills import loader

        legacy = tmp_path / "claude_skills"
        _make_skill_md(legacy / "pptx", "pptx")
        monkeypatch.setattr(loader, "LEGACY_CLAUDE_SKILLS_DIR", legacy)
        assert loader.has_legacy_claude_skills() is True

    def test_has_legacy_skips_underscore_prefixed_only(self, tmp_path, monkeypatch):
        """A directory full of ``_disabled`` entries doesn't count as
        'has skills' for the migration hint — those would never load."""
        from weather_agents.skills import loader

        legacy = tmp_path / "claude_skills"
        _make_skill_md(legacy / "_disabled", "x")
        monkeypatch.setattr(loader, "LEGACY_CLAUDE_SKILLS_DIR", legacy)
        assert loader.has_legacy_claude_skills() is False

    def test_wa_skills_dir_empty_true_when_missing(self, tmp_path, monkeypatch):
        from weather_agents.skills import loader

        monkeypatch.setattr(loader, "WA_SKILLS_DIR", tmp_path / "nope")
        assert loader.wa_skills_dir_empty() is True

    def test_wa_skills_dir_empty_false_when_has_skill(self, tmp_path, monkeypatch):
        from weather_agents.skills import loader

        wa_root = tmp_path / "wa_skills"
        _make_skill_md(wa_root / "alpha", "alpha")
        monkeypatch.setattr(loader, "WA_SKILLS_DIR", wa_root)
        assert loader.wa_skills_dir_empty() is False


class TestMigrateFromClaude:
    def test_copies_skill_dirs(self, tmp_path, monkeypatch):
        from weather_agents.skills import loader

        legacy = tmp_path / "claude_skills"
        sky = tmp_path / "wa_skills"
        _make_skill_md(legacy / "pptx", "pptx", body="pptx body")
        _make_skill_md(legacy / "pdf", "pdf", body="pdf body")

        monkeypatch.setattr(loader, "LEGACY_CLAUDE_SKILLS_DIR", legacy)
        monkeypatch.setattr(loader, "LEGACY_CLAUDE_PLUGINS_DIR", tmp_path / "claude_plugins")
        monkeypatch.setattr(loader, "WA_SKILLS_DIR", sky)
        monkeypatch.setattr(loader, "WA_PLUGINS_DIR", tmp_path / "wa_plugins")

        summary = loader.migrate_from_claude()
        assert sorted(summary["skills_copied"]) == ["pdf", "pptx"]
        assert summary["skills_skipped"] == []
        # Files actually present in the destination, AND the body
        # content survived the copy.
        assert (sky / "pptx" / "SKILL.md").exists()
        assert "pptx body" in (sky / "pptx" / "SKILL.md").read_text(encoding="utf-8")

    def test_copies_nested_reference_files(self, tmp_path, monkeypatch):
        """SKILL.md often references sibling files (e.g. editing.md,
        pptxgenjs.md). Migration must copy the WHOLE skill folder, not
        just SKILL.md — otherwise ``read_file`` from inside the LLM
        would 404 after migration."""
        from weather_agents.skills import loader

        legacy = tmp_path / "claude_skills"
        sky = tmp_path / "wa_skills"
        _make_skill_md(legacy / "pptx", "pptx")
        (legacy / "pptx" / "editing.md").write_text("editing guide", encoding="utf-8")
        (legacy / "pptx" / "examples").mkdir()
        (legacy / "pptx" / "examples" / "demo.py").write_text("print('hi')", encoding="utf-8")

        monkeypatch.setattr(loader, "LEGACY_CLAUDE_SKILLS_DIR", legacy)
        monkeypatch.setattr(loader, "LEGACY_CLAUDE_PLUGINS_DIR", tmp_path / "claude_plugins")
        monkeypatch.setattr(loader, "WA_SKILLS_DIR", sky)
        monkeypatch.setattr(loader, "WA_PLUGINS_DIR", tmp_path / "wa_plugins")

        loader.migrate_from_claude()
        assert (sky / "pptx" / "editing.md").read_text(encoding="utf-8") == "editing guide"
        assert (sky / "pptx" / "examples" / "demo.py").exists()

    def test_idempotent_skips_existing(self, tmp_path, monkeypatch):
        from weather_agents.skills import loader

        legacy = tmp_path / "claude_skills"
        sky = tmp_path / "wa_skills"
        _make_skill_md(legacy / "pptx", "pptx", body="legacy body")
        # Pre-existing destination with DIFFERENT content — must NOT be
        # clobbered. This is the user's customised copy.
        _make_skill_md(sky / "pptx", "pptx", body="user customised body")

        monkeypatch.setattr(loader, "LEGACY_CLAUDE_SKILLS_DIR", legacy)
        monkeypatch.setattr(loader, "LEGACY_CLAUDE_PLUGINS_DIR", tmp_path / "claude_plugins")
        monkeypatch.setattr(loader, "WA_SKILLS_DIR", sky)
        monkeypatch.setattr(loader, "WA_PLUGINS_DIR", tmp_path / "wa_plugins")

        summary = loader.migrate_from_claude()
        assert summary["skills_skipped"] == ["pptx"]
        assert summary["skills_copied"] == []
        body = (sky / "pptx" / "SKILL.md").read_text(encoding="utf-8")
        assert "user customised body" in body
        assert "legacy body" not in body

    def test_dry_run_does_not_touch_disk(self, tmp_path, monkeypatch):
        from weather_agents.skills import loader

        legacy = tmp_path / "claude_skills"
        sky = tmp_path / "wa_skills"
        _make_skill_md(legacy / "pptx", "pptx")

        monkeypatch.setattr(loader, "LEGACY_CLAUDE_SKILLS_DIR", legacy)
        monkeypatch.setattr(loader, "LEGACY_CLAUDE_PLUGINS_DIR", tmp_path / "claude_plugins")
        monkeypatch.setattr(loader, "WA_SKILLS_DIR", sky)
        monkeypatch.setattr(loader, "WA_PLUGINS_DIR", tmp_path / "wa_plugins")

        summary = loader.migrate_from_claude(dry_run=True)
        assert summary["skills_copied"] == ["pptx"]
        assert summary["dry_run"] is True
        # Destination must not have been created.
        assert not (sky / "pptx").exists()

    def test_copies_plugin_trees(self, tmp_path, monkeypatch):
        from weather_agents.skills import loader

        legacy = tmp_path / "claude_plugins"
        sky = tmp_path / "wa_plugins"
        plugin_src = (
            legacy / "anthropic-skills" / "plugins" / "anthropic-skills" / "skills" / "pptx"
        )
        _make_skill_md(plugin_src, "pptx")

        monkeypatch.setattr(loader, "LEGACY_CLAUDE_SKILLS_DIR", tmp_path / "claude_skills")
        monkeypatch.setattr(loader, "LEGACY_CLAUDE_PLUGINS_DIR", legacy)
        monkeypatch.setattr(loader, "WA_SKILLS_DIR", tmp_path / "wa_skills")
        monkeypatch.setattr(loader, "WA_PLUGINS_DIR", sky)

        summary = loader.migrate_from_claude()
        assert summary["plugins_copied"] == ["anthropic-skills/plugins/anthropic-skills"]
        dst = (
            sky
            / "anthropic-skills"
            / "plugins"
            / "anthropic-skills"
            / "skills"
            / "pptx"
            / "SKILL.md"
        )
        assert dst.exists()

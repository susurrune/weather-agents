"""Tests for the local user profile + per-agent custom persona store."""

from __future__ import annotations

import pytest

from weather_agents.core import config, profile


@pytest.fixture(autouse=True)
def _tmp_config_dir(tmp_path, monkeypatch):
    """Redirect ~/.skyloom to a tmp dir so tests never touch the real one."""
    monkeypatch.setattr(config, "USER_CONFIG_DIR", tmp_path / ".skyloom")


class TestUserProfile:
    def test_roundtrip_and_format(self):
        assert profile.load_profile() == {}
        assert profile.format_profile_for_prompt("zh") == ""

        profile.set_profile_field("称呼", "阿K")
        profile.set_profile_field("喜好", "巴赫")
        assert profile.load_profile() == {"称呼": "阿K", "喜好": "巴赫"}

        block = profile.format_profile_for_prompt("zh")
        assert "关于用户" in block and "阿K" in block and "巴赫" in block
        en = profile.format_profile_for_prompt("en")
        assert "About the user" in en

    def test_clear_field_and_all(self):
        profile.set_profile_field("a", "1")
        profile.set_profile_field("b", "2")
        profile.clear_profile_field("a")
        assert profile.load_profile() == {"b": "2"}
        profile.clear_profile_field()  # wipe all
        assert profile.load_profile() == {}

    def test_empty_key_ignored(self):
        profile.set_profile_field("   ", "x")
        assert profile.load_profile() == {}

    def test_corrupt_file_is_safe(self):
        p = config.USER_CONFIG_DIR / "profile.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{ not json", encoding="utf-8")
        assert profile.load_profile() == {}  # degrades, no crash


class TestPersona:
    def test_save_load_clear(self):
        assert profile.load_persona("fair") is None
        assert profile.save_persona("fair", "  我是定制的晴。  ") is True
        assert profile.load_persona("fair") == "我是定制的晴。"
        profile.clear_persona("fair")
        assert profile.load_persona("fair") is None

    def test_unknown_agent_rejected(self):
        assert profile.save_persona("zephyr", "x") is False
        assert profile.load_persona("zephyr") is None

    def test_personas_are_per_agent(self):
        profile.save_persona("fog", "雾的新设定")
        profile.save_persona("fair", "晴的新设定")
        assert profile.load_persona("fog") == "雾的新设定"
        assert profile.load_persona("fair") == "晴的新设定"

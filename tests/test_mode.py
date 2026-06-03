"""Tests for the interactive ModeController."""

from __future__ import annotations

import pytest
import yaml

from weather_agents.cli.mode import InteractiveMode, ModeController


@pytest.fixture
def isolated_user_dir(monkeypatch, tmp_path):
    """Redirect USER_CONFIG_DIR so tests never touch the real ~/.skyloom."""
    import weather_agents.core.config as cfg_mod

    user_dir = tmp_path / ".skyloom"
    user_dir.mkdir()
    monkeypatch.setattr(cfg_mod, "USER_CONFIG_DIR", user_dir)
    cfg_mod.invalidate_cache()
    yield user_dir
    cfg_mod.invalidate_cache()


class TestEnum:
    def test_three_modes_exist(self):
        assert {m.value for m in InteractiveMode} == {"default", "plan", "auto"}

    def test_parse_accepts_known_values(self):
        assert InteractiveMode.parse("default") is InteractiveMode.DEFAULT
        assert InteractiveMode.parse("PLAN") is InteractiveMode.PLAN
        assert InteractiveMode.parse(" auto ") is InteractiveMode.AUTO

    def test_parse_rejects_unknown(self):
        assert InteractiveMode.parse("turbo") is None
        assert InteractiveMode.parse(None) is None
        assert InteractiveMode.parse("") is None


class TestController:
    def test_default_initial_mode_is_default(self, isolated_user_dir):
        c = ModeController()
        assert c.current is InteractiveMode.DEFAULT

    def test_construction_does_no_io(self, isolated_user_dir, monkeypatch):
        # Regression: pre-lazy ModeController() called load_config() in __init__
        # adding ~2s to every cli subcommand startup. Loading must be deferred
        # until the first .current read.
        calls = {"n": 0}
        import weather_agents.cli.mode as mode_mod

        orig = mode_mod.load_config

        def counted():
            calls["n"] += 1
            return orig()

        monkeypatch.setattr(mode_mod, "load_config", counted)
        ModeController()
        assert calls["n"] == 0, "construction must not trigger load_config"

    def test_set_persists_to_user_config(self, isolated_user_dir):
        c = ModeController()
        c.set(InteractiveMode.PLAN)
        data = yaml.safe_load((isolated_user_dir / "config.yaml").read_text("utf-8"))
        assert data["cli"]["interactive_mode"] == "plan"

    def test_initial_loads_from_persisted_value(self, isolated_user_dir):
        # Pretend a previous session saved auto.
        (isolated_user_dir / "config.yaml").write_text(
            yaml.dump({"cli": {"interactive_mode": "auto"}}), encoding="utf-8"
        )
        import weather_agents.core.config as cfg_mod

        cfg_mod.invalidate_cache()
        c = ModeController()
        assert c.current is InteractiveMode.AUTO

    def test_invalid_persisted_value_falls_back_to_default(self, isolated_user_dir):
        (isolated_user_dir / "config.yaml").write_text(
            yaml.dump({"cli": {"interactive_mode": "bogus"}}), encoding="utf-8"
        )
        import weather_agents.core.config as cfg_mod

        cfg_mod.invalidate_cache()
        c = ModeController()
        assert c.current is InteractiveMode.DEFAULT

    def test_cycle_order(self, isolated_user_dir):
        c = ModeController(initial=InteractiveMode.DEFAULT)
        assert c.cycle() is InteractiveMode.PLAN
        assert c.cycle() is InteractiveMode.AUTO
        assert c.cycle() is InteractiveMode.DEFAULT

    def test_cycle_persists(self, isolated_user_dir):
        c = ModeController(initial=InteractiveMode.DEFAULT)
        c.cycle()
        data = yaml.safe_load((isolated_user_dir / "config.yaml").read_text("utf-8"))
        assert data["cli"]["interactive_mode"] == "plan"

    def test_set_persist_false_skips_disk(self, isolated_user_dir):
        c = ModeController(initial=InteractiveMode.DEFAULT)
        c.set(InteractiveMode.AUTO, persist=False)
        assert c.current is InteractiveMode.AUTO
        assert not (isolated_user_dir / "config.yaml").exists()

    def test_set_preserves_other_user_config(self, isolated_user_dir):
        # Regression: an earlier version did its own write and dropped every
        # other key on the file (api_keys, model overrides, …) on each save.
        (isolated_user_dir / "config.yaml").write_text(
            yaml.dump(
                {
                    "llm": {
                        "default_model": "deepseek/deepseek-v4-flash",
                        "api_keys": {"openai": "sk-x"},
                    },
                    "memory": {"short_term_limit": 80},
                }
            ),
            encoding="utf-8",
        )
        c = ModeController(initial=InteractiveMode.DEFAULT)
        c.set(InteractiveMode.PLAN)
        data = yaml.safe_load((isolated_user_dir / "config.yaml").read_text("utf-8"))
        assert data["cli"]["interactive_mode"] == "plan"
        assert data["llm"]["api_keys"]["openai"] == "sk-x"
        assert data["llm"]["default_model"] == "deepseek/deepseek-v4-flash"
        assert data["memory"]["short_term_limit"] == 80

    def test_label_unique_per_mode(self, isolated_user_dir):
        labels = set()
        for m in InteractiveMode:
            c = ModeController(initial=m)
            labels.add(c.label()[0])
        assert len(labels) == 3

    def test_describe_non_empty(self, isolated_user_dir):
        for m in InteractiveMode:
            c = ModeController(initial=m)
            assert c.describe()


class TestConfigIntegration:
    def test_app_config_cli_section_default(self, isolated_user_dir):
        from weather_agents.core.config import load_config

        cfg = load_config()
        assert cfg.cli.interactive_mode == "default"

    def test_app_config_picks_up_user_override(self, isolated_user_dir):
        (isolated_user_dir / "config.yaml").write_text(
            yaml.dump({"cli": {"interactive_mode": "plan"}}), encoding="utf-8"
        )
        import weather_agents.core.config as cfg_mod

        cfg_mod.invalidate_cache()
        from weather_agents.core.config import load_config

        cfg = load_config()
        assert cfg.cli.interactive_mode == "plan"

"""Tests for configuration system."""

from __future__ import annotations

# ── Unit tests for config logic ─────────────────────────────────────────────


class TestConfigCore:
    def test_load_model_catalog(self):
        from weather_agents.core.config import load_model_catalog

        catalog = load_model_catalog()
        assert isinstance(catalog, dict)
        assert len(catalog) > 0

    def test_model_catalog_has_expected_providers(self):
        from weather_agents.core.config import load_model_catalog

        catalog = load_model_catalog()
        providers = {k.lower() for k in catalog}
        assert "openai" in providers or "anthropic" in providers

    def test_set_and_delete_config(self, temp_config_dir):
        from weather_agents.core.config import delete_config, load_config, set_config

        ok, msg = set_config("temperature", "0.33")
        assert ok, msg

        cfg = load_config()
        assert cfg.llm.temperature == 0.33

        ok, msg = delete_config("temperature")
        assert ok, msg

        cfg = load_config()
        assert cfg.llm.temperature == 0.7

    def test_set_model_config(self, temp_config_dir):
        from weather_agents.core.config import delete_config, load_config, set_config

        set_config("model.fog", "gpt-4o")
        cfg = load_config()
        assert cfg.agents.fog.model == "gpt-4o"

        delete_config("model.fog")
        cfg = load_config()
        assert cfg.agents.fog.model is None

    def test_set_api_key(self, temp_config_dir):
        from weather_agents.core.config import delete_config, load_config, set_config

        set_config("api_key.openai", "sk-test-key-123")
        cfg = load_config()
        assert cfg.llm.api_keys.get("openai") == "sk-test-key-123"

        delete_config("api_key.openai")
        cfg = load_config()
        assert "openai" not in cfg.llm.api_keys

    def test_set_invalid_key(self):
        from weather_agents.core.config import set_config

        ok, msg = set_config("invalid.key", "value")
        assert not ok

    def test_delete_nonexistent(self, temp_config_dir):
        from weather_agents.core.config import delete_config

        ok, msg = delete_config("temperature")
        assert ok

    def test_set_unknown_agent(self):
        from weather_agents.core.config import set_config

        ok, msg = set_config("model.nonexistent", "gpt-4o")
        assert not ok

    def test_set_default_model(self, temp_config_dir):
        from weather_agents.core.config import delete_config, load_config, set_config

        set_config("default_model", "gpt-4o")
        cfg = load_config()
        assert cfg.llm.default_model == "gpt-4o"

        delete_config("default_model")
        cfg = load_config()
        assert cfg.llm.default_model == "deepseek/deepseek-v4-flash"


class TestConfigValidation:
    def test_temperature_out_of_range_rejected(self):
        from weather_agents.core.config import set_config

        ok, msg = set_config("temperature", "5.0")
        assert not ok
        assert "temperature" in msg.lower()

    def test_temperature_negative_rejected(self):
        from weather_agents.core.config import set_config

        ok, msg = set_config("temperature", "-0.5")
        assert not ok

    def test_max_tokens_negative_rejected(self):
        from weather_agents.core.config import set_config

        ok, _ = set_config("max_tokens", "-100")
        assert not ok

    def test_timeout_too_large_rejected(self):
        from weather_agents.core.config import set_config

        ok, _ = set_config("timeout", "99999")
        assert not ok

    def test_invalid_float_value(self):
        from weather_agents.core.config import set_config

        ok, msg = set_config("temperature", "not-a-number")
        assert not ok
        assert "invalid" in msg.lower()


class TestEnvResolution:
    def test_resolve_env_existing(self, monkeypatch):
        from weather_agents.core.config import _resolve_env

        monkeypatch.setenv("WA_TEST_KEY", "hello")
        assert _resolve_env("${WA_TEST_KEY}") == "hello"

    def test_resolve_env_missing_returns_empty_and_warns(self, monkeypatch, caplog):
        import logging

        from weather_agents.core.config import _resolve_env

        monkeypatch.delenv("WA_MISSING_KEY", raising=False)
        with caplog.at_level(logging.WARNING, logger="weather_agents.config"):
            result = _resolve_env("${WA_MISSING_KEY}")
        assert result == ""
        assert any("WA_MISSING_KEY" in r.message for r in caplog.records)

    def test_resolve_env_passthrough(self):
        from weather_agents.core.config import _resolve_env

        assert _resolve_env("plain-value") == "plain-value"


class TestDotenvLoading:
    def test_load_dotenv_sets_vars(self, monkeypatch, tmp_path):
        from weather_agents.core.config import _load_dotenv

        dotenv_file = tmp_path / ".env"
        dotenv_file.write_text(
            "WA_DOTENV_TEST=hello_world\n# comment line\nWA_DOTENV_OTHER=other_val"
        )

        monkeypatch.delenv("WA_DOTENV_TEST", raising=False)
        monkeypatch.delenv("WA_DOTENV_OTHER", raising=False)
        monkeypatch.chdir(tmp_path)

        _load_dotenv()
        import os

        assert os.environ.get("WA_DOTENV_TEST") == "hello_world"
        assert os.environ.get("WA_DOTENV_OTHER") == "other_val"

    def test_load_dotenv_does_not_override_existing(self, monkeypatch, tmp_path):
        from weather_agents.core.config import _load_dotenv

        dotenv_file = tmp_path / ".env"
        dotenv_file.write_text("EXISTING_VAR=new_value")

        monkeypatch.setenv("EXISTING_VAR", "original_value")
        monkeypatch.chdir(tmp_path)

        _load_dotenv()
        import os

        assert os.environ["EXISTING_VAR"] == "original_value"

    def test_load_dotenv_no_file(self, monkeypatch, tmp_path):
        from weather_agents.core.config import _load_dotenv

        monkeypatch.chdir(tmp_path)
        # Should not raise
        _load_dotenv()

    def test_load_dotenv_parses_quoted_values(self, monkeypatch, tmp_path):
        from weather_agents.core.config import _load_dotenv

        dotenv_file = tmp_path / ".env"
        dotenv_file.write_text("QUOTED_KEY=\"value with spaces\"\nSINGLE_KEY='single_val'")

        monkeypatch.delenv("QUOTED_KEY", raising=False)
        monkeypatch.delenv("SINGLE_KEY", raising=False)
        monkeypatch.chdir(tmp_path)

        _load_dotenv()
        import os

        assert os.environ.get("QUOTED_KEY") == "value with spaces"
        assert os.environ.get("SINGLE_KEY") == "single_val"


class TestLanguageConfig:
    def test_default_language_is_zh(self):
        from weather_agents.core.config import invalidate_cache, load_config

        invalidate_cache()
        cfg = load_config()
        assert cfg.llm.language == "zh"

    def test_language_loaded_from_config(self, temp_config_dir):
        from weather_agents.core.config import invalidate_cache, load_config

        invalidate_cache()
        cfg = load_config()
        # Default should be zh
        assert cfg.llm.language in ("zh", "en")


class TestMaxToolRoundsConfig:
    def test_default_max_tool_rounds(self):
        from weather_agents.core.config import AppConfig

        cfg = AppConfig()
        for name in ("fog", "rain", "frost", "snow", "dew"):
            agent_cfg = getattr(cfg.agents, name)
            assert agent_cfg.max_tool_rounds == 20

    def test_max_tool_rounds_loaded(self, temp_config_dir):
        from weather_agents.core.config import invalidate_cache, load_config, set_config

        set_config("default_model", "deepseek/deepseek-v4-flash")
        invalidate_cache()
        cfg = load_config()
        for name in ("fog", "rain", "frost", "snow", "dew"):
            agent_cfg = getattr(cfg.agents, name)
            assert agent_cfg.max_tool_rounds == 20

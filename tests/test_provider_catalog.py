"""Tests for the YAML-driven provider catalog (round 11).

Round 11 split the hard-coded ``_PROVIDER_ENV`` / ``_KNOWN_PROVIDERS``
sets in ``llm.py`` out into ``config/providers.yaml`` so wa now ships
recognition + env-var routing for 35+ providers (OpenAI / Anthropic /
DeepSeek + every major Chinese provider + Bedrock / Azure / Groq /
Together / Fireworks / Replicate / xAI / Perplexity / Ollama / vLLM /
LM Studio / llama.cpp). These tests pin the catalog shape so a
silent breakage (e.g. someone deleting providers.yaml or breaking
load_provider_catalog) gets caught.
"""

from __future__ import annotations


class TestLoadProviderCatalog:
    def test_loads_bundled_catalog(self):
        from weather_agents.core.config import (
            invalidate_provider_cache,
            load_provider_catalog,
        )

        invalidate_provider_cache()
        cat = load_provider_catalog()
        assert isinstance(cat, dict)
        # Spot-check coverage — these MUST be present in the shipped catalog.
        for must in (
            "openai",
            "anthropic",
            "deepseek",
            "google_gemini",
            "zhipu",
            "alibaba_dashscope",
            "moonshot",
            "doubao",
            "baidu_ernie",
            "tencent_hunyuan",
            "xai",
            "ollama",
        ):
            assert must in cat, f"shipped catalog missing provider: {must}"

    def test_each_entry_has_required_fields(self):
        from weather_agents.core.config import load_provider_catalog

        cat = load_provider_catalog()
        for prov_id, entry in cat.items():
            assert "env_var" in entry, f"{prov_id} missing env_var"
            assert "region" in entry, f"{prov_id} missing region"
            # Region must be one of the values the CLI display groups by.
            assert entry["region"] in {"US", "CN", "EU", "Local", "Aggregator", "Other"}, (
                f"{prov_id} has unknown region '{entry['region']}'"
            )

    def test_user_override_merges_on_top(self, tmp_path, monkeypatch):
        """User can drop a providers.yaml into ~/.weather-agents/ to add
        their own provider entries. The user file is merged on top of
        the bundled one so existing entries can be customised and new
        ones added without forking the install."""
        from weather_agents.core import config as cfg_mod

        # Point the user dir at a tmp_path with a custom providers.yaml.
        user_dir = tmp_path / ".weather-agents"
        user_dir.mkdir()
        (user_dir / "providers.yaml").write_text(
            "my_custom_provider:\n"
            "  env_var: MY_CUSTOM_KEY\n"
            "  region: US\n"
            "  notes: A private provider\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(cfg_mod, "USER_CONFIG_DIR", user_dir)
        cfg_mod.invalidate_provider_cache()
        cat = cfg_mod.load_provider_catalog()

        assert "my_custom_provider" in cat
        assert cat["my_custom_provider"]["env_var"] == "MY_CUSTOM_KEY"
        # Existing bundled entries still present after the merge.
        assert "openai" in cat

        cfg_mod.invalidate_provider_cache()


class TestGetProviderEnvVar:
    def test_canonical_id_resolves(self):
        from weather_agents.core.config import (
            get_provider_env_var,
            invalidate_provider_cache,
        )

        invalidate_provider_cache()
        assert get_provider_env_var("openai") == "OPENAI_API_KEY"
        assert get_provider_env_var("anthropic") == "ANTHROPIC_API_KEY"
        assert get_provider_env_var("zhipu") == "ZHIPUAI_API_KEY"

    def test_alias_resolves_to_canonical_env_var(self):
        """A SKILL.md or user config might reference ``glm`` instead of
        ``zhipu``; both must point at the same env var."""
        from weather_agents.core.config import get_provider_env_var

        assert get_provider_env_var("glm") == "ZHIPUAI_API_KEY"
        assert get_provider_env_var("kimi") == "MOONSHOT_API_KEY"
        assert get_provider_env_var("qwen") == "DASHSCOPE_API_KEY"
        assert get_provider_env_var("ark") == "ARK_API_KEY"

    def test_unknown_provider_falls_back_to_default_pattern(self):
        """Custom / never-heard-of providers should still resolve to a
        sensible env var name (UPPER_API_KEY) instead of None."""
        from weather_agents.core.config import get_provider_env_var

        assert get_provider_env_var("my-custom-provider") == "MY_CUSTOM_PROVIDER_API_KEY"

    def test_case_insensitive(self):
        from weather_agents.core.config import get_provider_env_var

        assert get_provider_env_var("OpenAI") == get_provider_env_var("openai")
        assert get_provider_env_var("DEEPSEEK") == get_provider_env_var("deepseek")


class TestResolveProviderAlias:
    def test_canonical_name_unchanged(self):
        from weather_agents.core.config import resolve_provider_alias

        assert resolve_provider_alias("zhipu") == "zhipu"
        assert resolve_provider_alias("anthropic") == "anthropic"

    def test_alias_resolves_to_canonical(self):
        from weather_agents.core.config import resolve_provider_alias

        assert resolve_provider_alias("glm") == "zhipu"
        assert resolve_provider_alias("kimi") == "moonshot"
        assert resolve_provider_alias("qwen") == "alibaba_dashscope"
        assert resolve_provider_alias("ark") == "doubao"
        assert resolve_provider_alias("gemini") == "google_gemini"

    def test_unknown_name_passes_through_lowercased(self):
        from weather_agents.core.config import resolve_provider_alias

        assert resolve_provider_alias("Custom") == "custom"


class TestLLMProviderEnvMapDerived:
    """``llm._PROVIDER_ENV`` and ``llm._KNOWN_PROVIDERS`` are now derived
    from the YAML catalog. Verify the derivation produced the expected
    coverage so a future "I'll just inline this" refactor doesn't
    silently shrink the supported set."""

    def test_provider_env_map_includes_all_catalog_entries(self):
        from weather_agents.core.config import load_provider_catalog
        from weather_agents.core.llm import _PROVIDER_ENV

        cat = load_provider_catalog()
        for prov_id, entry in cat.items():
            expected_env = entry.get("env_var")
            if expected_env:
                assert _PROVIDER_ENV.get(prov_id.lower()) == expected_env, (
                    f"_PROVIDER_ENV[{prov_id}] != {expected_env}"
                )

    def test_aliases_share_env_var(self):
        """``glm/glm-4`` model id routes through the same env var as
        ``zhipu/glm-4``. Verify alias coverage in the derived map."""
        from weather_agents.core.llm import _PROVIDER_ENV

        assert _PROVIDER_ENV.get("glm") == _PROVIDER_ENV.get("zhipu") == "ZHIPUAI_API_KEY"
        assert _PROVIDER_ENV.get("kimi") == _PROVIDER_ENV.get("moonshot") == "MOONSHOT_API_KEY"

    def test_known_providers_set_includes_chinese_providers(self):
        """Pre-round-11 the ``<provider>/<model>`` router rejected any
        Chinese provider prefix and fell through to OpenAI, surfacing as
        "OPENAI_API_KEY missing". Pin coverage of the post-refactor set."""
        from weather_agents.core.llm import _KNOWN_PROVIDERS

        for cn in (
            "deepseek",
            "zhipu",
            "moonshot",
            "doubao",
            "alibaba_dashscope",
            "baidu_ernie",
            "tencent_hunyuan",
            "minimax",
            "siliconflow",
        ):
            assert cn in _KNOWN_PROVIDERS, f"_KNOWN_PROVIDERS missing {cn}"


class TestModelCatalogCoverage:
    """models.yaml must list at least one current model per provider
    (and ideally ~4 flagship models). Round 12 expanded coverage from
    5 to 34 providers; this test pins that breadth so a future "clean
    up the yaml" PR can't silently remove a Chinese provider."""

    def test_every_listed_provider_has_at_least_one_model(self):
        from weather_agents.core.config import load_model_catalog

        cat = load_model_catalog()
        for provider, models in cat.items():
            assert models, f"models.yaml has no models for {provider}"

    def test_critical_providers_present(self):
        """Spot-check that the providers users most often switch to
        all have entries. Catches catalog regressions where someone
        deletes an entire YAML block by accident."""
        from weather_agents.core.config import load_model_catalog

        cat = load_model_catalog()
        for must in (
            "openai",
            "anthropic",
            "deepseek",
            "google_gemini",
            "zhipu",
            "alibaba_dashscope",
            "moonshot",
            "doubao",
            "baidu_ernie",
            "tencent_hunyuan",
            "xai",
            "ollama",
        ):
            assert must in cat, f"models.yaml dropped {must}"

    def test_every_model_provider_resolves_to_a_catalog_entry(self):
        """A model id pointing at a provider that isn't in
        providers.yaml (and isn't an alias of one) means the router
        can't route to it — silent misconfiguration. Lock the cross-
        reference, but accept aliases as a valid resolution since some
        ``provider:`` fields use the LiteLLM-style short prefix
        (``gemini`` for ``google_gemini``, ``vertex_ai`` for
        ``google_vertex``, ``bedrock`` for ``aws_bedrock``, etc.)."""
        from weather_agents.core.config import (
            load_model_catalog,
            load_provider_catalog,
            resolve_provider_alias,
        )

        provider_ids = set(load_provider_catalog().keys())
        models = load_model_catalog()
        missing: list[tuple[str, str]] = []
        for top_key, entries in models.items():
            for m in entries:
                p = m.get("provider")
                if not (isinstance(p, str) and p):
                    continue
                # Either the canonical id matches, or the alias resolves
                # to one. Anything else is a real misconfiguration.
                if p not in provider_ids and resolve_provider_alias(p) not in provider_ids:
                    missing.append((top_key, p))
        assert not missing, f"models referencing unknown providers: {missing}"

    def test_each_model_has_required_metadata(self):
        from weather_agents.core.config import load_model_catalog

        cat = load_model_catalog()
        for provider, models in cat.items():
            for m in models:
                assert m.get("name"), f"{provider}: model missing name"
                ctx = m.get("context_window")
                assert isinstance(ctx, (int, float)) and ctx > 0, (
                    f"{provider}/{m['name']}: missing/invalid context_window"
                )

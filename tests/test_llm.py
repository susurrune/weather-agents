"""Tests for the LLM client."""

from __future__ import annotations

import pytest

from weather_agents.core.llm import (
    LLMClient,
    LLMResponse,
    _is_transient_error,
    estimate_cost,
)


class TestEstimateCost:
    def test_known_model(self):
        cost = estimate_cost("gpt-4o-mini", 1000, 500)
        assert cost > 0
        assert cost < 1.0

    def test_unknown_model_fallback(self):
        cost = estimate_cost("unknown-model", 1000, 500)
        assert cost > 0

    def test_zero_tokens(self):
        cost = estimate_cost("gpt-4o", 0, 0)
        assert cost == 0.0

    def test_local_model_free(self):
        cost = estimate_cost("ollama/llama3", 10000, 5000)
        assert cost == 0.0


class TestLLMResponse:
    def test_defaults(self):
        r = LLMResponse(content="hello")
        assert r.content == "hello"
        assert r.tool_calls == []
        assert r.model == ""
        assert r.cost == 0.0

    def test_with_tool_calls(self):
        r = LLMResponse(
            content="",
            tool_calls=[{"id": "1", "name": "read_file", "arguments": {"path": "/tmp"}}],
            model="gpt-4o",
        )
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0]["name"] == "read_file"


class TestLLMClientUsageTracking:
    @pytest.fixture
    def client(self, app_config, tool_registry):
        return LLMClient(app_config, tool_registry)

    def test_initial_stats_empty(self, client):
        assert client.get_usage_stats() == {}
        assert client.get_total_cost() == 0.0

    def test_track_usage(self, client):
        client._track_usage("fog", "gpt-4o-mini", 100, 50)
        stats = client.get_usage_stats()
        assert "fog" in stats
        assert stats["fog"]["calls"] == 1
        assert stats["fog"]["prompt_tokens"] == 100
        assert stats["fog"]["completion_tokens"] == 50
        assert stats["fog"]["cost"] > 0

    def test_cumulative_tracking(self, client):
        client._track_usage("fog", "gpt-4o-mini", 100, 50)
        client._track_usage("fog", "gpt-4o-mini", 200, 100)
        stats = client.get_usage_stats()
        assert stats["fog"]["calls"] == 2
        assert stats["fog"]["prompt_tokens"] == 300

    def test_multiple_agents(self, client):
        client._track_usage("fog", "gpt-4o", 100, 50)
        client._track_usage("rain", "gpt-4o-mini", 200, 100)
        stats = client.get_usage_stats()
        assert len(stats) == 2
        assert "fog" in stats
        assert "rain" in stats

    def test_reset_stats(self, client):
        client._track_usage("fog", "gpt-4o", 100, 50)
        client.reset_usage_stats()
        assert client.get_usage_stats() == {}
        assert client.get_total_cost() == 0.0

    def test_budget_check_passes(self, app_config, tool_registry):
        client = LLMClient(app_config, tool_registry, cost_limit=10.0)
        client._check_budget()  # should not raise

    def test_budget_check_fails(self, app_config, tool_registry):
        client = LLMClient(app_config, tool_registry, cost_limit=0.001)
        client._track_usage("fog", "gpt-4o", 10000, 5000)
        with pytest.raises(RuntimeError, match="Cost limit exceeded"):
            client._check_budget()


class TestTransientErrorClassifier:
    def test_timeout_is_transient(self):
        # TimeoutError == asyncio.TimeoutError as of Python 3.11.
        assert _is_transient_error(TimeoutError())

    def test_connection_error_is_transient(self):
        assert _is_transient_error(ConnectionError())

    def test_status_429_is_transient(self):
        exc = RuntimeError("rate limited")
        exc.status_code = 429  # type: ignore[attr-defined]
        assert _is_transient_error(exc)

    def test_status_500_is_transient(self):
        exc = RuntimeError("server error")
        exc.status_code = 500  # type: ignore[attr-defined]
        assert _is_transient_error(exc)

    def test_status_400_is_not_transient(self):
        exc = RuntimeError("bad request")
        exc.status_code = 400  # type: ignore[attr-defined]
        assert not _is_transient_error(exc)

    def test_value_error_is_not_transient(self):
        # Programmer/config errors should not be retried.
        assert not _is_transient_error(ValueError("bad config"))
        assert not _is_transient_error(KeyError("missing"))

    def test_named_litellm_error_classes_are_transient(self):
        class RateLimitError(Exception):
            pass

        class APITimeoutError(Exception):
            pass

        assert _is_transient_error(RateLimitError())
        assert _is_transient_error(APITimeoutError())


class TestLLMCacheKey:
    def test_cache_key_includes_temperature(self):
        from weather_agents.core.cache import LLMCache

        cache = LLMCache(max_size=10, ttl_seconds=60)
        msgs = [{"role": "user", "content": "hi"}]
        cache.set("gpt-4o", msgs, "first answer", {"temperature": 0.5})
        # Different temperature must miss.
        assert cache.get("gpt-4o", msgs, {"temperature": 0.9}) is None
        # Same params must hit.
        assert cache.get("gpt-4o", msgs, {"temperature": 0.5}) == "first answer"

    def test_cache_refuses_short_responses(self):
        from weather_agents.core.cache import LLMCache

        cache = LLMCache(max_size=10, ttl_seconds=60)
        msgs = [{"role": "user", "content": "hi"}]
        cache.set("gpt-4o", msgs, "ok", {"temperature": 0.5})  # 2 chars — refused
        assert cache.get("gpt-4o", msgs, {"temperature": 0.5}) is None


class TestLLMCacheLifecycle:
    def test_expired_entry_returns_none(self):
        import time as _time

        from weather_agents.core.cache import LLMCache

        cache = LLMCache(max_size=10, ttl_seconds=0)  # 0-second TTL
        msgs = [{"role": "user", "content": "hi"}]
        cache.set("gpt-4o", msgs, "test answer", {"temperature": 0.5})
        # No need to sleep — cache lookup will see elapsed >= 0 and
        # treat the entry as expired (time.time() - ts > 0).
        assert cache.get("gpt-4o", msgs, {"temperature": 0.5}) is None

    def test_eviction_when_over_max_size(self):
        from weather_agents.core.cache import LLMCache

        cache = LLMCache(max_size=2, ttl_seconds=300)
        msgs = [{"role": "user", "content": "hi"}]
        cache.set("model-a", msgs, "long answer aaa")  # 15 chars, accepted
        cache.set("model-b", msgs, "long answer bbb")
        cache.set("model-c", msgs, "long answer ccc")  # evicts the oldest
        # model-a should have been evicted (LRU)
        assert cache.get("model-a", msgs) is None
        # model-b and model-c should still be present
        assert cache.get("model-b", msgs) == "long answer bbb"
        assert cache.get("model-c", msgs) == "long answer ccc"
        assert cache.size == 2

    def test_clear_empties_cache(self):
        from weather_agents.core.cache import LLMCache

        cache = LLMCache(max_size=10, ttl_seconds=300)
        msgs = [{"role": "user", "content": "hi"}]
        cache.set("gpt-4o", msgs, "some answer")
        assert cache.size == 1
        cache.clear()
        assert cache.size == 0
        assert cache.get("gpt-4o", msgs) is None

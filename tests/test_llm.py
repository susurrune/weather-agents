"""Tests for the LLM client."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from weather_agents.core.llm import (
    LLMClient,
    LLMResponse,
    _estimate_tokens,
    _format_user_facing_error,
    _get_litellm,
    _is_transient_error,
    _lazy_litellm,
    _split_provider,
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


class TestSplitProvider:
    def test_with_known_provider(self):
        provider, stripped = _split_provider("openai/gpt-4o")
        assert provider == "openai"
        assert stripped == "gpt-4o"

    def test_without_slash(self):
        provider, stripped = _split_provider("gpt-4o")
        assert provider is None
        assert stripped == "gpt-4o"

    def test_with_unknown_provider(self):
        provider, stripped = _split_provider("unknown/model")
        assert provider is None
        assert stripped == "unknown/model"  # unknown prefix not stripped

    def test_case_insensitive_provider(self):
        provider, stripped = _split_provider("OpenAI/gpt-4o")
        assert provider == "openai"


class TestFormatUserFacingError:
    def test_api_key_error(self):
        msg = _format_user_facing_error("gpt-4o", ValueError("invalid api_key"))
        assert "API key" in msg

    def test_rate_limit_error(self):
        msg = _format_user_facing_error("gpt-4o", ValueError("rate limit exceeded"))
        assert "速率受限" in msg

    def test_timeout_error(self):
        msg = _format_user_facing_error("gpt-4o", ValueError("timeout after 30s"))
        assert "超时" in msg

    def test_model_not_found(self):
        msg = _format_user_facing_error("gpt-4o", ValueError("model not found"))
        assert "模型 ID" in msg

    def test_bad_request(self):
        msg = _format_user_facing_error("gpt-4o", ValueError("bad request"))
        assert "Bad Request" in msg

    def test_generic_error(self):
        msg = _format_user_facing_error("gpt-4o", ValueError("something broke"))
        assert "调用失败" in msg

    def test_no_error(self):
        msg = _format_user_facing_error("gpt-4o", None)
        assert "unknown error" in msg or "调用失败" in msg


class TestEstimateTokens:
    def test_english_text(self):
        tokens = _estimate_tokens("hello world, this is a test")
        assert tokens >= 1

    def test_cjk_text(self):
        tokens = _estimate_tokens("你好世界，这是一个测试")
        assert tokens >= 1

    def test_mixed_text(self):
        tokens = _estimate_tokens("hello 你好 world 世界")
        assert tokens >= 1

    def test_empty_string(self):
        tokens = _estimate_tokens("")
        assert tokens == 1


class TestHasKeyForModel:
    @pytest.fixture
    def client(self, app_config, tool_registry):
        return LLMClient(app_config, tool_registry)

    def test_known_provider_with_key_set(self, client):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False):
            assert client._has_key_for_model("openai/gpt-4o")

    def test_known_provider_without_key(self, client):
        with patch.dict(os.environ, {}, clear=True):
            result = client._has_key_for_model("openai/gpt-4o")
            assert not result

    def test_unknown_provider_returns_true(self, client):
        result = client._has_key_for_model("unknown/model")
        assert result

    def test_model_name_matches_provider(self, client):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant"}, clear=False):
            assert client._has_key_for_model("claude-sonnet-4-6")


class TestLLMClientConfig:
    def test_get_model_default(self, app_config, tool_registry):
        client = LLMClient(app_config, tool_registry)
        model = client._get_model()
        assert model == app_config.llm.default_model

    def test_get_model_for_agent_with_config(self, app_config, tool_registry):
        app_config.agents.fog = Mock()
        app_config.agents.fog.model = "gpt-4o"
        client = LLMClient(app_config, tool_registry)
        model = client._get_model("fog")
        assert model == "gpt-4o"

    def test_get_model_for_agent_without_config(self, app_config, tool_registry):
        client = LLMClient(app_config, tool_registry)
        model = client._get_model("nonexistent")
        assert model == app_config.llm.default_model

    def test_get_retries(self, app_config, tool_registry):
        client = LLMClient(app_config, tool_registry)
        retries = client._get_retries()
        assert retries >= 0


class TestLLMClientComplete:
    @pytest.mark.asyncio
    async def test_complete_calls_with_retry(self, app_config, tool_registry):
        client = LLMClient(app_config, tool_registry)
        mock_result = LLMResponse(content="test response", model="gpt-4o-mini")
        client._complete_with_retry = AsyncMock(return_value=mock_result)
        result = await client.complete(
            [{"role": "user", "content": "hi"}],
            agent_name="fog",
        )
        assert result.content == "test response"
        client._complete_with_retry.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_complete_uses_model_override(self, app_config, tool_registry):
        client = LLMClient(app_config, tool_registry)
        client._complete_with_retry = AsyncMock(return_value=LLMResponse(content="ok"))
        await client.complete(
            [{"role": "user", "content": "hi"}],
            overrides={"model": "gpt-4o"},
        )
        args = client._complete_with_retry.call_args[0]
        assert args[0] == "gpt-4o"  # first positional arg is the model

    @pytest.mark.asyncio
    async def test_complete_triggers_budget_check(self, app_config, tool_registry):
        client = LLMClient(app_config, tool_registry, cost_limit=0.0)
        client._track_usage("fog", "gpt-4o", 9999999, 9999999)
        with pytest.raises(RuntimeError, match="Cost limit exceeded"):
            await client.complete([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self, app_config, tool_registry):
        client = LLMClient(app_config, tool_registry)
        client._complete_with_retry = AsyncMock(
            side_effect=[RuntimeError("first failed"), LLMResponse(content="fallback ok")]
        )
        result = await client.complete(
            [{"role": "user", "content": "hi"}],
            overrides={"model": "gpt-4o"},
        )
        assert result.content == "fallback ok"

    @pytest.mark.asyncio
    async def test_all_fallbacks_fail(self, app_config, tool_registry):
        client = LLMClient(app_config, tool_registry)
        client._complete_with_retry = AsyncMock(side_effect=RuntimeError("all failed"))
        result = await client.complete(
            [{"role": "user", "content": "hi"}],
            overrides={"model": "gpt-4o"},
        )
        assert "调用失败" in result.content


class TestCompleteWithRetry:
    """Tests _complete_with_retry by mocking litellm.acompletion()."""

    @pytest.fixture
    def client(self, app_config, tool_registry):
        return LLMClient(app_config, tool_registry)

    def _mock_litellm_response(
        self,
        content="",
        tool_calls=None,
        model="gpt-4o-mini",
        prompt_tokens=10,
        completion_tokens=20,
    ):
        """Build a mock response shaped like litellm's acompletion return value."""
        msg = Mock()
        msg.content = content
        msg.tool_calls = tool_calls
        msg.reasoning_content = None
        choice = Mock()
        choice.message = msg
        usage = Mock()
        usage.prompt_tokens = prompt_tokens
        usage.completion_tokens = completion_tokens
        resp = Mock()
        resp.choices = [choice]
        resp.usage = usage
        resp.model = model
        return resp

    @pytest.mark.asyncio
    async def test_successful_completion(self, client):
        mock_resp = self._mock_litellm_response(content="Hello world")
        with patch("weather_agents.core.llm._get_litellm") as mock_get:
            mock_lm = Mock()
            mock_lm.acompletion = AsyncMock(return_value=mock_resp)
            mock_get.return_value = mock_lm

            result = await client._complete_with_retry(
                "gpt-4o-mini",
                [{"role": "user", "content": "hi"}],
            )

        assert result.content == "Hello world"
        assert result.model == "gpt-4o-mini"
        assert result.usage["prompt_tokens"] == 10
        assert result.usage["completion_tokens"] == 20

    @pytest.mark.asyncio
    async def test_completion_with_tool_calls(self, client):
        tc = Mock()
        tc.id = "call_1"
        tc.type = "function"
        tc.function.name = "read_file"
        tc.function.arguments = '{"path": "/tmp"}'
        mock_resp = self._mock_litellm_response(tool_calls=[tc])
        with patch("weather_agents.core.llm._get_litellm") as mock_get:
            mock_lm = Mock()
            mock_lm.acompletion = AsyncMock(return_value=mock_resp)
            mock_get.return_value = mock_lm

            result = await client._complete_with_retry(
                "gpt-4o-mini",
                [{"role": "user", "content": "read file"}],
                tools=["read_file"],
            )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == "read_file"

    @pytest.mark.asyncio
    async def test_retry_on_transient_error(self, client):
        mock_resp = self._mock_litellm_response(content="finally ok")
        with patch("weather_agents.core.llm._get_litellm") as mock_get:
            mock_lm = Mock()
            mock_lm.acompletion = AsyncMock(side_effect=[TimeoutError("timeout"), mock_resp])
            mock_get.return_value = mock_lm

            result = await client._complete_with_retry(
                "gpt-4o-mini",
                [{"role": "user", "content": "hi"}],
            )

        assert result.content == "finally ok"
        assert mock_lm.acompletion.await_count == 2

    @pytest.mark.asyncio
    async def test_non_transient_error_raises(self, client):
        with patch("weather_agents.core.llm._get_litellm") as mock_get:
            mock_lm = Mock()
            mock_lm.acompletion = AsyncMock(side_effect=ValueError("bad request"))
            mock_get.return_value = mock_lm

            with pytest.raises(ValueError):
                await client._complete_with_retry(
                    "gpt-4o-mini",
                    [{"role": "user", "content": "hi"}],
                )

    @pytest.mark.asyncio
    async def test_cache_hit_skips_api_call(self, client):
        # Pre-populate cache with matching params that _complete_with_retry uses
        msgs = [{"role": "user", "content": "hello"}]
        cache_params = {
            "temperature": client.config.llm.temperature,
            "max_tokens": client.config.llm.max_tokens,
        }
        client.cache.set("gpt-4o-mini", msgs, "cached response", cache_params)

        with patch("weather_agents.core.llm._get_litellm") as mock_get:
            mock_lm = Mock()
            mock_lm.acompletion = AsyncMock()
            mock_get.return_value = mock_lm

            result = await client._complete_with_retry("gpt-4o-mini", msgs)

        assert result.content == "cached response"
        mock_lm.acompletion.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_provider_routing(self, client):
        mock_resp = self._mock_litellm_response(content="routed")
        with patch("weather_agents.core.llm._get_litellm") as mock_get:
            mock_lm = Mock()
            mock_lm.acompletion = AsyncMock(return_value=mock_resp)
            mock_get.return_value = mock_lm

            await client._complete_with_retry(
                "anthropic/claude-sonnet-4-6",
                [{"role": "user", "content": "hi"}],
            )

        call_kwargs = mock_lm.acompletion.call_args[1]
        assert call_kwargs["custom_llm_provider"] == "anthropic"
        assert call_kwargs["model"] == "anthropic/claude-sonnet-4-6"


class TestStream:
    @pytest.fixture
    def client(self, app_config, tool_registry):
        return LLMClient(app_config, tool_registry)

    def _make_chunk(self, content: str) -> Mock:
        delta = Mock()
        delta.content = content
        delta.tool_calls = None
        delta.reasoning_content = None
        choice = Mock()
        choice.delta = delta
        chunk = Mock()
        chunk.choices = [choice]
        return chunk

    @pytest.mark.asyncio
    async def test_stream_yields_content(self, client):
        chunks = [self._make_chunk("Hello"), self._make_chunk(" World")]

        # Make a mock that works as an async iterable
        async def async_iter():
            for c in chunks:
                yield c

        with patch("weather_agents.core.llm._get_litellm") as mock_get:
            mock_lm = Mock()
            mock_lm.acompletion = AsyncMock(return_value=async_iter())
            mock_lm.token_counter = Mock(return_value=5)
            mock_get.return_value = mock_lm

            collected = []
            async for text in client.stream([{"role": "user", "content": "hi"}]):
                collected.append(text)

        assert "".join(collected) == "Hello World"

    @pytest.mark.asyncio
    async def test_stream_timeout_yields_error(self, client):
        with patch("weather_agents.core.llm._get_litellm") as mock_get:
            mock_lm = Mock()
            mock_lm.acompletion = AsyncMock(side_effect=TimeoutError("stream timeout"))
            mock_get.return_value = mock_lm

            collected = []
            async for text in client.stream([{"role": "user", "content": "hi"}]):
                collected.append(text)

        # The message is "timed out" (not "timeout"), so check for "timed"
        assert any("timed" in t.lower() for t in collected)

    @pytest.mark.asyncio
    async def test_stream_exception_yields_error(self, client):
        with patch("weather_agents.core.llm._get_litellm") as mock_get:
            mock_lm = Mock()
            mock_lm.acompletion = AsyncMock(side_effect=ValueError("api error"))
            mock_get.return_value = mock_lm

            collected = []
            async for text in client.stream([{"role": "user", "content": "hi"}]):
                collected.append(text)

        assert any("error" in t.lower() for t in collected)


class TestStreamWithTools:
    @pytest.fixture
    def client(self, app_config, tool_registry):
        return LLMClient(app_config, tool_registry)

    def _make_content_chunk(self, text: str) -> Mock:
        delta = Mock()
        delta.content = text
        delta.tool_calls = None
        delta.reasoning_content = None
        choice = Mock()
        choice.delta = delta
        chunk = Mock()
        chunk.choices = [choice]
        return chunk

    def _make_tool_chunk(self, idx: int, id: str = "", name: str = "", args: str = "") -> Mock:
        tc_delta = Mock()
        tc_delta.index = idx
        tc_delta.id = id
        tc_delta.function = Mock()
        tc_delta.function.name = name
        tc_delta.function.arguments = args
        delta = Mock()
        delta.content = None
        delta.tool_calls = [tc_delta]
        delta.reasoning_content = None
        choice = Mock()
        choice.delta = delta
        chunk = Mock()
        chunk.choices = [choice]
        return chunk

    @pytest.mark.asyncio
    async def test_stream_with_tool_calls(self, client):
        chunks = [
            self._make_tool_chunk(0, id="call_1", name="read_file"),
            self._make_tool_chunk(0, args='{"path": "/tmp"}'),
        ]

        async def async_iter():
            for c in chunks:
                yield c

        with patch("weather_agents.core.llm._get_litellm") as mock_get:
            mock_lm = Mock()
            mock_lm.acompletion = AsyncMock(return_value=async_iter())
            mock_lm.token_counter = Mock(return_value=5)
            mock_get.return_value = mock_lm

            events = []
            async for event in client.stream_with_tools(
                [{"role": "user", "content": "read"}],
                tools=["read_file"],
                tool_registry=client.tool_registry,
            ):
                events.append(event)

        tool_events = [e for e in events if e.type == "tool_call"]
        assert len(tool_events) == 1
        assert tool_events[0].tool_call["function"]["name"] == "read_file"

        done_events = [e for e in events if e.type == "done"]
        assert len(done_events) == 1

    @pytest.mark.asyncio
    async def test_stream_with_tools_content_and_tool(self, client):
        chunks = [
            self._make_content_chunk("I'll read the file"),
            self._make_tool_chunk(0, id="call_1", name="read_file"),
            self._make_tool_chunk(0, args='{"path": "/tmp"}'),
        ]

        async def async_iter():
            for c in chunks:
                yield c

        with patch("weather_agents.core.llm._get_litellm") as mock_get:
            mock_lm = Mock()
            mock_lm.acompletion = AsyncMock(return_value=async_iter())
            mock_lm.token_counter = Mock(return_value=5)
            mock_get.return_value = mock_lm

            events = []
            async for event in client.stream_with_tools(
                [{"role": "user", "content": "read"}],
            ):
                events.append(event)

        content_events = [e for e in events if e.type == "content"]
        assert len(content_events) == 1
        assert content_events[0].text == "I'll read the file"

    @pytest.mark.asyncio
    async def test_stream_with_tools_all_fallbacks_fail(self, client):
        with patch("weather_agents.core.llm._get_litellm") as mock_get:
            mock_lm = Mock()
            mock_lm.acompletion = AsyncMock(side_effect=ValueError("api error"))
            mock_get.return_value = mock_lm

            events = []
            async for event in client.stream_with_tools(
                [{"role": "user", "content": "read"}],
            ):
                events.append(event)

        error_events = [e for e in events if e.type == "error"]
        assert len(error_events) == 1


class TestLazyLitellm:
    def test_lazy_import_returns_module(self):
        lm = _lazy_litellm()
        assert lm is not None

    def test_get_litellm_sentinel(self):
        lm = _get_litellm()
        assert lm is not None


class TestAnthropicPromptCache:
    """Anthropic prompt cache markers cut input cost ~80% on round 2+
    of a turn by reusing a 5-min KV cache of system + tools. The
    transformation must be a no-op for non-Anthropic providers."""

    def test_detects_anthropic_models(self):
        from weather_agents.core.llm import _is_anthropic_model

        assert _is_anthropic_model("claude-3-5-sonnet") is True
        assert _is_anthropic_model("anthropic/claude-opus-4") is True
        assert _is_anthropic_model("claude-haiku-3") is True
        assert _is_anthropic_model("deepseek/v4") is False
        assert _is_anthropic_model("gpt-4o") is False
        assert _is_anthropic_model("openai/gpt-5") is False

    def test_anthropic_system_gets_cache_control(self):
        from weather_agents.core.llm import _apply_anthropic_cache_control

        msgs = [
            {"role": "system", "content": "You are fog."},
            {"role": "user", "content": "hi"},
        ]
        m2, _ = _apply_anthropic_cache_control("claude-3-5-sonnet", msgs, None)
        # System rewritten to content-block form with cache marker
        assert isinstance(m2[0]["content"], list)
        block = m2[0]["content"][0]
        assert block["type"] == "text"
        assert block["text"] == "You are fog."
        assert block["cache_control"] == {"type": "ephemeral"}
        # User message untouched
        assert m2[1] == {"role": "user", "content": "hi"}

    def test_anthropic_last_tool_gets_cache_control(self):
        from weather_agents.core.llm import _apply_anthropic_cache_control

        tools = [
            {"type": "function", "function": {"name": "a"}},
            {"type": "function", "function": {"name": "b"}},
        ]
        _, t2 = _apply_anthropic_cache_control(
            "claude-3-5-sonnet", [{"role": "user", "content": "x"}], tools
        )
        assert t2 is not None
        # Only the LAST tool gets the marker (caches all earlier ones too)
        assert "cache_control" not in t2[0]
        assert t2[-1]["cache_control"] == {"type": "ephemeral"}

    def test_non_anthropic_passthrough(self):
        from weather_agents.core.llm import _apply_anthropic_cache_control

        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        tools = [{"type": "function", "function": {"name": "x"}}]
        m2, t2 = _apply_anthropic_cache_control("deepseek/v4", msgs, tools)
        # Inputs returned UNCHANGED for non-Anthropic targets
        assert m2 is msgs or m2 == msgs
        assert t2 is tools or t2 == tools
        # No cache_control anywhere
        assert "cache_control" not in str(m2[0]["content"])
        assert "cache_control" not in t2[0]

    def test_non_destructive(self):
        """Helper must not mutate the inputs."""
        from weather_agents.core.llm import _apply_anthropic_cache_control

        msgs = [{"role": "system", "content": "sys"}]
        tools = [{"type": "function", "function": {"name": "x"}}]
        msgs_before = [dict(m) for m in msgs]
        tools_before = [dict(t) for t in tools]
        _apply_anthropic_cache_control("claude-3-5-sonnet", msgs, tools)
        assert msgs == msgs_before
        assert tools == tools_before

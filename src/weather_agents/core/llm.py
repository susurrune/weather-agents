"""LLM abstraction layer using LiteLLM with retry, fallback, cost tracking, and budget control."""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm

from weather_agents.core.cache import LLMCache
from weather_agents.core.config import AppConfig
from weather_agents.core.logger import get_logger, log_event
from weather_agents.core.tool import ToolRegistry

log = get_logger("llm")

# Silence LiteLLM's noisy stderr banners (the "Give Feedback / Get Help" lines
# and verbose dump on every failure). Users on WA_DEBUG=1 can opt back in.
if os.environ.get("WA_DEBUG") != "1":
    litellm.suppress_debug_info = True
    with contextlib.suppress(Exception):
        litellm.set_verbose = False  # type: ignore[attr-defined]
    # Tame LiteLLM's loggers as well.
    import logging as _logging

    for _name in ("LiteLLM", "litellm", "litellm.router", "litellm.proxy"):
        _logging.getLogger(_name).setLevel(_logging.ERROR)


# When the user gives a `<provider>/<model>` form, force LiteLLM to route by
# the prefix even if `<model>` is not in its built-in registry. Without this,
# unknown deepseek/anthropic IDs (e.g. preview models) fall through to the
# default OpenAI client and surface "OPENAI_API_KEY missing" instead of
# routing to the right provider.
_KNOWN_PROVIDERS = {
    "openai",
    "azure",
    "anthropic",
    "deepseek",
    "ollama",
    "groq",
    "mistral",
    "cohere",
    "together_ai",
    "openrouter",
    "gemini",
    "vertex_ai",
}


def _split_provider(model: str) -> tuple[str | None, str]:
    """Return (provider, stripped_model) if model looks like `<provider>/<name>`."""
    if "/" not in model:
        return None, model
    head, tail = model.split("/", 1)
    if head.lower() in _KNOWN_PROVIDERS:
        return head.lower(), tail
    return None, model


_PROVIDER_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cohere": "COHERE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def _format_user_facing_error(model: str, err: BaseException | None) -> str:
    """Turn a low-level LiteLLM exception into a one-line, actionable message."""
    text = str(err) if err else "unknown error"
    provider, _ = _split_provider(model)

    # AuthenticationError / missing API key is the #1 case after a fresh install.
    lowered = text.lower()
    if "api_key" in lowered or "authentication" in lowered or "unauthorized" in lowered:
        env_var = _PROVIDER_ENV.get(provider or "", "the appropriate *_API_KEY")
        # Report which providers have keys configured
        configured = [p for p, e in _PROVIDER_ENV.items() if os.environ.get(e)]
        hint = f"已配置: {', '.join(configured)}。" if configured else "未配置任何 API key。"
        return (
            f"❌  {model} 调用失败：缺少或无效的 API key。\n"
            f"请确认 `{env_var}` 已设置，或运行 `wa init` 重新配置。{hint}"
        )
    if "rate limit" in lowered or "429" in text:
        return f"❌  {model} 速率受限，请稍后重试。"
    if "timeout" in lowered:
        return f"❌  {model} 请求超时，请稍后重试或调高 `wa config set timeout 180`。"
    if "model" in lowered and ("not found" in lowered or "does not exist" in lowered):
        return (
            f"❌  {model} 不是该 provider 的有效模型 ID。\n"
            f"运行 `wa config models` 查看可用模型，或 `wa init` 重新选择。"
        )
    # Bad request (often due to malformed message sequence from corrupted memory)
    err_name = type(err).__name__.lower() if err else ""
    if (
        any(
            kw in lowered
            for kw in ("bad request", "invalid_request", "tool_calls", "tool messages")
        )
        or "badrequest" in err_name
    ):
        short = text.splitlines()[0][:200]
        return (
            f"❌  {model} 调用失败 (Bad Request)：{short}\n"
            f"会话消息序列可能损坏，可运行 `wa memory clear` 清理后重试。"
        )
    # Generic fallback — short, no stack trace, no LiteLLM banner.
    short = text.splitlines()[0][:200] if text and text.strip() else type(err).__name__
    return f"❌  {model} 调用失败：{short}"


def _estimate_tokens(text: str) -> int:
    """Estimate token count for mixed CJK/English text.

    CJK characters ~2 tokens each, non-CJK ~4 chars per token.
    """
    cjk = sum(1 for c in text if "一" <= c <= "鿿" or "　" <= c <= "〿")
    other = len(text) - cjk
    return max(1, cjk * 2 + other // 4)


# Cost per 1K tokens (input / output) — USD
_MODEL_COST_ESTIMATES: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4.1": (0.002, 0.008),
    "gpt-4.1-mini": (0.0004, 0.0016),
    "gpt-4.1-nano": (0.0001, 0.0004),
    "o3": (0.01, 0.04),
    "o4-mini": (0.0011, 0.0044),
    "claude-sonnet-4-6": (0.003, 0.015),
    "claude-opus-4-7": (0.005, 0.025),
    "claude-haiku-4-5": (0.0008, 0.004),
    "deepseek-v4-flash": (0.00014, 0.00028),
    "deepseek-v4-pro": (0.00174, 0.00348),
    "deepseek/deepseek-v4-flash": (0.00014, 0.00028),
    "deepseek/deepseek-v4-pro": (0.00174, 0.00348),
    "gemini/gemini-2.5-flash": (0.0003, 0.0025),
    "gemini/gemini-2.5-pro": (0.00125, 0.01),
    "ollama/llama3": (0.0, 0.0),
    "ollama/qwen2.5": (0.0, 0.0),
}

_FALLBACK_CHAINS: dict[str, list[str]] = {
    "gpt-4o": ["gpt-4o-mini"],
    "gpt-4o-mini": ["gpt-4o"],
    "gpt-4.1": ["gpt-4.1-mini", "gpt-4o-mini"],
    "gpt-4.1-mini": ["gpt-4o-mini"],
    "gpt-4.1-nano": ["gpt-4.1-mini"],
    "o3": ["o4-mini", "gpt-4.1"],
    "o4-mini": ["gpt-4.1-mini"],
    "claude-sonnet-4-6": ["claude-haiku-4-5", "gpt-4.1-mini"],
    "claude-opus-4-7": ["claude-sonnet-4-6", "gpt-4.1"],
    "claude-haiku-4-5": ["gpt-4.1-mini"],
    "deepseek-v4-flash": ["gpt-4.1-mini"],
    "deepseek-v4-pro": ["deepseek-v4-flash", "gpt-4.1-mini"],
    "deepseek/deepseek-v4-flash": ["gpt-4.1-mini"],
    "deepseek/deepseek-v4-pro": ["deepseek/deepseek-v4-flash", "gpt-4.1-mini"],
    "gemini/gemini-2.5-flash": ["gemini/gemini-2.5-pro", "gpt-4.1-mini"],
    "gemini/gemini-2.5-pro": ["gpt-4.1"],
}

_RETRYABLE_STATUSES = {408, 425, 429, 500, 502, 503, 504}


def _is_transient_error(exc: BaseException) -> bool:
    """Decide whether an exception is worth retrying.

    Retries only on known-transient classes (timeouts, rate limits, 5xx) rather
    than blindly retrying every error — which used to mask config bugs.
    """
    status = getattr(exc, "status_code", 0) or getattr(exc, "http_status", 0)
    if status and status in _RETRYABLE_STATUSES:
        return True
    if isinstance(exc, asyncio.TimeoutError | TimeoutError | ConnectionError):
        return True
    # LiteLLM-specific transient classes (best-effort, names are stable)
    name = type(exc).__name__
    return name in {
        "RateLimitError",
        "APITimeoutError",
        "APIConnectionError",
        "ServiceUnavailableError",
        "InternalServerError",
        "Timeout",
    }


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    costs = _MODEL_COST_ESTIMATES.get(model, (0.001, 0.002))
    return (prompt_tokens / 1000) * costs[0] + (completion_tokens / 1000) * costs[1]


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    model: str = ""
    usage: dict = field(default_factory=dict)
    cost: float = 0.0
    reasoning_content: str | None = None


@dataclass
class StreamEvent:
    """A single event in a streaming LLM response."""

    type: Literal["content", "tool_call", "done", "error", "reasoning"]
    text: str = ""
    tool_call: dict | None = None
    usage: dict | None = None
    reasoning_content: str | None = None


class LLMClient:
    """Unified LLM client with retry, fallback chains, caching, cost tracking, and budget control."""

    def __init__(
        self,
        config: AppConfig,
        tool_registry: ToolRegistry,
        cost_limit: float | None = None,
    ) -> None:
        self.config = config
        self.tool_registry = tool_registry
        self.cache = LLMCache(max_size=256, ttl_seconds=120)
        self._usage_stats: dict[str, dict] = {}
        self._total_cost: float = 0.0
        self._cost_limit = cost_limit

    def _get_model(self, agent_name: str | None = None) -> str:
        if agent_name:
            agent_cfg = getattr(self.config.agents, agent_name, None)
            if agent_cfg and agent_cfg.model:
                return str(agent_cfg.model)
        return self.config.llm.default_model

    def _get_retries(self) -> int:
        return getattr(self.config.llm, "max_retries", 2)

    def _track_usage(
        self,
        agent_name: str | None,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        key = agent_name or "default"
        if key not in self._usage_stats:
            self._usage_stats[key] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "calls": 0,
                "cost": 0.0,
            }
        s = self._usage_stats[key]
        s["prompt_tokens"] += prompt_tokens
        s["completion_tokens"] += completion_tokens
        s["calls"] += 1
        cost = estimate_cost(model, prompt_tokens, completion_tokens)
        s["cost"] += cost
        self._total_cost += cost

    def get_usage_stats(self) -> dict[str, dict]:
        return dict(self._usage_stats)

    def get_total_cost(self) -> float:
        return self._total_cost

    def reset_usage_stats(self) -> None:
        self._usage_stats.clear()
        self._total_cost = 0.0

    def _check_budget(self) -> None:
        if self._cost_limit is not None and self._total_cost >= self._cost_limit:
            raise RuntimeError(
                f"Cost limit exceeded: ${self._total_cost:.4f} >= ${self._cost_limit:.4f}"
            )

    def _has_key_for_model(self, model: str) -> bool:
        """Check whether an API key is available for the given model."""
        provider, _ = _split_provider(model)
        if provider is None:
            # Try to guess from model name
            lowered = model.lower()
            for p in _KNOWN_PROVIDERS:
                if p in lowered:
                    provider = p
                    break
        if provider is None:
            return True  # can't determine; don't skip
        env_var = _PROVIDER_ENV.get(provider) or f"{provider.upper()}_API_KEY"
        return bool(os.environ.get(env_var))

    async def complete(
        self,
        messages: list[dict],
        agent_name: str | None = None,
        tools: list[str] | None = None,
        stream: bool = False,
    ) -> LLMResponse:
        self._check_budget()
        model = self._get_model(agent_name)

        fallback_models = [m for m in _FALLBACK_CHAINS.get(model, []) if self._has_key_for_model(m)]
        models_to_try = [model] + fallback_models

        primary_error: Exception | None = None
        for i, attempt_model in enumerate(models_to_try):
            try:
                self._check_budget()
                return await self._complete_with_retry(
                    attempt_model,
                    messages,
                    agent_name,
                    tools,
                    stream,
                )
            except Exception as e:
                if i == 0:
                    primary_error = e
                log.warning(
                    "llm_fallback",
                    extra={
                        "model": attempt_model,
                        "agent": agent_name,
                        "error": str(e),
                    },
                )
                continue

        log.error(
            "llm_all_failed",
            extra={
                "models": models_to_try,
                "agent": agent_name,
                "error": str(primary_error),
            },
        )
        # Report the primary model's error — the last fallback's error
        # (e.g. gpt-4o-mini auth failure) is misleading when the real
        # problem was with the primary model.
        return LLMResponse(
            content=_format_user_facing_error(model, primary_error),
            model=model,
        )

    async def _complete_with_retry(
        self,
        model: str,
        messages: list[dict],
        agent_name: str | None = None,
        tools: list[str] | None = None,
        stream: bool = False,
    ) -> LLMResponse:
        tool_schemas = self.tool_registry.get_schemas(tools) if tools else None

        # Cache key must include sampling params so different temperature/max_tokens
        # don't collide on the same prompt.
        cache_params = {
            "temperature": self.config.llm.temperature,
            "max_tokens": self.config.llm.max_tokens,
        }
        use_cache = not tools and not stream
        if use_cache:
            cached = self.cache.get(model, messages, cache_params)
            if cached is not None:
                log_event(log, "cache_hit", model=model, agent=agent_name)
                return LLMResponse(content=cached, model=model)

        max_retries = self._get_retries()
        last_error: Exception | None = None

        # Force provider routing when the model name carries a `<provider>/`
        # prefix — fixes preview/unknown model IDs falling through to OpenAI.
        provider, _stripped = _split_provider(model)

        for attempt in range(max_retries + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": self.config.llm.temperature,
                    "max_tokens": self.config.llm.max_tokens,
                    "timeout": self.config.llm.timeout,
                }
                if provider:
                    kwargs["custom_llm_provider"] = provider
                if tool_schemas:
                    kwargs["tools"] = tool_schemas

                start = time.monotonic()
                response = await litellm.acompletion(**kwargs)
                elapsed = time.monotonic() - start

                content = ""
                tool_calls: list[dict] = []
                reasoning_content: str | None = None
                choice = response.choices[0]

                if choice.message.content:
                    content = choice.message.content

                if getattr(choice.message, "reasoning_content", None):
                    reasoning_content = choice.message.reasoning_content

                if choice.message.tool_calls:
                    for tc in choice.message.tool_calls:
                        tool_calls.append(
                            {
                                "id": tc.id,
                                "type": getattr(tc, "type", "function"),
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                        )

                prompt_tokens = response.usage.prompt_tokens if response.usage else 0
                completion_tokens = response.usage.completion_tokens if response.usage else 0
                actual_model = response.model or model

                self._track_usage(
                    agent_name,
                    actual_model,
                    prompt_tokens,
                    completion_tokens,
                )

                log_event(
                    log,
                    "llm_call",
                    model=actual_model,
                    agent=agent_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    duration_ms=round(elapsed * 1000),
                    tool_calls=len(tool_calls),
                )

                if use_cache and content and not tool_calls:
                    self.cache.set(model, messages, content, cache_params)

                return LLMResponse(
                    content=content,
                    tool_calls=tool_calls,
                    model=actual_model,
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    },
                    cost=estimate_cost(
                        actual_model,
                        prompt_tokens,
                        completion_tokens,
                    ),
                    reasoning_content=reasoning_content,
                )

            except Exception as e:
                last_error = e
                if _is_transient_error(e) and attempt < max_retries:
                    delay = min(2**attempt * 1.0, 10.0)
                    log.warning(
                        "llm_retry",
                        extra={
                            "model": model,
                            "agent": agent_name,
                            "attempt": attempt + 1,
                            "delay": delay,
                            "error": str(e),
                        },
                    )
                    await asyncio.sleep(delay)
                else:
                    raise

        raise last_error  # type: ignore[misc]

    async def stream(
        self,
        messages: list[dict],
        agent_name: str | None = None,
    ) -> AsyncIterator[str]:
        self._check_budget()
        model = self._get_model(agent_name)

        provider, _stripped = _split_provider(model)
        try:
            stream_kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": self.config.llm.temperature,
                "max_tokens": self.config.llm.max_tokens,
                "timeout": self.config.llm.timeout,
                "stream": True,
            }
            if provider:
                stream_kwargs["custom_llm_provider"] = provider
            response = await litellm.acompletion(**stream_kwargs)

            full_content = ""
            start = time.monotonic()
            async with asyncio.timeout(self.config.llm.timeout):
                async for chunk in response:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        full_content += delta.content
                        yield delta.content

            elapsed = time.monotonic() - start
            try:
                prompt_tokens = int(litellm.token_counter(model=model, messages=messages))
            except Exception:
                prompt_tokens = max(1, _estimate_tokens(str(messages)))
            try:
                completion_tokens = int(
                    litellm.token_counter(
                        model=model,
                        messages=[{"role": "assistant", "content": full_content}],
                    )
                )
            except Exception:
                completion_tokens = max(1, _estimate_tokens(full_content))
            self._track_usage(agent_name, model, prompt_tokens, completion_tokens)
            log_event(
                log,
                "llm_stream",
                model=model,
                agent=agent_name,
                duration_ms=round(elapsed * 1000),
                chars=len(full_content),
            )

        except TimeoutError:
            yield f"\n[Stream timed out after {self.config.llm.timeout}s]"
        except Exception as e:
            yield f"\n[Stream error: {e}]"

    async def stream_with_tools(
        self,
        messages: list[dict],
        agent_name: str | None = None,
        tools: list[str] | None = None,
        tool_registry: Any = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream completion with tool-call awareness.

        Yields StreamEvent objects:
        - StreamEvent(type="content", text="...") for text chunks
        - StreamEvent(type="tool_call", tool_call={...}) when a tool call is complete
        - StreamEvent(type="done", usage={...}) at end of stream
        """
        self._check_budget()
        model = self._get_model(agent_name)
        provider, _stripped = _split_provider(model)

        stream_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self.config.llm.temperature,
            "max_tokens": self.config.llm.max_tokens,
            "timeout": self.config.llm.timeout,
            "stream": True,
        }
        if provider:
            stream_kwargs["custom_llm_provider"] = provider

        if tools and tool_registry:
            stream_kwargs["tools"] = tool_registry.get_schemas(tools)

        full_content = ""
        reasoning_content: str | None = None
        tool_call_acc: dict[int, dict[str, Any]] = {}
        start = time.monotonic()

        try:
            response = await litellm.acompletion(**stream_kwargs)
            async with asyncio.timeout(self.config.llm.timeout):
                async for chunk in response:
                    delta = chunk.choices[0].delta

                    if delta.content:
                        full_content += delta.content
                        yield StreamEvent(type="content", text=delta.content)

                    # Capture reasoning_content (DeepSeek thinking mode etc.)
                    if getattr(delta, "reasoning_content", None):
                        if reasoning_content is None:
                            reasoning_content = ""
                        reasoning_content += delta.reasoning_content
                        yield StreamEvent(type="reasoning", text=delta.reasoning_content)

                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in tool_call_acc:
                                tool_call_acc[idx] = {
                                    "id": tc_delta.id or "",
                                    "function": {"name": "", "arguments": ""},
                                }
                            acc = tool_call_acc[idx]
                            if tc_delta.id:
                                acc["id"] = tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    acc["function"]["name"] += tc_delta.function.name
                                if tc_delta.function.arguments:
                                    acc["function"]["arguments"] += tc_delta.function.arguments
        except Exception as e:
            yield StreamEvent(type="error", text=_format_user_facing_error(model, e))
            return

        # Emit fully accumulated tool calls after all streaming chunks are processed.
        # Must NOT emit mid-stream: tool call arguments arrive across multiple chunks
        # (id/name in the first, arguments incrementally after).
        for idx in sorted(tool_call_acc.keys()):
            tc = tool_call_acc[idx]
            if tc["id"] and tc["function"]["name"]:
                yield StreamEvent(
                    type="tool_call",
                    tool_call={
                        "id": tc["id"],
                        "type": "function",
                        "function": tc["function"],
                    },
                )

        elapsed = time.monotonic() - start
        prompt_tokens = 0
        completion_tokens = 0
        try:
            prompt_tokens = int(litellm.token_counter(model=model, messages=messages))
            completion_tokens = int(
                litellm.token_counter(
                    model=model,
                    messages=[{"role": "assistant", "content": full_content}],
                )
            )
        except Exception:
            prompt_tokens = max(1, len(str(messages)) // 4)
            completion_tokens = max(1, len(full_content) // 4)
        self._track_usage(agent_name, model, prompt_tokens, completion_tokens)
        log_event(
            log,
            "llm_stream",
            model=model,
            agent=agent_name,
            duration_ms=round(elapsed * 1000),
            chars=len(full_content),
        )
        yield StreamEvent(
            type="done",
            usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
            reasoning_content=reasoning_content,
        )

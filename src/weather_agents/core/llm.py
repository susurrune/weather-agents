"""LLM abstraction layer using LiteLLM with retry, fallback, cost tracking, and budget control."""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

from weather_agents.core.cache import LLMCache
from weather_agents.core.config import AppConfig
from weather_agents.core.logger import get_logger, log_event
from weather_agents.core.tool import ToolRegistry

log = get_logger("llm")

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


def _lazy_litellm():
    """Lazy import of litellm — avoids ~1.5s import overhead for ``wa --help``."""
    import litellm as _lm

    # Silence LiteLLM's noisy stderr banners
    if os.environ.get("WA_DEBUG") != "1":
        _lm.suppress_debug_info = True
        with contextlib.suppress(Exception):
            _lm.set_verbose = False  # type: ignore[attr-defined]
        import logging as _logging

        for _name in ("LiteLLM", "litellm", "litellm.router", "litellm.proxy"):
            _logging.getLogger(_name).setLevel(_logging.ERROR)
    return _lm


# Sentinel — actual import happens on first access.
_lm: Any = None


def _get_litellm():
    global _lm
    if _lm is None:
        _lm = _lazy_litellm()
    return _lm


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


def _is_anthropic_model(model: str) -> bool:
    """True when the model targets Anthropic's API.

    Detection covers both LiteLLM's `anthropic/` prefix and bare model
    names like `claude-3-5-sonnet`. The prompt-cache markers we attach
    are silently ignored by other providers but emitted via LiteLLM as
    extra fields, so we only apply them when the target actually
    understands them.
    """
    lowered = model.lower()
    if lowered.startswith("anthropic/") or lowered.startswith("claude"):
        return True
    provider, _ = _split_provider(model)
    return provider == "anthropic"


def _apply_anthropic_cache_control(
    model: str,
    messages: list[dict],
    tool_schemas: list[dict] | None,
) -> tuple[list[dict], list[dict] | None]:
    """Return (messages, tools) with Anthropic ephemeral cache markers.

    Anthropic charges full input tokens for repeated identical prefixes
    (system prompt + tool schemas, which we ship every round). Adding
    `cache_control: {"type": "ephemeral"}` to those blocks tells the API
    to reuse a 5-minute KV cache, dropping input cost by ~80% on round
    2+ of a turn. The markers are no-ops on non-Anthropic providers but
    we skip them anyway to keep payloads clean.

    The transformation is non-destructive: returns new lists, doesn't
    mutate the originals.
    """
    if not _is_anthropic_model(model):
        return messages, tool_schemas

    # System message: convert plain string to list-of-blocks form with
    # cache_control on the text block. If it's already structured (some
    # caller pre-built blocks), attach cache_control to the last block.
    new_messages: list[dict] = []
    cached_system = False
    for m in messages:
        if not cached_system and m.get("role") == "system":
            content = m.get("content")
            if isinstance(content, str) and content:
                new_messages.append(
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": content,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    }
                )
                cached_system = True
                continue
            if isinstance(content, list) and content:
                new_blocks = [dict(b) for b in content]
                new_blocks[-1] = {
                    **new_blocks[-1],
                    "cache_control": {"type": "ephemeral"},
                }
                new_messages.append({**m, "content": new_blocks})
                cached_system = True
                continue
        new_messages.append(m)

    # Tools: Anthropic accepts cache_control on a single tool block. Mark
    # the LAST tool — caching extends to all earlier ones too.
    new_tools: list[dict] | None = None
    if tool_schemas:
        new_tools = [dict(t) for t in tool_schemas]
        new_tools[-1] = {
            **new_tools[-1],
            "cache_control": {"type": "ephemeral"},
        }
    return new_messages, new_tools


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
    # True when _llm_loop ran out of iterations before the LLM produced a
    # tool-call-free answer. Callers should treat the content as incomplete.
    truncated: bool = False


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
        overrides: dict | None = None,
    ) -> LLMResponse:
        self._check_budget()
        ov = overrides or {}
        raw_model = ov.get("model")
        model: str = raw_model if isinstance(raw_model, str) else self._get_model(agent_name)

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
                    overrides=overrides,
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
        overrides: dict | None = None,
    ) -> LLMResponse:
        tool_schemas = self.tool_registry.get_schemas(tools) if tools else None

        # Apply skill config overrides
        ov = overrides or {}
        temperature = ov.get("temperature", self.config.llm.temperature)
        max_tokens = ov.get("max_tokens", self.config.llm.max_tokens)

        # Cache key must include sampling params so different temperature/max_tokens
        # don't collide on the same prompt.
        cache_params = {
            "temperature": temperature,
            "max_tokens": max_tokens,
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
                # Anthropic prompt cache: tag system + tools with
                # cache_control on every call. No-op for non-Anthropic.
                cached_msgs, cached_tools = _apply_anthropic_cache_control(
                    model, messages, tool_schemas
                )
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": cached_msgs,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "timeout": self.config.llm.timeout,
                }
                if provider:
                    kwargs["custom_llm_provider"] = provider
                if cached_tools:
                    kwargs["tools"] = cached_tools

                start = time.monotonic()
                response = await _get_litellm().acompletion(**kwargs)
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
            response = await _get_litellm().acompletion(**stream_kwargs)

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
                prompt_tokens = int(_get_litellm().token_counter(model=model, messages=messages))
            except Exception:
                prompt_tokens = max(1, _estimate_tokens(str(messages)))
            try:
                completion_tokens = int(
                    _get_litellm().token_counter(
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
        overrides: dict | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream completion with tool-call awareness.

        Yields StreamEvent objects:
        - StreamEvent(type="content", text="...") for text chunks
        - StreamEvent(type="tool_call", tool_call={...}) when a tool call is complete
        - StreamEvent(type="done", usage={...}) at end of stream

        Fallback chains apply only BEFORE the first chunk is yielded — once any
        content/tool_call has streamed out we cannot switch models without
        corrupting the response, so any later error becomes a terminal "error"
        event for the caller to handle.

        If overrides are provided, they take precedence over agent config:
        overrides = {"model": "claude-opus-4-7", "temperature": 0.3, "max_tokens": 32000}
        """
        self._check_budget()
        ov = overrides or {}
        raw_model = ov.get("model")
        primary_model: str = (
            raw_model if isinstance(raw_model, str) else self._get_model(agent_name)
        )
        fallback_models = [
            m for m in _FALLBACK_CHAINS.get(primary_model, []) if self._has_key_for_model(m)
        ]
        models_to_try = [primary_model] + fallback_models

        # Phase A: try to establish the stream against each model in turn. Only
        # acompletion() failures (auth, 5xx, connection refused, etc.) trigger
        # fallback; once we have an iterator we commit to that model.
        response = None
        used_model = primary_model
        provider = None
        primary_error: Exception | None = None
        for i, attempt_model in enumerate(models_to_try):
            ap, _ = _split_provider(attempt_model)
            raw_tool_schemas = (
                tool_registry.get_schemas(tools) if (tools and tool_registry) else None
            )
            # Anthropic cache markers — silently no-op for other providers
            cached_msgs, cached_tools = _apply_anthropic_cache_control(
                attempt_model, messages, raw_tool_schemas
            )
            stream_kwargs: dict[str, Any] = {
                "model": attempt_model,
                "messages": cached_msgs,
                "temperature": ov.get("temperature", self.config.llm.temperature),
                "max_tokens": ov.get("max_tokens", self.config.llm.max_tokens),
                "timeout": self.config.llm.timeout,
                "stream": True,
            }
            if ap:
                stream_kwargs["custom_llm_provider"] = ap
            if cached_tools:
                stream_kwargs["tools"] = cached_tools
            try:
                response = await _get_litellm().acompletion(**stream_kwargs)
                used_model = attempt_model
                provider = ap
                break
            except Exception as e:
                if i == 0:
                    primary_error = e
                log.warning(
                    "stream_fallback",
                    extra={"model": attempt_model, "agent": agent_name, "error": str(e)},
                )
                continue

        if response is None:
            # Every model failed before streaming even started. Report the
            # primary model's error (the chain's last error is usually less
            # actionable, e.g. a fallback's missing key).
            yield StreamEvent(
                type="error", text=_format_user_facing_error(primary_model, primary_error)
            )
            return

        model = used_model
        _ = provider  # already baked into stream_kwargs

        full_content = ""
        reasoning_content: str | None = None
        tool_call_acc: dict[int, dict[str, Any]] = {}
        start = time.monotonic()

        try:
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
            prompt_tokens = int(_get_litellm().token_counter(model=model, messages=messages))
            completion_tokens = int(
                _get_litellm().token_counter(
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

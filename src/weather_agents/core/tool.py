"""Tool registration and execution framework with retry support."""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from weather_agents.core.middleware import MiddlewareChain

from weather_agents.core.circuit_breaker import get_breaker
from weather_agents.core.logger import get_logger

_log = get_logger("tool")

_CACHE_MAXSIZE = 128


def _make_cache_key(kwargs: dict) -> str:
    """Build a deterministic cache key from tool kwargs.

    Uses json.dumps(sort_keys=True) so dict/list values are encoded
    consistently across calls. ``default=str`` falls back to ``repr`` for
    non-JSON-serializable objects so unusual argument shapes still produce
    a stable key instead of raising TypeError on cache lookup.
    """
    import json as _json

    try:
        return _json.dumps(kwargs, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return repr(sorted(kwargs.items()))


# Process-wide tool result cache. Survives tool unregister/reregister
# (skill activation cycles, MCP reconnects) — the previous per-Tool-instance
# cache lost its contents every time a tool was re-added, which is common
# during skill toggling. Bounded by _CACHE_MAXSIZE entries per tool name.
class _ToolResultStore:
    def __init__(self) -> None:
        self._store: dict[str, OrderedDict[str, str]] = {}

    def get(self, tool_name: str, key: str) -> str | None:
        bucket = self._store.get(tool_name)
        if bucket is None:
            return None
        value = bucket.get(key)
        if value is not None:
            bucket.move_to_end(key)
        return value

    def set(self, tool_name: str, key: str, value: str) -> None:
        bucket = self._store.setdefault(tool_name, OrderedDict())
        bucket[key] = value
        if len(bucket) > _CACHE_MAXSIZE:
            bucket.popitem(last=False)

    def clear(self, tool_name: str | None = None) -> None:
        if tool_name is None:
            self._store.clear()
        else:
            self._store.pop(tool_name, None)


_RESULT_STORE = _ToolResultStore()


# Lenient JSON-schema-like type check. Values from LiteLLM tool calls often
# arrive as strings even for declared "number" / "boolean" — we accept those
# common coercion cases instead of rejecting precise but inconvenient inputs.
_TYPE_CHECKS: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
}


def _value_matches_schema_type(value: Any, schema_type: str) -> bool:
    expected = _TYPE_CHECKS.get(schema_type)
    if expected is None:
        return True  # unknown declared type — don't reject
    if isinstance(value, expected):
        # bool is a subclass of int — reject it when integer/number was
        # declared so the schema check stays type-strict.
        return not (schema_type in ("integer", "number") and isinstance(value, bool))
    # Lenient: accept str representations of numbers/booleans the LLM often
    # produces when serializing tool args.
    if schema_type in ("number", "integer") and isinstance(value, str):
        try:
            float(value) if schema_type == "number" else int(value)
            return True
        except ValueError:
            return False
    if schema_type == "boolean" and isinstance(value, str):
        return value.lower() in ("true", "false")
    return False


@dataclass
class ToolParameter:
    name: str
    type: str  # "string", "number", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Any = None


@dataclass
class Tool:
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    handler: Callable[..., Coroutine[Any, Any, str]] | None = None
    max_retries: int = 2
    retry_delay: float = 0.5
    dangerous: bool = False  # high-risk tools need audit + approval
    cacheable: bool = True  # read-only tools can cache results across calls

    def __post_init__(self) -> None:
        # Tool fields are immutable after construction — the schema is too,
        # so we build it once and reuse it across every LLM turn instead of
        # rebuilding O(n_params) dicts for every tool on every call.
        # The result cache lives in the process-wide _RESULT_STORE so
        # cached values survive tool re-registration (skill toggle, MCP
        # reconnect) — the previous per-instance OrderedDict was wiped
        # every time a tool was re-added.
        self._schema: dict | None = None

    def to_function_schema(self) -> dict:
        """Convert to OpenAI function calling schema (cached after first build)."""
        if self._schema is not None:
            return self._schema
        properties: dict[str, Any] = {}
        required: list[str] = []

        for p in self.parameters:
            properties[p.name] = {
                "type": p.type,
                "description": p.description,
            }
            if p.default is not None:
                properties[p.name]["default"] = p.default
            if p.required:
                required.append(p.name)

        self._schema = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
        return self._schema

    async def execute(self, *, agent_name: str | None = None, **kwargs) -> str:
        if self.handler is None:
            return f"Tool '{self.name}' has no handler implemented."

        # Lightweight schema pre-check: required-field presence and primitive
        # type sanity. Without this, a missing/wrong-typed arg costs a full
        # handler invocation + TypeError round-trip back to the LLM (~1-3s
        # of latency per retry). Returning a structured error WITH the correct
        # signature gives the model enough to self-correct on the next turn.
        if self.parameters:
            sig_hint = ", ".join(
                f"{p.name}:{p.type}" + ("" if p.required else "?") for p in self.parameters
            )
            for p in self.parameters:
                if p.required and p.name not in kwargs:
                    return (
                        f"Error: tool '{self.name}' missing required argument "
                        f"'{p.name}'. Expected signature: ({sig_hint})"
                    )
            for p in self.parameters:
                if p.name not in kwargs:
                    continue
                v = kwargs[p.name]
                if not _value_matches_schema_type(v, p.type):
                    return (
                        f"Error: tool '{self.name}' argument '{p.name}' has wrong "
                        f"type (got {type(v).__name__}, expected {p.type}). "
                        f"Expected signature: ({sig_hint})"
                    )

        # Middleware pre-hooks (ACL, rate limit, etc.) — fail-fast policy deny
        from weather_agents.core.middleware import get_middleware_chain

        chain = get_middleware_chain()
        if chain is not None:
            allowed, reason = await chain.run_pre(self.name, agent_name, kwargs)
            if not allowed:
                return f"Error: {reason}"

        # Circuit breaker check — fail-fast if the breaker is OPEN.
        # The "[CircuitBreakerOpen]" prefix is a contract with the agent
        # layer: chat_stream detects it and removes the offending tool from
        # the active tool set for the remainder of the turn so the LLM does
        # not waste iterations re-calling a known-broken tool.
        breaker = get_breaker(self.name)
        if not breaker.allow_request():
            _log.warning(
                "circuit_open",
                extra={"tool": self.name, "state": str(breaker.state)},
            )
            return (
                f"Error: [CircuitBreakerOpen] Tool '{self.name}' is temporarily "
                f"unavailable (breaker {breaker.state.value}). Auto-retry after "
                f"cooldown."
            )

        should_cache = self.cacheable and not self.dangerous
        # Result cache hit (read-only tools only — avoids repeated I/O).
        # The store is process-wide, keyed by (tool_name, kwargs), so
        # entries survive tool re-registration (skill toggle, MCP reconnect).
        if should_cache:
            key = _make_cache_key(kwargs)
            cached = _RESULT_STORE.get(self.name, key)
            if cached is not None:
                return cached

        last_error = ""
        start = time.monotonic()
        for attempt in range(self.max_retries + 1):
            try:
                result = await self.handler(**kwargs)
                if should_cache:
                    key = _make_cache_key(kwargs)
                    _RESULT_STORE.set(self.name, key, result)
                breaker.record_success()
                await self._run_post_hooks(
                    chain, self.name, agent_name, kwargs, result, True, start
                )
                return result
            except TypeError as e:
                _log.warning(
                    "tool_bad_args",
                    extra={"tool": self.name, "error": str(e), "kwargs": list(kwargs)},
                )
                breaker.record_failure()
                msg = f"Error: tool '{self.name}' called with invalid arguments: {e}"
                await self._run_post_hooks(chain, self.name, agent_name, kwargs, msg, False, start)
                return msg
            except Exception as e:
                last_error = str(e)
                breaker.record_failure()
                if attempt < self.max_retries:
                    _log.warning(
                        "tool_retry",
                        extra={
                            "tool": self.name,
                            "attempt": attempt + 1,
                            "error": last_error,
                        },
                    )
                    await asyncio.sleep(self.retry_delay * (2**attempt))

        _log.error(
            "tool_failed",
            extra={"tool": self.name, "retries": self.max_retries, "error": last_error},
        )
        msg = f"Error executing tool '{self.name}' after {self.max_retries} retries: {last_error}"
        await self._run_post_hooks(chain, self.name, agent_name, kwargs, msg, False, start)
        return msg

    @staticmethod
    async def _run_post_hooks(
        chain: MiddlewareChain | None,
        tool_name: str,
        agent_name: str | None,
        kwargs: dict[str, Any],
        result: str,
        success: bool,
        start: float,
    ) -> None:
        if chain is not None:
            duration_ms = (time.monotonic() - start) * 1000
            await chain.run_post(tool_name, agent_name, kwargs, result, success, duration_ms)


class ToolRegistry:
    """Central registry for all tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        """Remove a tool by name. Returns True if it was registered."""
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def get_tools(self, names: list[str] | None = None) -> list[Tool]:
        if names is None:
            return list(self._tools.values())
        return [self._tools[n] for n in names if n in self._tools]

    def get_schemas(self, names: list[str] | None = None) -> list[dict]:
        return [t.to_function_schema() for t in self.get_tools(names)]

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    def merge(self, other: ToolRegistry) -> None:
        """Merge another registry into this one."""
        self._tools.update(other._tools)


# Global tool registry
global_registry = ToolRegistry()

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
    """Build a deterministic cache key from tool kwargs."""
    return str(sorted(kwargs.items()))


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
        self._result_cache: OrderedDict[str, str] = OrderedDict()

    def to_function_schema(self) -> dict:
        """Convert to OpenAI function calling schema."""
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

        return {
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

    async def execute(self, *, agent_name: str | None = None, **kwargs) -> str:
        if self.handler is None:
            return f"Tool '{self.name}' has no handler implemented."

        # Middleware pre-hooks (ACL, rate limit, etc.) — fail-fast policy deny
        from weather_agents.core.middleware import get_middleware_chain

        chain = get_middleware_chain()
        if chain is not None:
            allowed, reason = await chain.run_pre(self.name, agent_name, kwargs)
            if not allowed:
                return f"Error: {reason}"

        # Circuit breaker check — fail-fast if the breaker is OPEN
        breaker = get_breaker(self.name)
        if not breaker.allow_request():
            _log.warning(
                "circuit_open",
                extra={"tool": self.name, "state": str(breaker.state)},
            )
            return (
                f"Tool '{self.name}' is temporarily unavailable "
                f"(circuit breaker {breaker.state.value}). "
                f"Auto-retry after cooldown."
            )

        should_cache = self.cacheable and not self.dangerous
        # Result cache hit (read-only tools only — avoids repeated I/O)
        if should_cache:
            key = _make_cache_key(kwargs)
            cached = self._result_cache.get(key)
            if cached is not None:
                self._result_cache.move_to_end(key)
                return cached

        last_error = ""
        start = time.monotonic()
        for attempt in range(self.max_retries + 1):
            try:
                result = await self.handler(**kwargs)
                if should_cache:
                    key = _make_cache_key(kwargs)
                    self._result_cache[key] = result
                    if len(self._result_cache) > _CACHE_MAXSIZE:
                        self._result_cache.popitem(last=False)
                breaker.record_success()
                await self._run_post_hooks(chain, self.name, agent_name, kwargs, result, True, start)
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

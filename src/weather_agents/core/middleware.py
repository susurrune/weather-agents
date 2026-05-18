"""Middleware/interceptor chain for tool execution.

Provides ACL (access control), rate limiting, and audit logging
as composable hooks around Tool.execute().
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Protocol

from weather_agents.core.bus import Event, EventType, MessageBus
from weather_agents.core.logger import get_logger

_log = get_logger("middleware")


class Middleware(Protocol):
    """Protocol for middleware that wraps tool execution."""

    async def pre(
        self, tool_name: str, agent_name: str | None, kwargs: dict[str, Any]
    ) -> tuple[bool, str | None]:
        """Called before tool execution. Return (True, None) to allow, or (False, reason) to deny."""
        ...

    async def post(
        self,
        tool_name: str,
        agent_name: str | None,
        kwargs: dict[str, Any],
        result: str,
        success: bool,
        duration_ms: float,
    ) -> None:
        """Called after tool execution (success or failure)."""
        ...


@dataclass
class ACLRule:
    """Per-agent ACL rule."""
    allowed_tools: set[str] = field(default_factory=set)
    denied_tools: set[str] = field(default_factory=set)


class ACLMiddleware:
    """Access-control list middleware.

    Controls which agents can call which tools. When ``allow_by_default`` is
    True (default), only explicitly denied tools are blocked; when False,
    only explicitly allowed tools are permitted.
    """

    def __init__(self, allow_by_default: bool = True) -> None:
        self.allow_by_default = allow_by_default
        self._rules: dict[str, ACLRule] = {}

    def allow(self, agent_name: str, *tool_names: str) -> None:
        """Explicitly allow *agent_name* to call *tool_names*."""
        rule = self._rules.setdefault(agent_name, ACLRule())
        rule.allowed_tools.update(tool_names)
        rule.denied_tools.difference_update(tool_names)

    def deny(self, agent_name: str, *tool_names: str) -> None:
        """Explicitly deny *agent_name* from calling *tool_names*."""
        rule = self._rules.setdefault(agent_name, ACLRule())
        rule.denied_tools.update(tool_names)
        rule.allowed_tools.difference_update(tool_names)

    def remove_rules(self, agent_name: str) -> None:
        """Remove all ACL rules for *agent_name*."""
        self._rules.pop(agent_name, None)

    async def pre(
        self, tool_name: str, agent_name: str | None, kwargs: dict[str, Any]
    ) -> tuple[bool, str | None]:
        if agent_name is None:
            return (True, None) if self.allow_by_default else (False, "agent_name is required")

        rule = self._rules.get(agent_name)
        if rule is None:
            return (True, None) if self.allow_by_default else (False, f"agent '{agent_name}' has no ACL rules")

        if tool_name in rule.denied_tools:
            return (False, f"agent '{agent_name}' is not allowed to call '{tool_name}'")

        if not self.allow_by_default and tool_name not in rule.allowed_tools:
            return (False, f"agent '{agent_name}' is not allowed to call '{tool_name}'")

        return (True, None)

    async def post(
        self,
        tool_name: str,
        agent_name: str | None,
        kwargs: dict[str, Any],
        result: str,
        success: bool,
        duration_ms: float,
    ) -> None:
        pass


class RateLimitMiddleware:
    """Sliding-window rate limiter per tool.

    Limits the number of calls to each tool within a rolling time window.
    Per-tool overrides can be set via ``set_limit()``.
    """

    def __init__(self, max_calls: int = 30, window_seconds: float = 60.0) -> None:
        self.default_max_calls = max_calls
        self.default_window = window_seconds
        self._calls: dict[str, list[float]] = defaultdict(list)
        self._overrides: dict[str, tuple[int, float]] = {}

    def set_limit(self, tool_name: str, max_calls: int, window_seconds: float) -> None:
        """Set a per-tool rate limit override."""
        self._overrides[tool_name] = (max_calls, window_seconds)

    def clear(self) -> None:
        """Clear all recorded calls and overrides."""
        self._calls.clear()
        self._overrides.clear()

    async def pre(
        self, tool_name: str, agent_name: str | None, kwargs: dict[str, Any]
    ) -> tuple[bool, str | None]:
        max_calls, window = self._overrides.get(
            tool_name, (self.default_max_calls, self.default_window)
        )

        now = time.monotonic()
        cutoff = now - window

        calls = [t for t in self._calls[tool_name] if t > cutoff]
        if calls:
            self._calls[tool_name] = calls
        else:
            self._calls.pop(tool_name, None)  # prevent unbounded key growth

        if len(calls) >= max_calls:
            remaining = int(calls[0] + window - now) if calls else 0
            return (
                False,
                f"rate limit exceeded for '{tool_name}': {max_calls} calls per {window}s window (retry in ~{remaining}s)",
            )

        self._calls[tool_name].append(now)
        return (True, None)

    async def post(
        self,
        tool_name: str,
        agent_name: str | None,
        kwargs: dict[str, Any],
        result: str,
        success: bool,
        duration_ms: float,
    ) -> None:
        pass


class AuditMiddleware:
    """Audit-logging middleware.

    Publishes tool-call events to the message bus with timing,
    agent identity, success/failure, and truncated result preview.
    Not intended for third-party code — meant for internal observability.
    """

    def __init__(self, bus: MessageBus | None = None) -> None:
        self._bus = bus

    async def pre(
        self, tool_name: str, agent_name: str | None, kwargs: dict[str, Any]
    ) -> tuple[bool, str | None]:
        return (True, None)

    async def post(
        self,
        tool_name: str,
        agent_name: str | None,
        kwargs: dict[str, Any],
        result: str,
        success: bool,
        duration_ms: float,
    ) -> None:
        if self._bus is None:
            return

        safe_args = {
            k: (str(v)[:200] if not isinstance(v, (int, float, bool)) else v)
            for k, v in kwargs.items()
        }
        safe_result = (result or "")[:500]

        try:
            self._bus.add_event(
                Event(
                    type=EventType.TOOL_CALL,
                    source=agent_name or "unknown",
                    data={
                        "tool": tool_name,
                        "args": safe_args,
                        "success": success,
                        "duration_ms": round(duration_ms, 1),
                        "result_preview": safe_result,
                    },
                )
            )
        except Exception as exc:
            _log.warning("audit_log_failed", extra={"tool": tool_name, "error": str(exc)})


@dataclass
class MiddlewareChain:
    """Chain of middleware hooks applied around tool execution.

    Pre-hooks run in registration order; if any returns ``(False, reason)``
    the chain short-circuits and the tool is denied.
    Post-hooks always run on success *or* failure.
    """

    _middleware: list[Middleware] = field(default_factory=list)

    def add(self, middleware: Middleware) -> None:
        """Register a middleware instance. Pre-hooks run in add() order."""
        self._middleware.append(middleware)

    async def run_pre(
        self, tool_name: str, agent_name: str | None, kwargs: dict[str, Any]
    ) -> tuple[bool, str | None]:
        for mw in self._middleware:
            allowed, reason = await mw.pre(tool_name, agent_name, kwargs)
            if not allowed:
                return (False, reason)
        return (True, None)

    async def run_post(
        self,
        tool_name: str,
        agent_name: str | None,
        kwargs: dict[str, Any],
        result: str,
        success: bool,
        duration_ms: float,
    ) -> None:
        for mw in self._middleware:
            try:
                await mw.post(tool_name, agent_name, kwargs, result, success, duration_ms)
            except Exception as exc:
                _log.warning(
                    "middleware_post_failed",
                    extra={"middleware": type(mw).__name__, "tool": tool_name, "error": str(exc)},
                )


# ── Global active chain ────────────────────────────────────────────────────

_global_chain: MiddlewareChain | None = None


def set_middleware_chain(chain: MiddlewareChain | None) -> None:
    """Set the global middleware chain used by Tool.execute()."""
    global _global_chain
    _global_chain = chain


def get_middleware_chain() -> MiddlewareChain | None:
    """Return the global middleware chain, or None."""
    return _global_chain

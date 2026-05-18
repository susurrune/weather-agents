"""Tests for middleware/interceptor layer."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from weather_agents.core.bus import Event, MessageBus
from weather_agents.core.middleware import (
    ACLMiddleware,
    AuditMiddleware,
    MiddlewareChain,
    RateLimitMiddleware,
    get_middleware_chain,
    set_middleware_chain,
)


class TestACLMiddleware:
    @pytest.mark.asyncio
    async def test_allow_by_default_allows_unknown_agent(self):
        acl = ACLMiddleware(allow_by_default=True)
        allowed, reason = await acl.pre("read_file", "fog", {})
        assert allowed is True
        assert reason is None

    @pytest.mark.asyncio
    async def test_deny_by_default_blocks_unknown_agent(self):
        acl = ACLMiddleware(allow_by_default=False)
        allowed, reason = await acl.pre("read_file", "fog", {})
        assert allowed is False
        assert "fog" in (reason or "")

    @pytest.mark.asyncio
    async def test_explicit_deny(self):
        acl = ACLMiddleware(allow_by_default=True)
        acl.deny("fog", "shell_exec")
        allowed, reason = await acl.pre("shell_exec", "fog", {})
        assert allowed is False
        assert "shell_exec" in (reason or "")

    @pytest.mark.asyncio
    async def test_explicit_deny_does_not_affect_other_tools(self):
        acl = ACLMiddleware(allow_by_default=True)
        acl.deny("fog", "shell_exec")
        allowed, _ = await acl.pre("read_file", "fog", {})
        assert allowed is True

    @pytest.mark.asyncio
    async def test_explicit_allow_overrides_deny(self):
        acl = ACLMiddleware(allow_by_default=True)
        acl.deny("fog", "shell_exec")
        acl.allow("fog", "shell_exec")
        allowed, _ = await acl.pre("shell_exec", "fog", {})
        assert allowed is True

    @pytest.mark.asyncio
    async def test_allow_by_default_false_with_explicit_allow(self):
        acl = ACLMiddleware(allow_by_default=False)
        acl.allow("fog", "read_file")
        allowed, _ = await acl.pre("read_file", "fog", {})
        assert allowed is True
        allowed2, _ = await acl.pre("shell_exec", "fog", {})
        assert allowed2 is False

    @pytest.mark.asyncio
    async def test_deny_removes_from_allowed(self):
        acl = ACLMiddleware(allow_by_default=False)
        acl.allow("fog", "shell_exec")
        acl.deny("fog", "shell_exec")
        allowed, _ = await acl.pre("shell_exec", "fog", {})
        assert allowed is False

    @pytest.mark.asyncio
    async def test_none_agent_with_allow_by_default(self):
        acl = ACLMiddleware(allow_by_default=True)
        allowed, _ = await acl.pre("read_file", None, {})
        assert allowed is True

    @pytest.mark.asyncio
    async def test_none_agent_with_deny_by_default(self):
        acl = ACLMiddleware(allow_by_default=False)
        allowed, reason = await acl.pre("read_file", None, {})
        assert allowed is False
        assert reason is not None

    @pytest.mark.asyncio
    async def test_remove_rules(self):
        acl = ACLMiddleware(allow_by_default=False)
        acl.allow("fog", "read_file")
        acl.remove_rules("fog")
        allowed, _ = await acl.pre("read_file", "fog", {})
        assert allowed is False

    @pytest.mark.asyncio
    async def test_post_is_noop(self):
        acl = ACLMiddleware()
        result = await acl.post("read_file", "fog", {}, "ok", True, 1.0)
        assert result is None


class TestRateLimitMiddleware:
    @pytest.mark.asyncio
    async def test_allows_within_limit(self):
        rl = RateLimitMiddleware(max_calls=5, window_seconds=60.0)
        for _ in range(5):
            allowed, _ = await rl.pre("read_file", "fog", {})
            assert allowed is True

    @pytest.mark.asyncio
    async def test_blocks_over_limit(self):
        rl = RateLimitMiddleware(max_calls=3, window_seconds=60.0)
        for _ in range(3):
            await rl.pre("read_file", "fog", {})
        allowed, reason = await rl.pre("read_file", "fog", {})
        assert allowed is False
        assert "rate limit" in (reason or "")

    @pytest.mark.asyncio
    async def test_different_tools_have_independent_limits(self):
        rl = RateLimitMiddleware(max_calls=2, window_seconds=60.0)
        await rl.pre("read_file", "fog", {})
        await rl.pre("read_file", "fog", {})
        allowed, _ = await rl.pre("shell_exec", "fog", {})
        assert allowed is True

    @pytest.mark.asyncio
    async def test_per_tool_override(self):
        rl = RateLimitMiddleware(max_calls=10, window_seconds=60.0)
        rl.set_limit("shell_exec", max_calls=1, window_seconds=60.0)
        allowed, _ = await rl.pre("shell_exec", "fog", {})
        assert allowed is True
        allowed, reason = await rl.pre("shell_exec", "fog", {})
        assert allowed is False
        assert "shell_exec" in (reason or "")

    @pytest.mark.asyncio
    async def test_clear_resets(self):
        rl = RateLimitMiddleware(max_calls=1, window_seconds=60.0)
        await rl.pre("read_file", "fog", {})
        rl.clear()
        allowed, _ = await rl.pre("read_file", "fog", {})
        assert allowed is True

    @pytest.mark.asyncio
    async def test_post_is_noop(self):
        rl = RateLimitMiddleware()
        result = await rl.post("read_file", "fog", {}, "ok", True, 1.0)
        assert result is None


class TestAuditMiddleware:
    @pytest.mark.asyncio
    async def test_publishes_event_on_success(self):
        bus = MagicMock(spec=MessageBus)
        audit = AuditMiddleware(bus=bus)
        await audit.post("read_file", "fog", {"path": "/tmp"}, "file content", True, 5.0)
        bus.add_event.assert_called_once()
        event: Event = bus.add_event.call_args[0][0]
        assert event.source == "fog"
        assert event.data["tool"] == "read_file"
        assert event.data["success"] is True
        assert event.data["duration_ms"] == 5.0

    @pytest.mark.asyncio
    async def test_publishes_event_on_failure(self):
        bus = MagicMock(spec=MessageBus)
        audit = AuditMiddleware(bus=bus)
        await audit.post("shell_exec", "dew", {"command": "rm"}, "Error: denied", False, 2.0)
        event: Event = bus.add_event.call_args[0][0]
        assert event.data["success"] is False

    @pytest.mark.asyncio
    async def test_no_bus_does_not_raise(self):
        audit = AuditMiddleware(bus=None)
        await audit.post("read_file", "fog", {}, "ok", True, 1.0)

    @pytest.mark.asyncio
    async def test_pre_is_always_allowed(self):
        audit = AuditMiddleware()
        allowed, _ = await audit.pre("read_file", "fog", {})
        assert allowed is True

    @pytest.mark.asyncio
    async def test_truncates_long_args(self):
        bus = MagicMock(spec=MessageBus)
        audit = AuditMiddleware(bus=bus)
        long_str = "x" * 1000
        await audit.post("read_file", "fog", {"path": long_str}, "ok", True, 1.0)
        event: Event = bus.add_event.call_args[0][0]
        assert len(event.data["args"]["path"]) <= 200

    @pytest.mark.asyncio
    async def test_bus_error_logged_not_raised(self):
        bus = MagicMock(spec=MessageBus)
        bus.add_event.side_effect = RuntimeError("bus down")
        audit = AuditMiddleware(bus=bus)
        await audit.post("read_file", "fog", {}, "ok", True, 1.0)


class TestMiddlewareChain:
    @pytest.mark.asyncio
    async def test_empty_chain_allows(self):
        chain = MiddlewareChain()
        allowed, _ = await chain.run_pre("read_file", "fog", {})
        assert allowed is True

    @pytest.mark.asyncio
    async def test_pre_hooks_in_order(self):
        mw1 = AsyncMock()
        mw1.pre.return_value = (True, None)
        mw2 = AsyncMock()
        mw2.pre.return_value = (True, None)

        chain = MiddlewareChain()
        chain.add(mw1)
        chain.add(mw2)
        allowed, _ = await chain.run_pre("read_file", "fog", {})
        assert allowed is True
        mw1.pre.assert_called_once()
        mw2.pre.assert_called_once()

    @pytest.mark.asyncio
    async def test_short_circuits_on_first_deny(self):
        mw1 = AsyncMock()
        mw1.pre.return_value = (False, "denied by mw1")
        mw2 = AsyncMock()

        chain = MiddlewareChain()
        chain.add(mw1)
        chain.add(mw2)
        allowed, reason = await chain.run_pre("read_file", "fog", {})
        assert allowed is False
        assert reason == "denied by mw1"
        mw2.pre.assert_not_called()

    @pytest.mark.asyncio
    async def test_runs_post_hooks_on_all_middleware(self):
        mw1 = AsyncMock()
        mw2 = AsyncMock()

        chain = MiddlewareChain()
        chain.add(mw1)
        chain.add(mw2)
        await chain.run_post("read_file", "fog", {}, "ok", True, 5.0)

        mw1.post.assert_called_once()
        mw2.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_post_hook_error_does_not_block_others(self):
        mw1 = AsyncMock()
        mw1.post.side_effect = RuntimeError("mw1 failed")
        mw2 = AsyncMock()

        chain = MiddlewareChain()
        chain.add(mw1)
        chain.add(mw2)
        await chain.run_post("read_file", "fog", {}, "ok", True, 5.0)

        mw2.post.assert_called_once()


class TestGlobalChain:
    def test_set_and_get(self):
        chain = MiddlewareChain()
        set_middleware_chain(chain)
        assert get_middleware_chain() is chain
        set_middleware_chain(None)

    def test_default_is_none(self):
        set_middleware_chain(None)
        assert get_middleware_chain() is None

    def test_clear_after_set(self):
        chain = MiddlewareChain()
        set_middleware_chain(chain)
        set_middleware_chain(None)
        assert get_middleware_chain() is None


@pytest.mark.asyncio
async def test_tool_execution_with_acl_deny(tool_registry):
    """Verify Tool.execute() rejects calls blocked by ACL middleware."""
    from unittest.mock import AsyncMock

    from weather_agents.core.tool import Tool

    tool = Tool(
        name="acl_test_tool",
        description="ACL test",
        parameters=[],
        handler=AsyncMock(return_value="ok"),
    )
    tool_registry.register(tool)

    acl = ACLMiddleware(allow_by_default=True)
    acl.deny("test_agent", "acl_test_tool")

    chain = MiddlewareChain()
    chain.add(acl)
    set_middleware_chain(chain)

    try:
        result = await tool.execute(agent_name="test_agent")
        assert "denied" in result or "not allowed" in result
    finally:
        set_middleware_chain(None)


@pytest.mark.asyncio
async def test_tool_execution_with_rate_limit(tool_registry):
    """Verify Tool.execute() rejects calls that exceed rate limit."""
    from unittest.mock import AsyncMock

    from weather_agents.core.tool import Tool

    tool = Tool(
        name="ratelimit_test_tool",
        description="Rate limit test",
        parameters=[],
        handler=AsyncMock(return_value="ok"),
    )
    tool_registry.register(tool)

    rl = RateLimitMiddleware(max_calls=1, window_seconds=60.0)
    chain = MiddlewareChain()
    chain.add(rl)
    set_middleware_chain(chain)

    try:
        first = await tool.execute(agent_name="test_agent")
        assert first == "ok"

        second = await tool.execute(agent_name="test_agent")
        assert "rate limit" in second
    finally:
        set_middleware_chain(None)


@pytest.mark.asyncio
async def test_tool_execution_with_audit(tool_registry):
    """Verify Tool.execute() triggers audit post-hooks."""
    from unittest.mock import AsyncMock, MagicMock

    from weather_agents.core.bus import MessageBus
    from weather_agents.core.tool import Tool

    bus = MagicMock(spec=MessageBus)
    tool = Tool(
        name="audit_test_tool",
        description="Audit test",
        parameters=[],
        handler=AsyncMock(return_value="ok"),
    )
    tool_registry.register(tool)

    audit = AuditMiddleware(bus=bus)
    chain = MiddlewareChain()
    chain.add(audit)
    set_middleware_chain(chain)

    try:
        result = await tool.execute(agent_name="auditor")
        assert result == "ok"
        bus.add_event.assert_called_once()
        event: Event = bus.add_event.call_args[0][0]
        assert event.data["tool"] == "audit_test_tool"
        assert event.data["success"] is True
    finally:
        set_middleware_chain(None)


@pytest.mark.asyncio
async def test_tool_execution_full_chain(tool_registry):
    """Verify all three middleware types compose correctly."""
    from unittest.mock import AsyncMock, MagicMock

    from weather_agents.core.bus import MessageBus
    from weather_agents.core.tool import Tool

    bus = MagicMock(spec=MessageBus)
    tool = Tool(
        name="full_chain_tool",
        description="Full chain test",
        parameters=[],
        handler=AsyncMock(return_value="done"),
    )
    tool_registry.register(tool)

    acl = ACLMiddleware(allow_by_default=True)
    acl.deny("bad_agent", "full_chain_tool")

    rl = RateLimitMiddleware(max_calls=10, window_seconds=60.0)
    audit = AuditMiddleware(bus=bus)

    chain = MiddlewareChain()
    chain.add(acl)
    chain.add(rl)
    chain.add(audit)
    set_middleware_chain(chain)

    try:
        result = await tool.execute(agent_name="good_agent")
        assert result == "done"
        bus.add_event.assert_called_once()

        result2 = await tool.execute(agent_name="bad_agent")
        assert "not allowed" in result2
    finally:
        set_middleware_chain(None)


@pytest.mark.asyncio
async def test_tool_execution_no_agent_name_still_works(tool_registry):
    """Verify execute() without agent_name is backward compatible."""
    from unittest.mock import AsyncMock

    from weather_agents.core.tool import Tool

    tool = Tool(
        name="compat_test_tool",
        description="Compat test",
        parameters=[],
        handler=AsyncMock(return_value="ok"),
    )
    tool_registry.register(tool)

    result = await tool.execute(path="/tmp")
    assert result == "ok"


@pytest.mark.asyncio
async def test_concurrent_execution_with_rate_limit(tool_registry):
    """Rate limiter correctly handles concurrent tool executions."""
    import asyncio as _asyncio
    from unittest.mock import AsyncMock
    from weather_agents.core.tool import Tool

    tool = Tool(
        name="concurrent_test_tool",
        description="Concurrent test",
        parameters=[],
        handler=AsyncMock(return_value="ok"),
    )
    tool_registry.register(tool)

    rl = RateLimitMiddleware(max_calls=5, window_seconds=60.0)
    chain = MiddlewareChain()
    chain.add(rl)
    set_middleware_chain(chain)

    try:
        results = await _asyncio.gather(*[
            tool.execute(agent_name=f"agent_{i}") for i in range(10)
        ])
        allowed = sum(1 for r in results if r == "ok")
        denied = sum(1 for r in results if "rate limit" in r)
        assert allowed == 5
        assert denied == 5
    finally:
        set_middleware_chain(None)

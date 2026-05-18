"""Tests for circuit breaker pattern."""

from __future__ import annotations

import time

import pytest

from weather_agents.core.circuit_breaker import CircuitBreaker, CircuitState, get_breaker


class TestCircuitBreaker:
    def test_initial_state_closed(self):
        cb = CircuitBreaker("test")
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request() is True

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.allow_request() is False

    def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker("test", failure_threshold=5)
        for _ in range(4):
            cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request() is True

    def test_records_success_resets_count(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED

    def test_half_open_after_timeout(self):
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0.05)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.allow_request() is False
        time.sleep(0.06)
        assert cb.allow_request() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_success_closes(self):
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0.05)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.06)
        cb.allow_request()  # transitions to HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0.05)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.06)
        cb.allow_request()  # transitions to HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_reset(self):
        cb = CircuitBreaker("test", failure_threshold=1)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb.allow_request() is True

    def test_get_breaker_returns_singleton(self):
        b1 = get_breaker("my_tool")
        b2 = get_breaker("my_tool")
        assert b1 is b2

    def test_get_breaker_different_names(self):
        b1 = get_breaker("tool_a")
        b2 = get_breaker("tool_b")
        assert b1 is not b2

    def test_individual_reset(self):
        cb = CircuitBreaker("reset_test", failure_threshold=1)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_circuit_open_returns_false_on_allow(self):
        cb = CircuitBreaker("test", failure_threshold=1)
        cb.record_failure()
        assert cb.allow_request() is False

    def test_failure_count_tracking(self):
        cb = CircuitBreaker("test", failure_threshold=10)
        for i in range(1, 6):
            cb.record_failure()
            assert cb.failure_count == i

    @pytest.mark.asyncio
    async def test_tool_execution_circuit_breaker(self, tool_registry):
        """Verify Tool.execute() rejects calls when breaker is open.

        Uses a unique tool name so global breaker state is not polluted
        for other tests.
        """
        import uuid
        from unittest.mock import AsyncMock

        from weather_agents.core.tool import Tool

        unique = f"breakable_test_{uuid.uuid4().hex[:6]}"

        tool = Tool(
            name=unique,
            description="breakable",
            parameters=[],
            handler=AsyncMock(return_value="ok"),
        )

        breaker = get_breaker(unique, failure_threshold=1)
        breaker.record_failure()
        assert breaker.allow_request() is False

        result = await tool.execute()
        assert "unavailable" in result or "circuit breaker" in result

    def test_half_open_allows_all_until_decision(self):
        """HALF_OPEN state allows all requests through until success/failure."""
        cb = CircuitBreaker("race_test", failure_threshold=1, recovery_timeout=0.05)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        import time as _time

        _time.sleep(0.06)

        # First call transitions OPEN → HALF_OPEN; all subsequent also allowed
        results = [cb.allow_request() for _ in range(10)]
        assert all(results)
        assert cb.state == CircuitState.HALF_OPEN

        # First failure in HALF_OPEN re-opens
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.allow_request() is False

    @pytest.mark.asyncio
    async def test_concurrent_tool_execution_during_open(self, tool_registry):
        """Concurrent tool executions when breaker is OPEN all get rejected."""
        import asyncio as _asyncio
        import uuid
        from unittest.mock import AsyncMock

        from weather_agents.core.tool import Tool

        unique = f"concurrent_cb_{uuid.uuid4().hex[:6]}"
        tool = Tool(
            name=unique,
            description="concurrent breaker test",
            parameters=[],
            handler=AsyncMock(return_value="ok"),
        )
        tool_registry.register(tool)

        breaker = get_breaker(unique, failure_threshold=1)
        breaker.record_failure()
        assert breaker.allow_request() is False

        results = await _asyncio.gather(*[tool.execute() for _ in range(5)])
        for r in results:
            assert "unavailable" in r or "circuit breaker" in r

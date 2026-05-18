"""Per-tool circuit breaker — fail-fast on cascading errors.

Three states:
  CLOSED    Normal operation — passes all requests.
  OPEN      Failures exceeded threshold — requests rejected immediately.
  HALF_OPEN Cooldown expired — one probe request to test recovery.

Usage::

    breaker = get_breaker("write_file")
    if not breaker.allow_request():
        return "tool temporarily unavailable"
    try:
        result = await handler(**kwargs)
        breaker.record_success()
    except Exception:
        breaker.record_failure()
"""

from __future__ import annotations

import time
from enum import StrEnum


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0

    @property
    def failure_count(self) -> int:
        return self._failure_count

    def allow_request(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        # HALF_OPEN — allow one probe
        return True

    def record_success(self) -> None:
        self._failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

    def reset(self) -> None:
        self.state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0


# Global per-tool circuit breaker registry
_BREAKERS: dict[str, CircuitBreaker] = {}


def get_breaker(
    name: str, *, failure_threshold: int = 3, recovery_timeout: float = 30.0
) -> CircuitBreaker:
    """Get or create a circuit breaker for a tool by name."""
    if name not in _BREAKERS:
        _BREAKERS[name] = CircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )
    return _BREAKERS[name]


def reset_all_breakers() -> None:
    for b in _BREAKERS.values():
        b.reset()


def breaker_states() -> dict[str, str]:
    """Snapshot of all breaker states for display / monitoring."""
    return {
        name: str(b.state.value)
        for name, b in _BREAKERS.items()
        if b._failure_count > 0 or b.state != CircuitState.CLOSED
    }

"""Async event-driven message bus for inter-agent communication."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

log = logging.getLogger("sky.bus")


class EventType(StrEnum):
    TASK_ASSIGNED = "task_assigned"
    TASK_COMPLETED = "task_completed"
    TASK_FEEDBACK = "task_feedback"
    AGENT_REQUEST = "agent_request"
    AGENT_RESPONSE = "agent_response"
    SYSTEM_EVENT = "system_event"
    STATE_CHANGE = "state_change"  # agent state changes
    LLM_CALL = "llm_call"  # LLM request made
    TOOL_CALL = "tool_call"  # tool was called


@dataclass
class Event:
    type: EventType
    source: str  # agent name or "system"
    target: str | None = None  # None = broadcast
    data: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


Handler = Callable[[Event], Coroutine[Any, Any, None]]


class MessageBus:
    """Pub/sub message bus for agent communication."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Handler]] = defaultdict(list)
        self._state_listeners: list[Handler] = []  # handle state changes
        self._history: list[Event] = []
        self._max_history = 2000

    def subscribe(self, agent_name: str, handler: Handler) -> None:
        self._subscribers[agent_name].append(handler)

    def unsubscribe(self, agent_name: str) -> None:
        self._subscribers.pop(agent_name, None)

    def on_state_change(self, handler: Handler) -> None:
        """Register a global handler for state change events."""
        self._state_listeners.append(handler)

    def remove_state_listener(self, handler: Handler) -> None:
        self._state_listeners.remove(handler)

    def _trim_history(self) -> None:
        if len(self._history) > self._max_history:
            del self._history[: len(self._history) - self._max_history]

    def add_event(self, event: Event) -> None:
        """Record event without routing (for state changes / local observation)."""
        self._history.append(event)
        self._trim_history()

    async def notify_state_change(self, event: Event) -> None:
        """Notify state change listeners (must be called from async context)."""
        if event.type != EventType.STATE_CHANGE:
            return
        for handler in self._state_listeners:
            try:
                await handler(event)
            except Exception as e:
                log.exception("state_change handler failed: %s", e)

    async def publish(self, event: Event) -> None:
        self._history.append(event)
        self._trim_history()

        if event.target:
            # Direct message
            handlers = self._subscribers.get(event.target, [])
            for handler in handlers:
                try:
                    await handler(event)
                except Exception as e:
                    log.exception("bus handler failed for target=%s: %s", event.target, e)
        else:
            # Broadcast
            for name, handlers in self._subscribers.items():
                if name == event.source:
                    continue
                for handler in handlers:
                    try:
                        await handler(event)
                    except Exception as e:
                        log.exception("bus handler failed for subscriber=%s: %s", name, e)

    def get_history(
        self,
        agent_name: str | None = None,
        event_type: EventType | None = None,
        limit: int = 50,
    ) -> list[Event]:
        events = self._history
        if agent_name:
            events = [e for e in events if agent_name in (e.source, e.target)]
        if event_type:
            events = [e for e in events if e.type == event_type]
        return events[-limit:]

"""Tests for the event bus."""

from __future__ import annotations

import pytest

from weather_agents.core.bus import Event, EventType


class TestMessageBus:
    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self, bus):
        received = []

        async def handler(event: Event):
            received.append(event)

        bus.subscribe("test_agent", handler)
        event = Event(type=EventType.SYSTEM_EVENT, source="system", data={"msg": "hello"})

        await bus.publish(event)

        assert len(received) == 1
        assert received[0].type == EventType.SYSTEM_EVENT
        assert received[0].data == {"msg": "hello"}

    @pytest.mark.asyncio
    async def test_direct_message(self, bus):
        received = []

        async def handler(event: Event):
            received.append(event)

        bus.subscribe("target_agent", handler)
        event = Event(
            type=EventType.TASK_ASSIGNED,
            source="snow",
            target="target_agent",
            data={"task": "test"},
        )

        await bus.publish(event)

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_unsubscribe(self, bus):
        received = []

        async def handler(event: Event):
            received.append(event)

        bus.subscribe("agent", handler)
        bus.unsubscribe("agent")

        await bus.publish(Event(type=EventType.SYSTEM_EVENT, source="system"))

        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_state_listener(self, bus):
        received = []

        async def listener(event: Event):
            received.append(event)

        bus.on_state_change(listener)
        await bus.notify_state_change(
            Event(
                type=EventType.STATE_CHANGE,
                source="fog",
                data={"old_state": "idle", "new_state": "thinking"},
            )
        )

        assert len(received) == 1
        assert received[0].type == EventType.STATE_CHANGE
        assert received[0].source == "fog"

    @pytest.mark.asyncio
    async def test_history(self, bus):
        for i in range(5):
            await bus.publish(
                Event(
                    type=EventType.SYSTEM_EVENT,
                    source="system",
                    data={"i": i},
                )
            )

        assert len(bus.get_history()) == 5
        assert len(bus.get_history(limit=2)) == 2

    @pytest.mark.asyncio
    async def test_history_filter_by_agent(self, bus):
        await bus.publish(Event(type=EventType.SYSTEM_EVENT, source="fog"))
        await bus.publish(Event(type=EventType.SYSTEM_EVENT, source="rain"))

        fog_events = bus.get_history(agent_name="fog")
        assert len(fog_events) == 1
        assert fog_events[0].source == "fog"

    @pytest.mark.asyncio
    async def test_remove_state_listener(self, bus):
        received = []

        async def listener(event: Event):
            received.append(event)

        bus.on_state_change(listener)
        bus.remove_state_listener(listener)
        await bus.notify_state_change(Event(type=EventType.STATE_CHANGE, source="fog", data={}))

        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_non_state_event_ignored_by_listeners(self, bus):
        received = []

        async def listener(event: Event):
            received.append(event)

        bus.on_state_change(listener)
        await bus.notify_state_change(
            Event(type=EventType.TASK_ASSIGNED, source="snow", data={"task": "x"})
        )
        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_state_listener_exception_does_not_block(self, bus):
        error_log = []

        async def failing_listener(event: Event):
            raise ValueError("listener failed")

        async def ok_listener(event: Event):
            error_log.append(event)

        bus.on_state_change(failing_listener)
        bus.on_state_change(ok_listener)
        await bus.notify_state_change(
            Event(type=EventType.STATE_CHANGE, source="fog", data={"state": "idle"})
        )
        # ok listener should still be called
        assert len(error_log) == 1

    @pytest.mark.asyncio
    async def test_handler_exception_on_direct_message(self, bus):
        good = []

        async def failing(event: Event):
            raise ValueError("boom")

        async def ok(event: Event):
            good.append(event)

        bus.subscribe("target", failing)
        bus.subscribe("target", ok)
        await bus.publish(Event(type=EventType.TASK_ASSIGNED, source="snow", target="target"))
        assert len(good) == 1

    @pytest.mark.asyncio
    async def test_broadcast_skips_source(self, bus):
        called_agents = []

        async def fog_handler(event: Event):
            called_agents.append("fog")

        async def rain_handler(event: Event):
            called_agents.append("rain")

        bus.subscribe("fog", fog_handler)
        bus.subscribe("rain", rain_handler)
        # fog publishes — fog's own handler should not be called
        await bus.publish(Event(type=EventType.SYSTEM_EVENT, source="fog"))
        assert "fog" not in called_agents
        assert "rain" in called_agents

    @pytest.mark.asyncio
    async def test_broadcast_handler_exception_does_not_block(self, bus):
        good = []

        async def failing(event: Event):
            raise ValueError("failed")

        async def ok(event: Event):
            good.append(event)

        bus.subscribe("rain", failing)
        bus.subscribe("dew", ok)
        await bus.publish(Event(type=EventType.SYSTEM_EVENT, source="fog"))
        assert len(good) == 1

    @pytest.mark.asyncio
    async def test_history_trimmed_at_max(self, bus):
        max_history = bus._max_history
        # Add more events than the limit
        for i in range(max_history + 100):
            await bus.publish(Event(type=EventType.SYSTEM_EVENT, source="system", data={"i": i}))
        assert len(bus._history) <= max_history

    def test_history_filter_by_type(self, bus):
        import asyncio

        async def _populate():
            await bus.publish(Event(type=EventType.TASK_ASSIGNED, source="snow"))
            await bus.publish(Event(type=EventType.SYSTEM_EVENT, source="system"))
            await bus.publish(Event(type=EventType.TASK_COMPLETED, source="rain"))

        asyncio.run(_populate())
        task_events = bus.get_history(event_type=EventType.TASK_ASSIGNED)
        assert len(task_events) == 1
        assert task_events[0].type == EventType.TASK_ASSIGNED

    def test_subscribe_and_add_event(self, bus):
        event = Event(type=EventType.SYSTEM_EVENT, source="system", data={"msg": "test"})
        bus.add_event(event)
        assert len(bus._history) == 1
        assert bus._history[0].data["msg"] == "test"

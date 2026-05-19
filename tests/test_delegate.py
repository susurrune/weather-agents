"""Tests for the delegate_to tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from weather_agents.core.agent import AgentState, TaskResult
from weather_agents.core.bus import MessageBus
from weather_agents.tools.delegate import AGENT_SPECIALTIES, create_delegate_tool


def _make_agent(name: str, display_name: str = "", emoji: str = "") -> Mock:
    agent = Mock()
    agent.name = name
    agent.display_name = display_name or name.title()
    agent.emoji = emoji or "T"
    agent.state = AgentState.IDLE
    agent.bus = MessageBus()
    agent.init = AsyncMock()
    agent.execute_task = AsyncMock(return_value=TaskResult(success=True, content="task done"))
    agent._set_state = AsyncMock()
    return agent


@pytest.fixture
def agent_map():
    return {
        "fog": _make_agent("fog", "雾", "~~"),
        "rain": _make_agent("rain", "雨", "//"),
        "frost": _make_agent("frost", "霜", "**"),
        "snow": _make_agent("snow", "雪", ".."),
        "dew": _make_agent("dew", "露", ",,"),
        "fair": _make_agent("fair", "晴", "**"),
    }


class TestCreateDelegateTool:
    def test_creates_tool_with_correct_name(self, agent_map):
        tool = create_delegate_tool(agent_map)
        assert tool.name == "delegate_to"

    def test_tool_has_parameters(self, agent_map):
        tool = create_delegate_tool(agent_map)
        names = [p.name for p in tool.parameters]
        assert "agent" in names
        assert "task" in names
        assert "context" in names

    def test_tool_description_lists_agents(self, agent_map):
        tool = create_delegate_tool(agent_map)
        assert "rain" in tool.description
        assert "frost" in tool.description

    def test_tool_generates_valid_schema(self, agent_map):
        tool = create_delegate_tool(agent_map)
        schema = tool.to_function_schema()
        assert schema["function"]["name"] == "delegate_to"
        params = schema["function"]["parameters"]
        assert "agent" in params["properties"]
        assert "task" in params["properties"]


class TestDelegateExecution:
    @pytest.mark.asyncio
    async def test_delegates_to_target_agent(self, agent_map):
        tool = create_delegate_tool(agent_map)
        await tool.execute(agent="rain", task="write hello world")
        agent_map["rain"].init.assert_awaited_once()
        agent_map["rain"].execute_task.assert_awaited_once()
        task_arg = agent_map["rain"].execute_task.call_args[0][0]
        assert task_arg.description == "write hello world"
        assert task_arg.assigned_to == "rain"

    @pytest.mark.asyncio
    async def test_returns_success_content(self, agent_map):
        agent_map["rain"].execute_task.return_value = TaskResult(
            success=True, content="generated code"
        )
        tool = create_delegate_tool(agent_map)
        result = await tool.execute(agent="rain", task="write code")
        assert "completed" in result
        assert "generated code" in result

    @pytest.mark.asyncio
    async def test_returns_failure_content(self, agent_map):
        agent_map["frost"].execute_task.return_value = TaskResult(
            success=False, content="review failed"
        )
        tool = create_delegate_tool(agent_map)
        result = await tool.execute(agent="frost", task="review code")
        assert "failed" in result
        assert "review failed" in result

    @pytest.mark.asyncio
    async def test_unknown_agent_returns_error(self, agent_map):
        tool = create_delegate_tool(agent_map)
        result = await tool.execute(agent="unknown", task="do something")
        assert "Unknown agent" in result
        assert "unknown" in result

    @pytest.mark.asyncio
    async def test_passes_context_as_metadata(self, agent_map):
        tool = create_delegate_tool(agent_map)
        await tool.execute(agent="rain", task="write code", context="use Python 3.12")
        task_arg = agent_map["rain"].execute_task.call_args[0][0]
        assert "use Python 3.12" in task_arg.metadata["context"]

    @pytest.mark.asyncio
    async def test_empty_context_not_in_metadata(self, agent_map):
        tool = create_delegate_tool(agent_map)
        await tool.execute(agent="rain", task="write code", context="")
        task_arg = agent_map["rain"].execute_task.call_args[0][0]
        assert task_arg.metadata == {}

    @pytest.mark.asyncio
    async def test_truncates_long_results(self, agent_map):
        long_content = "x" * 20000
        agent_map["rain"].execute_task.return_value = TaskResult(success=True, content=long_content)
        tool = create_delegate_tool(agent_map)
        result = await tool.execute(agent="rain", task="generate")
        assert len(result) < 20000
        assert "truncated" in result

    @pytest.mark.asyncio
    async def test_handles_execution_exception(self, agent_map):
        agent_map["dew"].execute_task.side_effect = RuntimeError("connection lost")
        tool = create_delegate_tool(agent_map)
        result = await tool.execute(agent="dew", task="deploy")
        assert "failed" in result
        assert "connection lost" in result

    @pytest.mark.asyncio
    async def test_inits_target_agent_before_execution(self, agent_map):
        tool = create_delegate_tool(agent_map)
        await tool.execute(agent="snow", task="plan something")
        agent_map["snow"].init.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_resets_error_state_after_task(self, agent_map):
        agent_map["rain"].state = AgentState.ERROR
        agent_map["rain"].execute_task.return_value = TaskResult(success=True, content="ok")
        tool = create_delegate_tool(agent_map)
        await tool.execute(agent="rain", task="fix it")
        agent_map["rain"]._set_state.assert_awaited_with(AgentState.IDLE)


class TestDelegateNestingGuard:
    @pytest.mark.asyncio
    async def test_prevents_nested_delegation(self, agent_map):
        tool = create_delegate_tool(agent_map)
        nested_result = None

        # Create a 3-level chain: rain -> frost -> dew
        # Depth 2 (dew) should be blocked (MAX_DEPTH = 2)
        async def _delegate_deep(task):
            nonlocal nested_result
            nested_result = await tool.execute(agent="dew", task="deploy")
            return TaskResult(success=True, content=nested_result)

        async def _delegate_first(task):
            agent_map["frost"].execute_task = _delegate_deep
            return await tool.execute(agent="frost", task="review")

        agent_map["rain"].execute_task = _delegate_first
        await tool.execute(agent="rain", task="write and review")
        assert nested_result is not None
        assert "depth limit" in nested_result.lower()

    @pytest.mark.asyncio
    async def test_depth_resets_after_completion(self, agent_map):
        tool = create_delegate_tool(agent_map)
        await tool.execute(agent="rain", task="first task")
        result = await tool.execute(agent="frost", task="second task")
        assert "completed" in result

    @pytest.mark.asyncio
    async def test_depth_resets_after_error(self, agent_map):
        agent_map["rain"].execute_task.side_effect = RuntimeError("boom")
        tool = create_delegate_tool(agent_map)
        await tool.execute(agent="rain", task="will fail")
        agent_map["frost"].execute_task.return_value = TaskResult(success=True, content="ok")
        agent_map["frost"].execute_task.side_effect = None
        result = await tool.execute(agent="frost", task="should work")
        assert "completed" in result

    @pytest.mark.asyncio
    async def test_concurrent_fanout_not_falsely_rejected(self, agent_map):
        """Snow delegating to three peers in parallel must not see any of
        them rejected with "depth limit reached" — they're siblings, not
        nested. Previously a per-tool counter shared across coroutines
        misclassified the 3rd concurrent call as nested."""
        import asyncio as _asyncio

        tool = create_delegate_tool(agent_map)

        async def _slow(_t):
            # Hold the "running" state so all three are in flight at once.
            await _asyncio.sleep(0.05)
            return TaskResult(success=True, content="done")

        for name in ("fog", "rain", "frost"):
            agent_map[name].execute_task = _slow

        results = await _asyncio.gather(
            tool.execute(agent="fog", task="task A"),
            tool.execute(agent="rain", task="task B"),
            tool.execute(agent="frost", task="task C"),
        )
        for r in results:
            assert "depth limit" not in r.lower(), f"falsely rejected: {r}"


class TestAgentSpecialties:
    def test_all_agents_have_specialties(self):
        expected = {"fog", "rain", "frost", "snow", "dew", "fair"}
        assert set(AGENT_SPECIALTIES.keys()) == expected

    def test_specialties_are_nonempty(self):
        for name, desc in AGENT_SPECIALTIES.items():
            assert len(desc) > 0, f"Empty specialty for {name}"


class TestBuildSharedContext:
    def test_without_calling_agent(self):
        from weather_agents.tools.delegate import _build_shared_context

        result = _build_shared_context(None, "")
        assert result == ""

    def test_with_context_string_only(self):
        from weather_agents.tools.delegate import _build_shared_context

        result = _build_shared_context(None, "some context")
        assert "some context" in result

    def test_with_calling_agent_messages(self):
        from weather_agents.tools.delegate import _build_shared_context

        agent = _make_agent("fog", "雾", "~~")
        agent.memory = Mock()
        agent.memory.short_term = []
        # No messages — should not add context
        result = _build_shared_context(agent, "")
        assert result == ""

    def test_with_recent_non_system_messages(self):

        from weather_agents.tools.delegate import _build_shared_context

        agent = _make_agent("fog", "雾", "~~")
        agent.memory = Mock()

        msg1 = Mock()
        msg1.role = "user"
        msg1.content = "hello from user"
        msg2 = Mock()
        msg2.role = "assistant"
        msg2.content = "hello from agent"

        agent.memory.short_term = [msg1, msg2]
        result = _build_shared_context(agent, "extra info")
        assert "extra info" in result
        assert "hello from user" in result
        assert "hello from agent" in result

    def test_system_messages_filtered_out(self):
        from weather_agents.tools.delegate import _build_shared_context

        agent = _make_agent("fog", "雾", "~~")
        agent.memory = Mock()

        sys_msg = Mock()
        sys_msg.role = "system"
        sys_msg.content = "system prompt"
        user_msg = Mock()
        user_msg.role = "user"
        user_msg.content = "user query"

        agent.memory.short_term = [sys_msg, user_msg]
        result = _build_shared_context(agent, "")
        assert "system prompt" not in result
        assert "user query" in result

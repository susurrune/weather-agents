"""Tests for system factory and task orchestration."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from weather_agents.core.bus import MessageBus
from weather_agents.core.config import AppConfig
from weather_agents.core.factory import (
    AGENT_CLASSES,
    AGENT_COLORS,
    AGENT_EMOJI,
    SystemContext,
    TaskExecutionResult,
    create_system_context,
    orchestrate_task,
)
from weather_agents.core.tool import ToolRegistry


class TestAgentMetadata:
    def test_all_agents_registered(self):
        assert set(AGENT_CLASSES.keys()) == {"fog", "rain", "frost", "snow", "dew", "fair"}

    def test_all_have_emojis(self):
        for name in AGENT_CLASSES:
            assert name in AGENT_EMOJI
            assert len(AGENT_EMOJI[name]) > 0

    def test_all_have_colors(self):
        for name in AGENT_CLASSES:
            assert name in AGENT_COLORS


class TestTaskExecutionResult:
    def test_result_defaults(self):
        r = TaskExecutionResult(
            id="1", agent="fog", description="test", success=True, content="done"
        )
        assert r.id == "1"
        assert r.success is True
        assert r.content == "done"

    def test_failure_result(self):
        r = TaskExecutionResult(
            id="2", agent="rain", description="fail", success=False, content="error"
        )
        assert r.success is False


class TestSystemContext:
    @pytest.mark.asyncio
    async def test_init_all_inits_agents(self):
        cfg = AppConfig()
        bus = MessageBus()
        registry = ToolRegistry()
        from weather_agents.core.llm import LLMClient

        llm = LLMClient(cfg, registry)

        agents = {}
        for name, cls in AGENT_CLASSES.items():
            ag = cls(config=cfg, llm=llm, bus=bus, tool_registry=registry)
            ag.init = AsyncMock()
            ag.close = AsyncMock()
            agents[name] = ag

        ctx = SystemContext(config=cfg, bus=bus, llm=llm, agent_map=agents)
        await ctx.init_all()

        for ag in agents.values():
            ag.init.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_all_closes_agents(self):
        cfg = AppConfig()
        bus = MessageBus()
        registry = ToolRegistry()
        from weather_agents.core.llm import LLMClient

        llm = LLMClient(cfg, registry)

        agents = {}
        for name, cls in AGENT_CLASSES.items():
            ag = cls(config=cfg, llm=llm, bus=bus, tool_registry=registry)
            ag.init = AsyncMock()
            ag.close = AsyncMock()
            agents[name] = ag

        ctx = SystemContext(config=cfg, bus=bus, llm=llm, agent_map=agents)
        await ctx.close_all()

        for ag in agents.values():
            ag.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_init_all_with_mcp(self):
        cfg = AppConfig()
        bus = MessageBus()
        registry = ToolRegistry()
        from weather_agents.core.llm import LLMClient

        llm = LLMClient(cfg, registry)
        mcp = Mock()
        mcp.connect_all = AsyncMock(return_value=["server1"])

        agents = {"snow": Mock(init=AsyncMock(), close=AsyncMock())}
        ctx = SystemContext(config=cfg, bus=bus, llm=llm, agent_map=agents, mcp=mcp)
        await ctx.init_all()

        mcp.connect_all.assert_awaited_once()
        assert ctx.mcp_status == ["server1"]

    @pytest.mark.asyncio
    async def test_init_all_mcp_failure_does_not_block(self):
        cfg = AppConfig()
        bus = MessageBus()
        registry = ToolRegistry()
        from weather_agents.core.llm import LLMClient

        llm = LLMClient(cfg, registry)
        mcp = Mock()
        mcp.connect_all = AsyncMock(side_effect=Exception("boom"))

        agents = {"snow": Mock(init=AsyncMock(), close=AsyncMock())}
        ctx = SystemContext(config=cfg, bus=bus, llm=llm, agent_map=agents, mcp=mcp)
        # Should not raise
        await ctx.init_all()
        agents["snow"].init.assert_awaited_once()


class TestOrchestrateTask:
    @pytest.mark.asyncio
    async def test_no_snow_agent_returns_error(self):
        tasks, results, summary = await orchestrate_task("do something", {}, snow=None)
        assert tasks == []
        assert results == []
        assert "not available" in summary

    @pytest.mark.asyncio
    async def test_no_tasks_generated(self):
        snow = Mock()
        snow.orchestrate = AsyncMock(return_value=[])
        snow.chat = AsyncMock(return_value="nothing to do")

        tasks, results, summary = await orchestrate_task("do something", {}, snow=snow)
        assert tasks == []
        assert results == []
        snow.orchestrate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_single_task_execution(self):
        from weather_agents.core.agent import Task

        task = Task(id="1", description="write code", assigned_to="rain")
        snow = Mock()
        snow.orchestrate = AsyncMock(return_value=[task])
        snow.chat = AsyncMock(return_value="done")

        rain = Mock()
        rain.execute_task = AsyncMock(return_value=Mock(success=True, content="code written"))

        tasks, results, summary = await orchestrate_task(
            "write something",
            agent_map={"rain": rain, "snow": snow},
        )
        assert len(results) == 1
        assert results[0].success is True
        assert "code written" in results[0].content
        rain.execute_task.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_task_with_dependencies(self):
        from weather_agents.core.agent import Task

        task1 = Task(id="1", description="step 1", assigned_to="fog")
        task2 = Task(id="2", description="step 2", assigned_to="rain", parent_id="1")

        snow = Mock()
        snow.orchestrate = AsyncMock(return_value=[task1, task2])
        snow.chat = AsyncMock(return_value="summary")

        fog = Mock()
        fog.execute_task = AsyncMock(return_value=Mock(success=True, content="research done"))
        rain = Mock()
        rain.execute_task = AsyncMock(return_value=Mock(success=True, content="code done"))

        tasks, results, summary = await orchestrate_task(
            "do pipeline",
            agent_map={"fog": fog, "rain": rain, "snow": snow},
        )
        assert len(results) == 2
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_callbacks_invoked(self):
        from weather_agents.core.agent import Task

        task = Task(id="1", description="test task", assigned_to="rain")
        snow = Mock()
        snow.orchestrate = AsyncMock(return_value=[task])
        snow.chat = AsyncMock(return_value="summary")

        rain = Mock()
        rain.execute_task = AsyncMock(return_value=Mock(success=True, content="done"))

        start_calls = []
        done_calls = []

        async def _on_start(t):
            start_calls.append(t.id)

        async def _on_done(t, r):
            done_calls.append((t.id, r.success))

        await orchestrate_task(
            "test",
            agent_map={"rain": rain, "snow": snow},
            on_task_start=_on_start,
            on_task_done=_on_done,
        )
        assert start_calls == ["1"]
        assert done_calls == [("1", True)]

    @pytest.mark.asyncio
    async def test_result_truncate(self):
        from weather_agents.core.agent import Task

        task = Task(id="1", description="test", assigned_to="rain")
        snow = Mock()
        snow.orchestrate = AsyncMock(return_value=[task])
        snow.chat = AsyncMock(return_value="ok")

        rain = Mock()
        rain.execute_task = AsyncMock(return_value=Mock(success=True, content="x" * 100))

        _, results, _ = await orchestrate_task(
            "test",
            agent_map={"rain": rain, "snow": snow},
            result_truncate=10,
        )
        assert len(results[0].content) == 10

    @pytest.mark.asyncio
    async def test_missing_agent_returns_error(self):
        from weather_agents.core.agent import Task

        task = Task(id="1", description="test", assigned_to="nonexistent")
        snow = Mock()
        snow.orchestrate = AsyncMock(return_value=[task])
        snow.chat = AsyncMock(return_value="summary")

        _, results, _ = await orchestrate_task("test", agent_map={"snow": snow})
        assert results[0].success is False
        assert "not found" in results[0].content

    @pytest.mark.asyncio
    async def test_dangling_dependency_fails_fast(self):
        """Task depending on an id that never appears in the plan must fail
        explicitly, not silently run without its upstream context."""
        from weather_agents.core.agent import Task

        # Task "2" depends on "999" which is never planned -> deadlock
        task = Task(id="2", description="needs upstream", assigned_to="rain")
        task.depends_on = ["999"]
        snow = Mock()
        snow.orchestrate = AsyncMock(return_value=[task])
        snow.chat = AsyncMock(return_value="summary")

        rain = Mock()
        rain.execute_task = AsyncMock(return_value=Mock(success=True, content="oops"))

        _, results, _ = await orchestrate_task(
            "broken plan",
            agent_map={"rain": rain, "snow": snow},
        )
        assert len(results) == 1
        assert results[0].success is False
        assert "dependency missing" in results[0].content
        assert "999" in results[0].content
        rain.execute_task.assert_not_awaited()


class TestCreateSystemContext:
    def test_creates_all_agents(self):
        with (
            patch("weather_agents.core.factory.load_config") as mock_load,
            patch("weather_agents.core.factory.register_builtin_tools"),
            patch("weather_agents.core.factory.register_all_skills"),
            patch("weather_agents.core.factory.PluginLoader") as mock_plugin,
        ):
            mock_load.return_value = AppConfig()
            mock_loader = Mock()
            mock_loader.load_from_directories = Mock(return_value=[])
            mock_plugin.return_value = mock_loader

            ctx = create_system_context()
            assert len(ctx.agent_map) == 6
            assert set(ctx.agent_map.keys()) == {"fog", "rain", "frost", "snow", "dew", "fair"}

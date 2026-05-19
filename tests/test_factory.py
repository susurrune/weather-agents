"""Tests for system factory and task orchestration."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from weather_agents.core.bus import MessageBus
from weather_agents.core.config import AppConfig
from weather_agents.core.factory import (
    AGENT_CLASSES,
    SystemContext,
    TaskExecutionResult,
    create_system_context,
    orchestrate_task,
)
from weather_agents.core.icons import AGENT_COLORS, AGENT_EMOJI
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

        ctx = SystemContext(config=cfg, bus=bus, llm=llm, agent_map=agents, tool_registry=registry)
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

        ctx = SystemContext(config=cfg, bus=bus, llm=llm, agent_map=agents, tool_registry=registry)
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
        ctx = SystemContext(
            config=cfg, bus=bus, llm=llm, agent_map=agents, mcp=mcp, tool_registry=registry
        )
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
        ctx = SystemContext(
            config=cfg, bus=bus, llm=llm, agent_map=agents, mcp=mcp, tool_registry=registry
        )
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
    async def test_upstream_data_passed_directly_to_downstream(self):
        """Upstream agent's full output is passed in the task description
        (no shared memory lookup needed)."""
        from weather_agents.core.agent import Task

        snow = Mock()
        snow.orchestrate = AsyncMock(
            return_value=[
                Task(id="1", description="step one", assigned_to="rain"),
                Task(id="2", description="step two", assigned_to="fog", depends_on=["1"]),
            ]
        )
        snow.chat = AsyncMock(return_value="summary")
        snow.memory = Mock()
        snow.memory.get_active_session = Mock(return_value=None)

        rain = Mock()
        rain.execute_task = AsyncMock(return_value=Mock(success=True, content="rain output"))

        executed_desc: list[str] = []

        async def _fog_execute(task):
            executed_desc.append(task.description)
            return Mock(success=True, content="fog output")

        fog = Mock()
        fog.execute_task = AsyncMock(side_effect=_fog_execute)
        fog.memory = Mock()

        await orchestrate_task(
            "goal",
            agent_map={"rain": rain, "fog": fog, "snow": snow},
        )

        # fog's task must include rain's full output in the description
        assert any("rain output" in desc for desc in executed_desc)

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


class TestQualityGateAndReplan:
    """Quality gate (thin/placeholder/truncated rejection) and the
    judge-driven re-plan loop. Together these ensure the orchestrator
    doesn't quietly accept "Done." and stop early."""

    def test_is_thin_content_classifies_placeholders(self):
        from weather_agents.core.factory import _is_thin_content

        assert _is_thin_content("") is True
        assert _is_thin_content("   ") is True
        assert _is_thin_content("Done.") is True
        assert _is_thin_content("好的") is True
        assert _is_thin_content("[truncated] max rounds reached") is True
        assert _is_thin_content("[Error: tool 'x' not found]") is True
        # Real (even short) answers must NOT be flagged
        assert _is_thin_content("code written") is False
        assert _is_thin_content("Read the file: lines 1-12") is False
        # Real sentence containing the word "done" survives
        assert _is_thin_content("All preflight checks done; proceeding.") is False

    def test_is_thin_content_catches_status_reports(self):
        """The "声称完成但无交付物" failure class — a short response that
        claims completion but contains no actual artifact markers."""
        from weather_agents.core.factory import _is_thin_content

        # Short status reports — should be flagged
        assert _is_thin_content("调研工作已完成。") is True
        assert _is_thin_content("已成功完成了调研任务。") is True
        assert _is_thin_content("文章已经写好。") is True
        assert _is_thin_content("Successfully completed the research.") is True
        assert _is_thin_content("I have completed the task as requested.") is True

        # Status keyword + real markdown content — must NOT be flagged
        long_real = (
            "## 5 个向量数据库对比\n\n"
            "| 名称 | 特性 |\n|---|---|\n| Milvus | 分布式 |\n"
            "调研已完成，详见上表。"
        )
        assert _is_thin_content(long_real) is False
        # Status keyword + URL — has a deliverable marker
        assert _is_thin_content("调研完成，参考 https://milvus.io 文档。") is False
        # Status keyword + code block
        assert _is_thin_content("已完成。\n```py\nfrom x import y\n```") is False
        # Long detailed response with "已完成" buried — survives
        assert (
            _is_thin_content(
                "Milvus 是分布式向量数据库，支持 HNSW 索引，单机可处理 10 亿向量。"
                "Qdrant 用 Rust 写，提供 REST API。Weaviate 内建多模态支持。"
                "Chroma 是嵌入式方案，适合原型。Vespa 是 Yahoo 出品，工业级。"
                "调研已完成。" * 2
            )
            is False
        )

    @pytest.mark.asyncio
    async def test_thin_content_triggers_retry(self):
        """An agent returning 'Done.' must be retried by the orchestrator."""
        from weather_agents.core.agent import Task

        attempts = {"n": 0}

        async def _flaky(_t):
            attempts["n"] += 1
            # First attempt: placeholder ack. Second: real output.
            if attempts["n"] == 1:
                return Mock(success=True, content="Done.")
            return Mock(success=True, content="actual deliverable here")

        rain = Mock()
        rain.execute_task = AsyncMock(side_effect=_flaky)

        snow = Mock()
        snow.orchestrate = AsyncMock(
            return_value=[Task(id="1", description="write something", assigned_to="rain")]
        )
        snow.chat = AsyncMock(return_value="ok")

        _, results, _ = await orchestrate_task(
            "write something",
            agent_map={"rain": rain, "snow": snow},
            max_task_retries=3,
        )
        assert attempts["n"] >= 2, "thin content should have triggered a retry"
        assert "actual deliverable" in results[0].content

    @pytest.mark.asyncio
    async def test_replan_called_when_judge_says_not_achieved(self):
        """Multi-task plan: if judge returns achieved=false, snow.replan_for_missing
        must be invoked with the missing description."""
        from weather_agents.core.agent import Task

        snow = Mock()
        # Initial plan: two tasks
        snow.orchestrate = AsyncMock(
            return_value=[
                Task(id="1", description="step A", assigned_to="fog"),
                Task(id="2", description="step B", assigned_to="rain"),
            ]
        )
        # Judge says "no, missing X" on round 1, "achieved" on round 2,
        # then a final summary. The judge runs each round, so we need
        # enough side-effects to cover all calls.
        snow.chat = AsyncMock(
            side_effect=[
                '{"achieved": false, "missing": "测试用例覆盖率"}',
                '{"achieved": true, "missing": ""}',
                "final summary",
            ]
        )
        # Replan returns one extra task
        snow.replan_for_missing = AsyncMock(
            return_value=[Task(id="3", description="add tests", assigned_to="frost")]
        )

        fog = Mock()
        fog.execute_task = AsyncMock(return_value=Mock(success=True, content="A done"))
        rain = Mock()
        rain.execute_task = AsyncMock(return_value=Mock(success=True, content="B done"))
        frost = Mock()
        frost.execute_task = AsyncMock(return_value=Mock(success=True, content="tests added"))

        _, results, _ = await orchestrate_task(
            "build thing",
            agent_map={"fog": fog, "rain": rain, "frost": frost, "snow": snow},
        )
        snow.replan_for_missing.assert_awaited_once()
        # The follow-up task ran
        frost.execute_task.assert_awaited_once()
        assert any(r.id == "3" for r in results)


class TestPlanConfirmGate:
    """on_planned returning False must abort orchestration before any
    sub-task runs — the contract PLAN mode relies on."""

    @pytest.mark.asyncio
    async def test_on_planned_false_aborts(self):
        from weather_agents.core.agent import Task

        snow = Mock()
        snow.orchestrate = AsyncMock(
            return_value=[Task(id="1", description="x", assigned_to="rain")]
        )
        snow.chat = AsyncMock(return_value="never called")

        rain = Mock()
        rain.execute_task = AsyncMock(return_value=Mock(success=True, content="oops"))

        async def _reject(_tasks):
            return False  # user pressed Esc

        tasks, results, summary = await orchestrate_task(
            "do thing",
            agent_map={"rain": rain, "snow": snow},
            on_planned=_reject,
        )
        # Tasks were planned but never executed
        assert len(tasks) == 1
        assert results == []
        assert "CANCELLED" in summary
        rain.execute_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_on_planned_true_proceeds(self):
        from weather_agents.core.agent import Task

        snow = Mock()
        snow.orchestrate = AsyncMock(
            return_value=[Task(id="1", description="x", assigned_to="rain")]
        )
        snow.chat = AsyncMock(return_value="ok")

        rain = Mock()
        rain.execute_task = AsyncMock(
            return_value=Mock(success=True, content="real deliverable text here")
        )

        async def _accept(_tasks):
            return True

        _, results, _ = await orchestrate_task(
            "do thing",
            agent_map={"rain": rain, "snow": snow},
            on_planned=_accept,
        )
        assert len(results) == 1
        rain.execute_task.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_on_planned_none_proceeds_backcompat(self):
        """Legacy callbacks returning None must still mean 'proceed'."""
        from weather_agents.core.agent import Task

        snow = Mock()
        snow.orchestrate = AsyncMock(
            return_value=[Task(id="1", description="x", assigned_to="rain")]
        )
        snow.chat = AsyncMock(return_value="ok")

        rain = Mock()
        rain.execute_task = AsyncMock(
            return_value=Mock(success=True, content="real deliverable text here")
        )

        async def _silent(_tasks):
            return None  # backward-compat

        _, results, _ = await orchestrate_task(
            "do thing",
            agent_map={"rain": rain, "snow": snow},
            on_planned=_silent,
        )
        assert len(results) == 1


# E2E integration tests for orchestration: plan -> execute -> judge -> replan.
# Uses shaped fake agents (not Mock specs) so callback wiring runs for real.


def _shaped_agent(name):
    from unittest.mock import AsyncMock

    class _Agent:
        def __init__(self):
            self.name = name
            self._exec_seq = []
            self._chat_seq = []
            self._exec_idx = 0
            self._chat_idx = 0
            self.execute_task = AsyncMock(side_effect=self._next_exec)
            self.chat = AsyncMock(side_effect=self._next_chat)
            self.orchestrate = AsyncMock()
            self.replan_for_missing = AsyncMock(return_value=[])

        def queue_exec(self, result):
            self._exec_seq.append(result)

        def queue_chat(self, text):
            self._chat_seq.append(text)

        async def _next_exec(self, *_a, **_k):
            assert self._exec_idx < len(self._exec_seq), f"unexpected execute_task call on {name}"
            r = self._exec_seq[self._exec_idx]
            self._exec_idx += 1
            return r

        async def _next_chat(self, *_a, **_k):
            assert self._chat_idx < len(self._chat_seq), f"unexpected chat call on {name}"
            r = self._chat_seq[self._chat_idx]
            self._chat_idx += 1
            return r

    return _Agent()


class TestOrchestrationE2E:
    @pytest.mark.asyncio
    async def test_happy_path_two_tasks_succeed(self):
        from unittest.mock import Mock

        from weather_agents.core.agent import Task

        snow = _shaped_agent("snow")
        fog = _shaped_agent("fog")
        rain = _shaped_agent("rain")

        async def _orch(_goal):
            return [
                Task(id="1", description="research X", assigned_to="fog"),
                Task(id="2", description="write Y", assigned_to="rain", parent_id="1"),
            ]

        snow.orchestrate = _orch
        fog.queue_exec(Mock(success=True, content="A. Milvus B. Qdrant C. Weaviate" * 5))
        rain.queue_exec(Mock(success=True, content="## Comparison\n| name | ... |" * 5))
        snow.queue_chat('{"achieved": true, "missing": ""}')
        snow.queue_chat("Final synthesized answer.")

        plan_seen = []
        starts = []
        dones = []

        async def _on_plan(tasks):
            plan_seen.append(list(tasks))
            return True

        async def _on_start(t):
            starts.append(t.id)

        async def _on_done(t, r):
            dones.append((t.id, r.success))

        tasks, results, summary = await orchestrate_task(
            "do something",
            agent_map={"fog": fog, "rain": rain, "snow": snow},
            on_planned=_on_plan,
            on_task_start=_on_start,
            on_task_done=_on_done,
        )
        assert len(plan_seen) == 1
        assert len(starts) == 2
        assert {t for t, _ in dones} == {"1", "2"}
        assert all(s for _, s in dones)
        assert len(results) == 2
        assert "Final synthesized answer" in summary

    @pytest.mark.asyncio
    async def test_replan_triggers_then_succeeds(self):
        """2-task initial plan → both thin → judge says no → snow replans
        1 more task → that succeeds → judge says yes → summary."""
        from unittest.mock import Mock

        from weather_agents.core.agent import Task

        snow = _shaped_agent("snow")
        fog = _shaped_agent("fog")

        async def _orch(_goal):
            return [
                Task(id="1", description="research", assigned_to="fog"),
                Task(id="2", description="summarize", assigned_to="fog"),
            ]

        snow.orchestrate = _orch
        # Each task gets max_task_retries=1 attempts. With thin content
        # each one is rejected, retried once, and finally accepted as
        # success=True content=thin (orchestrator never marks "completed"
        # as failed at that point; judge does the real assessment).
        for _ in range(2 * 1):  # 2 tasks × 1 attempt each (no retry)
            fog.queue_exec(Mock(success=True, content="调研已完成。"))
        # Replan task: real content this time
        fog.queue_exec(
            Mock(success=True, content="Milvus | Qdrant | Weaviate" * 10)
        )

        snow.queue_chat('{"achieved": false, "missing": "no actual names"}')
        snow.queue_chat('{"achieved": true, "missing": ""}')
        snow.queue_chat("Done.")

        async def _replan(*_a, **_k):
            return [Task(id="3", description="research better", assigned_to="fog")]

        snow.replan_for_missing = _replan

        plan_rounds = []

        async def _on_plan(tasks):
            plan_rounds.append(len(tasks))
            return True

        _, results, _ = await orchestrate_task(
            "research",
            agent_map={"fog": fog, "snow": snow},
            on_planned=_on_plan,
            max_task_retries=1,
        )
        # Plan: round 1 has 2 tasks, round 2 has 3 (initial 2 + replan 1)
        assert plan_rounds == [2, 3]
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_cycle_detection_fails_loudly(self):
        from weather_agents.core.agent import Task

        snow = _shaped_agent("snow")
        rain = _shaped_agent("rain")
        t1 = Task(id="1", description="A", assigned_to="rain")
        t1.depends_on = ["2"]
        t2 = Task(id="2", description="B", assigned_to="rain")
        t2.depends_on = ["1"]

        async def _orch(_goal):
            return [t1, t2]

        snow.orchestrate = _orch
        # Multi-result path: judge runs (results > 1), then summary.
        # Both tasks are cycle-detected (failed) — judge sees no real
        # deliverable; conservative fallback to achieved=true (so it
        # doesn't loop forever on a doomed plan). Queue: judge + summary.
        snow.queue_chat('{"achieved": true, "missing": ""}')
        snow.queue_chat("summary")
        _, results, _ = await orchestrate_task("circular", agent_map={"rain": rain, "snow": snow})
        assert all("cycle" in r.content.lower() for r in results)
        assert rain.execute_task.await_count == 0

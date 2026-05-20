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

        snow.chat_oneshot = AsyncMock(return_value="nothing to do")

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

        snow.chat_oneshot = AsyncMock(return_value="done")

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

        snow.chat_oneshot = AsyncMock(return_value="summary")

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

        snow.chat_oneshot = AsyncMock(return_value="summary")

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

        snow.chat_oneshot = AsyncMock(return_value="ok")

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

        snow.chat_oneshot = AsyncMock(return_value="summary")

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

        snow.chat_oneshot = AsyncMock(return_value="summary")
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

        snow.chat_oneshot = AsyncMock(return_value="summary")

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

        snow.chat_oneshot = AsyncMock(return_value="ok")

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
        # then a final summary. Both the judge and the summary route via
        # chat_oneshot since round 2 (lightweight-model path); chat is
        # kept around for any code path that still uses it.
        _chat_seq = [
            '{"achieved": false, "missing": "测试用例覆盖率"}',
            '{"achieved": true, "missing": ""}',
            "final summary",
        ]
        snow.chat_oneshot = AsyncMock(side_effect=_chat_seq)
        snow.chat = AsyncMock(side_effect=list(_chat_seq))
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

        snow.chat_oneshot = AsyncMock(return_value="never called")

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

        snow.chat_oneshot = AsyncMock(return_value="ok")

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

        snow.chat_oneshot = AsyncMock(return_value="ok")

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
            # chat_oneshot was added in round 2 to route judge/summary
            # through the lightweight model. The fake serves both from
            # the same queue so existing queue_chat scripts keep working
            # — production code is also designed so any single call site
            # uses one OR the other, not both.
            self.chat_oneshot = AsyncMock(side_effect=self._next_chat)
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
        fog.queue_exec(Mock(success=True, content="Milvus | Qdrant | Weaviate" * 10))

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


class TestObviouslyCompleteFastPath:
    """Skip the judge when every task succeeded with substantial output —
    paying ~5-15s + tokens to be told 'yes done' was the largest avoidable
    cost in the previous orchestration review. The fast-path must NEVER
    fire when content is thin or when any task failed: those are the
    exact cases the judge was designed to catch."""

    def test_helper_passes_when_all_substantial_and_successful(self):
        from weather_agents.core.factory import (
            TaskExecutionResult,
            _looks_obviously_complete,
        )

        results = [
            TaskExecutionResult(
                id="1",
                agent="fog",
                description="research",
                success=True,
                content="A" * 500,
            ),
            TaskExecutionResult(
                id="2",
                agent="rain",
                description="write",
                success=True,
                content="B" * 500,
            ),
        ]
        assert _looks_obviously_complete(results) is True

    def test_helper_rejects_thin_content(self):
        from weather_agents.core.factory import (
            TaskExecutionResult,
            _looks_obviously_complete,
        )

        results = [
            TaskExecutionResult(
                id="1",
                agent="fog",
                description="research",
                success=True,
                content="调研已完成。",  # 6 chars — thin status report
            ),
            TaskExecutionResult(
                id="2",
                agent="rain",
                description="write",
                success=True,
                content="X" * 800,
            ),
        ]
        assert _looks_obviously_complete(results) is False

    def test_helper_rejects_any_failure(self):
        from weather_agents.core.factory import (
            TaskExecutionResult,
            _looks_obviously_complete,
        )

        results = [
            TaskExecutionResult(
                id="1",
                agent="fog",
                description="r",
                success=True,
                content="A" * 500,
            ),
            TaskExecutionResult(
                id="2",
                agent="rain",
                description="w",
                success=False,
                content="B" * 500,
            ),
        ]
        assert _looks_obviously_complete(results) is False

    def test_helper_rejects_failure_marker_in_content(self):
        from weather_agents.core.factory import (
            TaskExecutionResult,
            _looks_obviously_complete,
        )

        # success=True but content carries a [truncated] envelope — the
        # tool layer marked it successful but the work is actually
        # incomplete. Must not fast-path past this.
        results = [
            TaskExecutionResult(
                id="1",
                agent="fog",
                description="r",
                success=True,
                content="A" * 500,
            ),
            TaskExecutionResult(
                id="2",
                agent="rain",
                description="w",
                success=True,
                content=("B" * 450) + " [truncated] max rounds reached",
            ),
        ]
        assert _looks_obviously_complete(results) is False

    @pytest.mark.asyncio
    async def test_fast_path_skips_judge_when_substantial(self):
        """Two substantial successful tasks must not consume a judge LLM
        call — the queued snow.chat should go to the SUMMARY, not the
        judge."""
        from weather_agents.core.agent import Task

        snow = _shaped_agent("snow")
        fog = _shaped_agent("fog")
        rain = _shaped_agent("rain")

        async def _orch(_goal):
            return [
                Task(id="1", description="research", assigned_to="fog"),
                Task(id="2", description="write", assigned_to="rain"),
            ]

        snow.orchestrate = _orch
        # Substantial content — should trigger fast-path
        fog.queue_exec(Mock(success=True, content="A" * 500))
        rain.queue_exec(Mock(success=True, content="B" * 500))
        # Only ONE chat call expected (summary). If the judge ran, the
        # _next_chat side_effect would consume this and the summary
        # call would assert-fail with "unexpected chat call on snow".
        snow.queue_chat("final summary text")

        _, results, summary = await orchestrate_task(
            "build thing",
            agent_map={"fog": fog, "rain": rain, "snow": snow},
        )
        assert len(results) == 2
        # Judge was NOT consulted — only the summary call used the queue.
        assert snow._chat_idx == 1
        assert "final summary" in summary

    @pytest.mark.asyncio
    async def test_fast_path_does_not_fire_on_thin_content(self):
        """Thin content (status report without deliverable) must still
        flow through the judge — that's the exact regression the judge
        guards against."""
        from weather_agents.core.agent import Task

        snow = _shaped_agent("snow")
        fog = _shaped_agent("fog")

        async def _orch(_goal):
            return [
                Task(id="1", description="r1", assigned_to="fog"),
                Task(id="2", description="r2", assigned_to="fog"),
            ]

        snow.orchestrate = _orch
        # Thin "已完成" stubs — must NOT fast-path.
        fog.queue_exec(Mock(success=True, content="已完成"))
        fog.queue_exec(Mock(success=True, content="已完成"))
        # Judge says achieved (conservative) so we exit without replanning.
        snow.queue_chat('{"achieved": true, "missing": ""}')
        snow.queue_chat("summary")

        _, results, _ = await orchestrate_task(
            "g",
            agent_map={"fog": fog, "snow": snow},
        )
        assert len(results) == 2
        # Both judge AND summary consumed = 2 chat calls.
        assert snow._chat_idx == 2


class TestToolCallSignature:
    """Tool-signature loop detector helper — covers the fingerprint shape
    for the call patterns that matter (file edits, shell, search,
    delegate). The signature itself must be stable so the same logical
    call always folds to the same bucket; otherwise the loop counter
    never accumulates and the detector is silently disabled."""

    def test_file_edit_keyed_on_path(self):
        from weather_agents.core.agent import _tool_call_signature

        s1 = _tool_call_signature("edit_file", {"path": "/a/b.js", "old_text": "x"})
        s2 = _tool_call_signature("edit_file", {"path": "/a/b.js", "old_text": "y"})
        assert s1 == s2  # different patches on same file -> same loop bucket
        s3 = _tool_call_signature("edit_file", {"path": "/a/c.js", "old_text": "x"})
        assert s1 != s3

    def test_bash_keyed_on_first_token(self):
        from weather_agents.core.agent import _tool_call_signature

        s1 = _tool_call_signature("run_bash", {"command": "where soffice"})
        s2 = _tool_call_signature("run_bash", {"command": "where soffice 2>nul"})
        s3 = _tool_call_signature("run_bash", {"command": "ls /tmp"})
        assert s1 == s2
        assert s1 != s3

    def test_search_lowercased_and_truncated(self):
        from weather_agents.core.agent import _tool_call_signature

        s1 = _tool_call_signature("web_search", {"query": "Python AsyncIO"})
        s2 = _tool_call_signature("web_search", {"query": "python asyncio"})
        assert s1 == s2  # case-insensitive

    def test_delegate_keyed_on_agent(self):
        from weather_agents.core.agent import _tool_call_signature

        s1 = _tool_call_signature("delegate_to", {"agent": "frost", "task": "A"})
        s2 = _tool_call_signature("delegate_to", {"agent": "frost", "task": "B"})
        s3 = _tool_call_signature("delegate_to", {"agent": "rain", "task": "A"})
        assert s1 == s2
        assert s1 != s3

    def test_unknown_tool_falls_back_to_name(self):
        from weather_agents.core.agent import _tool_call_signature

        s = _tool_call_signature("some_custom_tool", {"x": 1})
        # Coarse but safe — every call collapses to the same bucket,
        # which is fine for "called custom_tool 8 times" detection.
        assert s == "some_custom_tool"

    def test_none_args_safe(self):
        from weather_agents.core.agent import _tool_call_signature

        # The loop runs even when arg parsing failed — signature must not
        # crash on missing args dict.
        assert _tool_call_signature("edit_file", None) == "edit_file:"


class TestHardCapLowered:
    """Regression: hard cap was lowered 100 → 40 after the PPT case
    study. Bumping it back up without a code review usually signals
    'patch around a stuck loop', so test pins the new value."""

    def test_hard_cap_default(self, app_config, bus, mock_llm):
        from weather_agents.agents.fog import FogAgent
        from weather_agents.core.skill import SkillRegistry
        from weather_agents.core.tool import ToolRegistry

        agent = FogAgent(
            config=app_config,
            llm=mock_llm,
            bus=bus,
            tool_registry=ToolRegistry(),
            skill_registry=SkillRegistry(),
        )
        assert agent._max_tool_rounds_hard_cap == 40


class TestChatOneshotLightweightRouting:
    """chat_oneshot must route through the configured lightweight_model
    when available, and must NOT pollute the agent's chat history. Both
    properties are the reason for the helper's existence — the original
    judge/summary path used snow.chat() which persisted the verification
    prompt into short_term and routed through the heavy model."""

    @pytest.mark.asyncio
    async def test_uses_lightweight_model_when_configured(self, app_config, bus, mock_llm):
        from weather_agents.agents.fog import FogAgent
        from weather_agents.core.skill import SkillRegistry
        from weather_agents.core.tool import ToolRegistry

        app_config.llm.lightweight_model = "deepseek/deepseek-v4-flash"
        agent = FogAgent(
            config=app_config,
            llm=mock_llm,
            bus=bus,
            tool_registry=ToolRegistry(),
            skill_registry=SkillRegistry(),
        )
        await agent.chat_oneshot("classify this")
        # The mock LLM records the overrides it received — verify the
        # lightweight model was selected, not the agent's default.
        call_overrides = mock_llm.last_overrides
        assert call_overrides and call_overrides.get("model") == ("deepseek/deepseek-v4-flash")

    @pytest.mark.asyncio
    async def test_explicit_model_arg_wins_over_config(self, app_config, bus, mock_llm):
        from weather_agents.agents.fog import FogAgent
        from weather_agents.core.skill import SkillRegistry
        from weather_agents.core.tool import ToolRegistry

        app_config.llm.lightweight_model = "deepseek/deepseek-v4-flash"
        agent = FogAgent(
            config=app_config,
            llm=mock_llm,
            bus=bus,
            tool_registry=ToolRegistry(),
            skill_registry=SkillRegistry(),
        )
        await agent.chat_oneshot("x", model="claude-haiku-4-5")
        assert mock_llm.last_overrides["model"] == "claude-haiku-4-5"

    @pytest.mark.asyncio
    async def test_does_not_persist_to_short_term(self, app_config, bus, mock_llm):
        from weather_agents.agents.fog import FogAgent
        from weather_agents.core.skill import SkillRegistry
        from weather_agents.core.tool import ToolRegistry

        agent = FogAgent(
            config=app_config,
            llm=mock_llm,
            bus=bus,
            tool_registry=ToolRegistry(),
            skill_registry=SkillRegistry(),
        )
        before = len(agent.memory.short_term)
        await agent.chat_oneshot("don't pollute memory")
        # The base prompt itself may be added by other init paths, but
        # chat_oneshot specifically must not append a user message or
        # assistant reply for its own call.
        after = len(agent.memory.short_term)
        assert after == before

    @pytest.mark.asyncio
    async def test_falls_back_to_default_when_lightweight_unset(self, app_config, bus, mock_llm):
        from weather_agents.agents.fog import FogAgent
        from weather_agents.core.skill import SkillRegistry
        from weather_agents.core.tool import ToolRegistry

        app_config.llm.lightweight_model = None
        agent = FogAgent(
            config=app_config,
            llm=mock_llm,
            bus=bus,
            tool_registry=ToolRegistry(),
            skill_registry=SkillRegistry(),
        )
        await agent.chat_oneshot("x")
        # No model override -> LLMClient picks the agent default.
        overrides = mock_llm.last_overrides
        assert overrides is None or "model" not in overrides

    @pytest.mark.asyncio
    async def test_judge_routes_through_chat_oneshot(self):
        """Regression: _judge_goal_achievement must call snow.chat_oneshot,
        not snow.chat — the previous routing was the largest avoidable
        spend in the orchestration loop (heavy model for JSON yes/no)."""
        from weather_agents.core.agent import Task

        snow = _shaped_agent("snow")
        fog = _shaped_agent("fog")
        rain = _shaped_agent("rain")

        async def _orch(_goal):
            return [
                Task(id="1", description="a", assigned_to="fog"),
                Task(id="2", description="b", assigned_to="rain"),
            ]

        snow.orchestrate = _orch
        # Thin content forces the judge path (fast-path won't fire).
        fog.queue_exec(Mock(success=True, content="thin"))
        rain.queue_exec(Mock(success=True, content="thin"))
        # Two chats: judge says achieved, then summary.
        snow.queue_chat('{"achieved": true, "missing": ""}')
        snow.queue_chat("done")

        await orchestrate_task(
            "g",
            agent_map={"fog": fog, "rain": rain, "snow": snow},
        )
        # chat_oneshot mock should have been called twice (judge + summary).
        # snow.chat (heavy model) must not have been called.
        assert snow.chat_oneshot.await_count == 2
        assert snow.chat.await_count == 0


class TestToolRouterPerTurnCache:
    """Regression: select_relevant_tools must run AT MOST once per
    chat_stream turn when nothing invalidates the cache key (suppressed
    tools, active skill set). Previously the router ran every loop
    iteration — for a 20-round turn that was 19 wasted O(n_tools)
    keyword-scoring passes."""

    @pytest.mark.asyncio
    async def test_router_called_once_across_multiple_rounds(self, app_config, bus, mock_llm):
        from unittest.mock import patch

        from weather_agents.agents.fog import FogAgent
        from weather_agents.core.llm import StreamEvent
        from weather_agents.core.skill import SkillRegistry
        from weather_agents.core.tool import Tool, ToolRegistry

        registry = ToolRegistry()
        registry.register(
            Tool(
                name="echo",
                description="echo back",
                parameters=[],
                handler=AsyncMock(return_value="ok"),
            )
        )

        agent = FogAgent(
            config=app_config,
            llm=mock_llm,
            bus=bus,
            tool_registry=registry,
            skill_registry=SkillRegistry(),
        )

        # Build a stream that yields a tool_call on round 1 and a
        # plain content+done on round 2 — exercises the multi-round
        # path so we can verify the router cache.
        round_counter = {"n": 0}

        async def _stream_with_tools(*_a, **_kw):
            round_counter["n"] += 1
            if round_counter["n"] == 1:
                yield StreamEvent(
                    type="tool_call",
                    tool_call={
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "echo", "arguments": "{}"},
                    },
                )
                yield StreamEvent(
                    type="done",
                    usage={"prompt_tokens": 1, "completion_tokens": 1},
                )
            else:
                yield StreamEvent(type="content", text="done")
                yield StreamEvent(
                    type="done",
                    usage={"prompt_tokens": 1, "completion_tokens": 1},
                )

        mock_llm.stream_with_tools = _stream_with_tools

        # Patch the router so we can count calls. Patching the module
        # symbol is sufficient because the agent imports it locally
        # inside the helper (so each helper call resolves through this
        # module-level binding).
        with patch(
            "weather_agents.core.tool_router.select_relevant_tools",
            wraps=__import__(
                "weather_agents.core.tool_router",
                fromlist=["select_relevant_tools"],
            ).select_relevant_tools,
        ) as spy:
            events: list[dict] = []
            async for ev in agent.chat_stream("call echo"):
                events.append(ev)

            # 2 LLM rounds executed (tool_call → tool result → final),
            # but the router should only fire on round 1 (cache hit on
            # round 2 because skill set + suppressed_tools didn't
            # change). Tolerate the helper being entered twice with one
            # cache miss — the assertion is that it didn't run N times.
            assert spy.call_count <= 1, (
                f"router ran {spy.call_count}x across 2 rounds — per-turn cache regressed"
            )

    @pytest.mark.asyncio
    async def test_router_recomputes_when_suppressed_tools_change(self, app_config, bus, mock_llm):
        from unittest.mock import patch

        from weather_agents.agents.fog import FogAgent
        from weather_agents.core.llm import StreamEvent
        from weather_agents.core.skill import SkillRegistry
        from weather_agents.core.tool import Tool, ToolRegistry

        registry = ToolRegistry()
        registry.register(
            Tool(
                name="echo",
                description="echo",
                parameters=[],
                handler=AsyncMock(return_value="ok"),
            )
        )
        # Tool that ALWAYS returns [CircuitBreakerOpen] — adding the
        # tool name to suppressed_tools is what invalidates the router
        # cache key, forcing a recompute on the next round.
        registry.register(
            Tool(
                name="flaky",
                description="x",
                parameters=[],
                handler=AsyncMock(return_value="Error: [CircuitBreakerOpen] flaky down"),
            )
        )

        agent = FogAgent(
            config=app_config,
            llm=mock_llm,
            bus=bus,
            tool_registry=registry,
            skill_registry=SkillRegistry(),
        )

        round_counter = {"n": 0}

        async def _stream(*_a, **_kw):
            round_counter["n"] += 1
            if round_counter["n"] == 1:
                # Round 1: call the breaker-open tool to mutate
                # suppressed_tools.
                yield StreamEvent(
                    type="tool_call",
                    tool_call={
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "flaky", "arguments": "{}"},
                    },
                )
                yield StreamEvent(type="done", usage={"prompt_tokens": 1, "completion_tokens": 1})
            else:
                yield StreamEvent(type="content", text="ok")
                yield StreamEvent(type="done", usage={"prompt_tokens": 1, "completion_tokens": 1})

        mock_llm.stream_with_tools = _stream

        with patch(
            "weather_agents.core.tool_router.select_relevant_tools",
            wraps=__import__(
                "weather_agents.core.tool_router",
                fromlist=["select_relevant_tools"],
            ).select_relevant_tools,
        ) as spy:
            async for _ in agent.chat_stream("trigger breaker"):
                pass

            # suppressed_tools changed between round 1 and round 2 →
            # cache key changed → router must recompute once more.
            assert spy.call_count == 2

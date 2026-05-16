"""Regression tests for memory / delegation isolation bugs.

Covers fixes for symptoms observed in production:
- agents pulling fragments from prior unrelated sessions on a new turn
- caller agents adopting the voice of an agent they delegated to
- ``empty response, retrying`` appearing when delegation succeeds but the
  model emits no plain text after the tool call
- auto-compact firing too early and replacing user directives with
  narrative summaries
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from weather_agents.core.agent import (
    _call_agent_var,
    _synthesize_delegation_summary,
    get_call_agent,
)
from weather_agents.core.config import AppConfig, MemoryConfig


@pytest.fixture
def isolated_config(tmp_path):
    """AppConfig with a per-test SQLite DB so persisted rows don't leak."""
    cfg = AppConfig()
    cfg.memory = MemoryConfig(db_path=str(tmp_path / "isolated.db"), short_term_limit=50)
    return cfg


class TestInitCreatesSession:
    """``BaseAgent.init()`` must own a session — both for REPL and delegate targets."""

    @pytest.mark.asyncio
    async def test_init_auto_creates_session(self, isolated_config, mock_llm, bus, tool_registry):
        from weather_agents.agents.fog import FogAgent

        agent = FogAgent(config=isolated_config, llm=mock_llm, bus=bus, tool_registry=tool_registry)
        assert agent.memory.get_active_session() is None
        await agent.init()
        assert agent.memory.get_active_session() is not None
        await agent.close()

    @pytest.mark.asyncio
    async def test_init_idempotent_does_not_create_second_session(
        self, isolated_config, mock_llm, bus, tool_registry
    ):
        from weather_agents.agents.fog import FogAgent

        agent = FogAgent(config=isolated_config, llm=mock_llm, bus=bus, tool_registry=tool_registry)
        await agent.init()
        sid1 = agent.memory.get_active_session()
        await agent.init()
        sid2 = agent.memory.get_active_session()
        assert sid1 == sid2
        await agent.close()

    @pytest.mark.asyncio
    async def test_persisted_messages_carry_session_id(
        self, isolated_config, mock_llm, bus, tool_registry
    ):
        """Without a session id, persisted rows can leak across processes."""
        from weather_agents.agents.fog import FogAgent
        from weather_agents.core.memory import Memory

        agent = FogAgent(config=isolated_config, llm=mock_llm, bus=bus, tool_registry=tool_registry)
        await agent.init()
        agent.memory.add_message("user", "scoped message")
        await agent.memory._flush_pending()
        session_id = agent.memory.get_active_session()

        # Fresh memory in a separate "process"; without a session id it
        # should see nothing — confirming the previous turn cannot leak.
        mem2 = Memory(agent.memory.config, agent.memory.agent_name)
        await mem2.init_db()
        try:
            assert all(m.role == "system" for m in mem2.short_term)
            # Loading the original session does see the message.
            ok = await mem2.load_session(session_id)
            assert ok is True
            user_msgs = [m for m in mem2.short_term if m.role == "user"]
            assert any("scoped message" in m.content for m in user_msgs)
        finally:
            await mem2.close()
            await agent.close()


class TestCallAgentContextVar:
    """``_call_agent_var`` must be per-async-context, not a shared global."""

    def test_default_is_none(self):
        assert get_call_agent() is None

    @pytest.mark.asyncio
    async def test_concurrent_bindings_do_not_clobber(self):
        observed: dict[str, object] = {}

        async def _bind_and_observe(label: str) -> None:
            sentinel = Mock(name=f"agent-{label}")
            token = _call_agent_var.set(sentinel)
            try:
                # Yield to event loop so the sibling task also runs while
                # this task's binding is "active".
                await asyncio.sleep(0)
                observed[label] = get_call_agent()
            finally:
                _call_agent_var.reset(token)

        await asyncio.gather(_bind_and_observe("rain"), _bind_and_observe("fog"))

        # Each task must observe its own binding — proving they didn't
        # clobber each other via shared global.
        assert observed["rain"] is not observed["fog"]
        # After both tasks complete, no binding remains in the outer scope.
        assert get_call_agent() is None


class TestCompactPreservesDirectives:
    """``compact()`` must not erase user rules under the guise of summarisation."""

    @pytest.mark.asyncio
    async def test_directives_appear_in_digest(self, isolated_config, mock_llm, bus, tool_registry):
        from weather_agents.agents.fog import FogAgent

        # Make the LLM summary intentionally bland; the digest should still
        # carry the user directive verbatim via the directive-extraction
        # heuristic.
        mock_llm.complete = AsyncMock(
            return_value=Mock(
                content="- did some chit-chat",
                tool_calls=[],
                model="x",
                usage={},
                reasoning_content=None,
            )
        )
        agent = FogAgent(config=isolated_config, llm=mock_llm, bus=bus, tool_registry=tool_registry)
        await agent.init()
        # Populate enough messages to trip the compaction threshold.
        agent.memory.add_message("user", "禁止使用 emoji")
        agent.memory.add_message("assistant", "ok")
        for i in range(20):
            agent.memory.add_message("user", f"chit-chat {i}")
            agent.memory.add_message("assistant", f"reply {i}")

        await agent.compact(keep_recent=4)

        # The directive must survive in some form (verbatim in the new
        # digest's "Verbatim user directives" block).
        digest = next(
            m for m in agent.memory.short_term if m.role == "system" and "digest" in m.content
        )
        assert "禁止使用 emoji" in digest.content
        await agent.close()

    @pytest.mark.asyncio
    async def test_digest_is_system_role_not_user(
        self, isolated_config, mock_llm, bus, tool_registry
    ):
        """Inserting the digest as user/assistant makes the model 'continue'
        it as if it were unfinished dialogue — the root cause of the
        '之前你说了几次…' rambling. Must be system role."""
        from weather_agents.agents.fog import FogAgent

        mock_llm.complete = AsyncMock(
            return_value=Mock(
                content="- summary",
                tool_calls=[],
                model="x",
                usage={},
                reasoning_content=None,
            )
        )
        agent = FogAgent(config=isolated_config, llm=mock_llm, bus=bus, tool_registry=tool_registry)
        await agent.init()
        for i in range(20):
            agent.memory.add_message("user", f"u{i}")
            agent.memory.add_message("assistant", f"a{i}")

        await agent.compact(keep_recent=4)

        digest_msgs = [
            m for m in agent.memory.short_term if m.role == "system" and "digest" in m.content
        ]
        assert len(digest_msgs) == 1
        # No fake user/assistant pair was synthesized.
        synth_user = [
            m
            for m in agent.memory.short_term
            if m.role == "user" and "Context compressed" in m.content
        ]
        assert synth_user == []
        await agent.close()


class TestAutoCompactThreshold:
    """Auto-compact should only fire near the context limit, not at 75%."""

    @pytest.mark.asyncio
    async def test_threshold_is_at_least_90_percent(
        self, isolated_config, mock_llm, bus, tool_registry, monkeypatch
    ):
        from weather_agents.agents.fog import FogAgent

        agent = FogAgent(config=isolated_config, llm=mock_llm, bus=bus, tool_registry=tool_registry)
        await agent.init()

        # Pretend the model has a 100-token context window.
        monkeypatch.setattr(
            "weather_agents.core.agent.get_model_context_window",
            lambda _model: 100,
            raising=False,
        )
        # Also patch config import path used inside _should_auto_compact.
        from weather_agents.core import config as cfg_mod

        monkeypatch.setattr(cfg_mod, "get_model_context_window", lambda _model: 100)

        # 80% — must NOT auto-compact (used to at 75%).
        agent.memory.get_context_window_usage = lambda: {  # type: ignore[method-assign]
            "estimated_tokens": 80,
            "message_count": 1,
            "total_chars": 80,
            "limit": 100,
        }
        assert agent._should_auto_compact() is False

        # 95% — must auto-compact.
        agent.memory.get_context_window_usage = lambda: {  # type: ignore[method-assign]
            "estimated_tokens": 95,
            "message_count": 1,
            "total_chars": 95,
            "limit": 100,
        }
        assert agent._should_auto_compact() is True
        await agent.close()


class TestDelegateResultFraming:
    """The delegate result returned to the caller's memory must be framed so
    the caller's next LLM round treats it as third-party data, not its own
    voice."""

    def test_wrapper_tag_present_on_success(self):
        from weather_agents.core.agent import TaskResult
        from weather_agents.tools.delegate import create_delegate_tool

        target = Mock()
        target.name = "fog"
        target.display_name = "雾"
        target.emoji = "~"
        target.state = "idle"
        target.bus = Mock()
        target.bus.add_event = Mock()
        target.init = AsyncMock()
        target.execute_task = AsyncMock(
            return_value=TaskResult(success=True, content="research findings here")
        )
        target._set_state = AsyncMock()

        tool = create_delegate_tool({"fog": target})
        result = asyncio.run(tool.execute(agent="fog", task="research"))

        assert "<delegated_response" in result
        assert "</delegated_response>" in result
        assert "research findings here" in result
        # The hint must explicitly steer the caller away from mimicry.
        assert "own voice" in result.lower() or "do not" in result.lower()

    def test_result_under_truncation_cap(self):
        from weather_agents.core.agent import TaskResult
        from weather_agents.tools.delegate import _MAX_RESULT_CHARS, create_delegate_tool

        target = Mock()
        target.name = "rain"
        target.display_name = "雨"
        target.emoji = "/"
        target.state = "idle"
        target.bus = Mock()
        target.bus.add_event = Mock()
        target.init = AsyncMock()
        long_content = "x" * (_MAX_RESULT_CHARS * 3)
        target.execute_task = AsyncMock(return_value=TaskResult(success=True, content=long_content))
        target._set_state = AsyncMock()

        tool = create_delegate_tool({"rain": target})
        result = asyncio.run(tool.execute(agent="rain", task="write"))

        # The wrapper adds ~300 chars of framing on top of the truncated
        # content; keep the total well under 2x the cap.
        assert len(result) < _MAX_RESULT_CHARS * 2
        assert "truncated" in result


class TestSynthesizeDelegationSummary:
    def test_empty(self):
        assert _synthesize_delegation_summary([]) == ""

    def test_all_success(self):
        out = _synthesize_delegation_summary([("fog", True), ("rain", True)])
        assert "Delegated" in out
        assert "fog" in out
        assert "rain" in out
        assert "Failed" not in out

    def test_mixed(self):
        out = _synthesize_delegation_summary([("fog", True), ("rain", False)])
        assert "Delegated: fog" in out
        assert "Failed: rain" in out


class TestChatBindsContextVar:
    """The non-streaming ``chat()`` entrypoint must bind the ContextVar.

    Without this binding, the one-shot CLI path (``wa <agent> "msg"``)
    leaves tool handlers (``use_skill``, ``list_skills``) with no agent
    reference and they return "no active agent".
    """

    @pytest.mark.asyncio
    async def test_chat_binds_call_agent(self, isolated_config, mock_llm, bus, tool_registry):
        from weather_agents.agents.fog import FogAgent

        seen: dict[str, object] = {}

        async def _capture_complete(*args, **kwargs):
            seen["agent"] = get_call_agent()
            return Mock(content="ok", tool_calls=[], model="x", usage={}, reasoning_content=None)

        mock_llm.complete = _capture_complete
        agent = FogAgent(config=isolated_config, llm=mock_llm, bus=bus, tool_registry=tool_registry)
        await agent.init()
        await agent.chat("hi")
        assert seen["agent"] is agent
        # Binding cleaned up after return.
        assert get_call_agent() is None
        await agent.close()


class TestClearShortTermScope:
    """``clear_short_term`` must NOT delete persisted rows belonging to other sessions."""

    @pytest.mark.asyncio
    async def test_clear_does_not_touch_other_sessions(self, tmp_path):
        from weather_agents.core.config import MemoryConfig
        from weather_agents.core.memory import Memory

        cfg = MemoryConfig(db_path=str(tmp_path / "clr.db"), short_term_limit=50)
        mem = Memory(cfg, "fog")
        await mem.init_db()
        try:
            sid_a = await mem.create_session("A")
            mem.add_message("user", "msg-in-A")
            await mem._flush_pending()

            sid_b = await mem.create_session("B")
            mem.add_message("user", "msg-in-B")
            await mem._flush_pending()

            # Active session is B — clearing must leave A untouched.
            await mem.clear_short_term()

            ok = await mem.load_session(sid_a)
            assert ok is True
            user_msgs = [m for m in mem.short_term if m.role == "user"]
            assert any("msg-in-A" in m.content for m in user_msgs), (
                "clear_short_term silently destroyed session A's data"
            )

            # And session B's persisted rows ARE gone.
            ok = await mem.load_session(sid_b)
            assert ok is True
            user_msgs_b = [m for m in mem.short_term if m.role == "user"]
            assert not any("msg-in-B" in m.content for m in user_msgs_b)
        finally:
            await mem.close()


class TestAutoCompactErrorIsRecoverable:
    """A flaky summariser LLM call must not strand the user's input."""

    @pytest.mark.asyncio
    async def test_compact_failure_does_not_propagate_from_chat(
        self, isolated_config, mock_llm, bus, tool_registry, monkeypatch
    ):
        from weather_agents.agents.fog import FogAgent

        agent = FogAgent(config=isolated_config, llm=mock_llm, bus=bus, tool_registry=tool_registry)
        await agent.init()

        # Make compact() blow up unconditionally.
        async def _explode() -> str:
            raise RuntimeError("summariser 503")

        agent.compact = _explode  # type: ignore[method-assign]
        # And force the threshold check to fire.
        agent._should_auto_compact = lambda: True  # type: ignore[method-assign]

        # chat() should complete normally despite the compact failure —
        # logged, not raised.
        resp = await agent.chat("anything")
        assert resp == "test response"
        await agent.close()


class TestConftestIsolation:
    """The ``app_config`` fixture must point at the per-test tmp dir, not
    the user's real ``~/.weather-agents/memory.db``."""

    def test_app_config_uses_tmp_path(self, app_config, tmp_path):
        assert str(tmp_path) in app_config.memory.db_path
        assert "~/.weather-agents" not in app_config.memory.db_path

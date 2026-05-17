"""Tests for cross-process session resume — fixes the 'amnesia on reconnect' bug.

Before the fix, `wa chat` always created a new session per process, so users'
follow-up turns saw an empty short_term. Now `BaseAgent.init()` prefers to
resume this agent's most recent session.
"""

from __future__ import annotations

import os

import pytest

from weather_agents.core.config import AppConfig, MemoryConfig


@pytest.fixture
def shared_db_config(tmp_path):
    cfg = AppConfig()
    cfg.memory = MemoryConfig(db_path=str(tmp_path / "shared.db"), short_term_limit=50)
    return cfg


class TestResume:
    @pytest.mark.asyncio
    async def test_second_agent_resumes_previous_session(
        self, shared_db_config, mock_llm, bus, tool_registry
    ):
        from weather_agents.agents.fog import FogAgent

        # First "process": persist a user message.
        a1 = FogAgent(config=shared_db_config, llm=mock_llm, bus=bus, tool_registry=tool_registry)
        await a1.init()
        sid1 = a1.memory.get_active_session()
        a1.memory.add_message("user", "remember this please")
        await a1.memory._flush_pending()
        await a1.close()

        # Second "process": should resume sid1, not create a new session.
        a2 = FogAgent(config=shared_db_config, llm=mock_llm, bus=bus, tool_registry=tool_registry)
        await a2.init()
        try:
            assert a2.memory.get_active_session() == sid1
            user_msgs = [m for m in a2.memory.short_term if m.role == "user"]
            assert any("remember this please" in m.content for m in user_msgs)
        finally:
            await a2.close()

    @pytest.mark.asyncio
    async def test_wa_no_resume_env_disables_resume(
        self, shared_db_config, mock_llm, bus, tool_registry, monkeypatch
    ):
        from weather_agents.agents.fog import FogAgent

        a1 = FogAgent(config=shared_db_config, llm=mock_llm, bus=bus, tool_registry=tool_registry)
        await a1.init()
        sid1 = a1.memory.get_active_session()
        await a1.close()

        monkeypatch.setenv("WA_NO_RESUME", "1")
        a2 = FogAgent(config=shared_db_config, llm=mock_llm, bus=bus, tool_registry=tool_registry)
        await a2.init()
        try:
            assert a2.memory.get_active_session() != sid1
        finally:
            await a2.close()

    @pytest.mark.asyncio
    async def test_first_run_still_creates_session(
        self, shared_db_config, mock_llm, bus, tool_registry
    ):
        from weather_agents.agents.fog import FogAgent

        agent = FogAgent(config=shared_db_config, llm=mock_llm, bus=bus, tool_registry=tool_registry)
        await agent.init()
        try:
            assert agent.memory.get_active_session() is not None
        finally:
            await agent.close()

    @pytest.mark.asyncio
    async def test_resume_does_not_cross_agents(
        self, shared_db_config, mock_llm, bus, tool_registry
    ):
        """Fog's resume must not pick up Rain's session."""
        from weather_agents.agents.fog import FogAgent
        from weather_agents.agents.rain import RainAgent

        # Make sure resume is on for both, regardless of test order.
        os.environ.pop("WA_NO_RESUME", None)

        fog = FogAgent(config=shared_db_config, llm=mock_llm, bus=bus, tool_registry=tool_registry)
        await fog.init()
        fog_sid = fog.memory.get_active_session()
        fog.memory.add_message("user", "fog-only memory")
        await fog.memory._flush_pending()
        await fog.close()

        rain = RainAgent(config=shared_db_config, llm=mock_llm, bus=bus, tool_registry=tool_registry)
        await rain.init()
        try:
            # Rain should NOT inherit Fog's session.
            assert rain.memory.get_active_session() != fog_sid
            user_msgs = [m.content for m in rain.memory.short_term if m.role == "user"]
            assert not any("fog-only memory" in c for c in user_msgs)
        finally:
            await rain.close()

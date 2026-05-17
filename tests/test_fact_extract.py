"""Tests for automatic fact extraction.

Implements the 'durable long-term memory accumulates without user action'
piece of the memory design: every N turns the agent triggers a
fire-and-forget LLM pass that pulls user preferences out of the recent
conversation and writes them to long-term memory.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

# ── _parse_extracted_facts ──────────────────────────────────────────────


class TestParseExtractedFacts:
    """The parser must survive everything an LLM can throw at it."""

    def test_direct_json_array(self):
        from weather_agents.core.agent import BaseAgent

        out = BaseAgent._parse_extracted_facts(
            '[{"key": "pkg_mgr", "value": "pnpm", "category": "user_pref"}]'
        )
        assert out == [{"key": "pkg_mgr", "value": "pnpm", "category": "user_pref"}]

    def test_markdown_fenced_json(self):
        from weather_agents.core.agent import BaseAgent

        out = BaseAgent._parse_extracted_facts(
            "Sure, here are the facts:\n"
            '```json\n[{"key": "lang", "value": "Python"}]\n```\n'
            "Hope that helps!"
        )
        assert out == [{"key": "lang", "value": "Python"}]

    def test_raw_json_array_embedded_in_prose(self):
        from weather_agents.core.agent import BaseAgent

        out = BaseAgent._parse_extracted_facts(
            'I found: [{"key": "framework", "value": "FastAPI"}] please review.'
        )
        assert out == [{"key": "framework", "value": "FastAPI"}]

    def test_empty_string_returns_empty(self):
        from weather_agents.core.agent import BaseAgent

        assert BaseAgent._parse_extracted_facts("") == []
        assert BaseAgent._parse_extracted_facts("   ") == []

    def test_garbage_returns_empty(self):
        from weather_agents.core.agent import BaseAgent

        assert BaseAgent._parse_extracted_facts("Sorry, I cannot help.") == []
        assert BaseAgent._parse_extracted_facts("{not: valid json}") == []

    def test_non_list_object_returns_empty(self):
        from weather_agents.core.agent import BaseAgent

        # An object instead of an array — be strict.
        assert BaseAgent._parse_extracted_facts('{"key": "x", "value": "y"}') == []

    def test_drops_non_dict_items(self):
        from weather_agents.core.agent import BaseAgent

        out = BaseAgent._parse_extracted_facts('[{"key": "ok", "value": "v"}, "garbage", 42]')
        assert out == [{"key": "ok", "value": "v"}]


# ── _extract_facts_async: end-to-end with mocked LLM ────────────────────


def _build_test_agent(tmp_path):
    """Build a minimal agent backed by real Memory + a mocked LLM client."""
    from weather_agents.agents.fog import FogAgent
    from weather_agents.core.bus import MessageBus
    from weather_agents.core.config import AppConfig, MemoryConfig
    from weather_agents.core.tool import ToolRegistry

    cfg = AppConfig()
    cfg.memory = MemoryConfig(db_path=str(tmp_path / "extract.db"), short_term_limit=50)
    llm = MagicMock()
    llm.complete = AsyncMock()
    bus = MessageBus()
    tr = ToolRegistry()
    return FogAgent(config=cfg, llm=llm, bus=bus, tool_registry=tr), llm


class TestExtractFactsAsync:
    @pytest.mark.asyncio
    async def test_extracts_and_writes_to_long_term(self, tmp_path):
        agent, llm = _build_test_agent(tmp_path)
        try:
            await agent.init()
            agent.memory.add_message("user", "我用 pnpm 装依赖")
            agent.memory.add_message("assistant", "好的")
            agent.memory.add_message("user", "项目语言是 Python")
            agent.memory.add_message("assistant", "了解")
            await agent.memory._flush_pending()

            llm.complete.return_value = MagicMock(
                content='[{"key": "pkg_mgr", "value": "pnpm", "category": "user_pref"},'
                ' {"key": "project_lang", "value": "Python", "category": "project"}]'
            )

            written = await agent._extract_facts_async()
            assert written == 2

            facts = await agent.memory.recall(limit=20)
            keys = {f["key"] for f in facts}
            assert "pkg_mgr" in keys
            assert "project_lang" in keys
        finally:
            await agent.close()

    @pytest.mark.asyncio
    async def test_too_few_messages_skips_llm(self, tmp_path):
        agent, llm = _build_test_agent(tmp_path)
        try:
            await agent.init()
            # Only one turn — should skip extraction entirely.
            agent.memory.add_message("user", "hi")
            written = await agent._extract_facts_async()
            assert written == 0
            llm.complete.assert_not_called()
        finally:
            await agent.close()

    @pytest.mark.asyncio
    async def test_llm_exception_returns_zero_not_raise(self, tmp_path):
        agent, llm = _build_test_agent(tmp_path)
        try:
            await agent.init()
            for i in range(6):
                agent.memory.add_message("user" if i % 2 == 0 else "assistant", f"m{i}")
            llm.complete.side_effect = RuntimeError("network down")
            written = await agent._extract_facts_async()
            assert written == 0
        finally:
            await agent.close()

    @pytest.mark.asyncio
    async def test_invalid_json_extracts_nothing(self, tmp_path):
        agent, llm = _build_test_agent(tmp_path)
        try:
            await agent.init()
            for i in range(6):
                agent.memory.add_message("user" if i % 2 == 0 else "assistant", f"m{i}")
            llm.complete.return_value = MagicMock(content="Sorry, I cannot help.")
            written = await agent._extract_facts_async()
            assert written == 0
        finally:
            await agent.close()

    @pytest.mark.asyncio
    async def test_drops_facts_with_empty_key_or_value(self, tmp_path):
        agent, llm = _build_test_agent(tmp_path)
        try:
            await agent.init()
            for i in range(6):
                agent.memory.add_message("user" if i % 2 == 0 else "assistant", f"m{i}")
            llm.complete.return_value = MagicMock(
                content='[{"key": "", "value": "x"}, {"key": "y", "value": ""},'
                ' {"key": "good", "value": "v"}]'
            )
            written = await agent._extract_facts_async()
            assert written == 1
            facts = await agent.memory.recall(limit=20)
            assert any(f["key"] == "good" for f in facts)
            assert not any(f["key"] == "" for f in facts)
        finally:
            await agent.close()


# ── _maybe_extract_facts: turn-count gate ───────────────────────────────


class TestMaybeExtractFactsGate:
    """The counter must only fire every N turns, and env-vars must disable."""

    @pytest.mark.asyncio
    async def test_does_not_fire_before_threshold(self, tmp_path, monkeypatch):
        agent, _ = _build_test_agent(tmp_path)
        monkeypatch.setenv("WA_EXTRACT_EVERY_N", "5")
        try:
            await agent.init()
            for _ in range(4):
                agent._maybe_extract_facts()
            # No async task should have been spawned yet.
            assert not agent._pending_extracts
        finally:
            await agent.close()

    @pytest.mark.asyncio
    async def test_fires_at_threshold(self, tmp_path, monkeypatch):
        agent, llm = _build_test_agent(tmp_path)
        llm.complete = AsyncMock(return_value=MagicMock(content="[]"))
        monkeypatch.setenv("WA_EXTRACT_EVERY_N", "3")
        try:
            await agent.init()
            for _ in range(3):
                agent._maybe_extract_facts()
            # A task was scheduled — wait for it to drain.
            tasks = list(agent._pending_extracts)
            assert tasks, "expected an extraction task to be scheduled"
            import asyncio as _asyncio

            await _asyncio.gather(*tasks, return_exceptions=True)
        finally:
            await agent.close()

    @pytest.mark.asyncio
    async def test_wa_no_extract_env_disables(self, tmp_path, monkeypatch):
        agent, _ = _build_test_agent(tmp_path)
        monkeypatch.setenv("WA_NO_EXTRACT", "1")
        monkeypatch.setenv("WA_EXTRACT_EVERY_N", "1")  # would otherwise fire immediately
        try:
            await agent.init()
            agent._maybe_extract_facts()
            assert not agent._pending_extracts
        finally:
            await agent.close()

    @pytest.mark.asyncio
    async def test_every_n_zero_disables(self, tmp_path, monkeypatch):
        agent, _ = _build_test_agent(tmp_path)
        monkeypatch.setenv("WA_EXTRACT_EVERY_N", "0")
        try:
            await agent.init()
            for _ in range(20):
                agent._maybe_extract_facts()
            assert not agent._pending_extracts
        finally:
            await agent.close()

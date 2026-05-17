"""Tests for the memory system."""

from __future__ import annotations

import pytest

from weather_agents.core.config import MemoryConfig
from weather_agents.core.memory import Memory, Message


@pytest.fixture
def memory_config(tmp_path):
    return MemoryConfig(db_path=str(tmp_path / "test_memory.db"), short_term_limit=10)


@pytest.fixture
async def mem(memory_config):
    m = Memory(memory_config, "test_agent")
    await m.init_db()
    yield m
    await m.close()


class TestShortTermMemory:
    @pytest.mark.asyncio
    async def test_add_and_get_messages(self, mem):
        mem.add_message("user", "hello")
        mem.add_message("assistant", "hi there")
        msgs = mem.get_messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["content"] == "hi there"

    @pytest.mark.asyncio
    async def test_system_message_preserved_on_truncation(self, mem):
        mem.add_message("system", "you are an agent")
        for i in range(15):
            mem.add_message("user", f"msg {i}")
        # System message should be preserved
        msgs = mem.get_messages()
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "you are an agent"

    @pytest.mark.asyncio
    async def test_clear_short_term(self, mem):
        mem.add_message("system", "system prompt")
        mem.add_message("user", "hello")
        mem.add_message("assistant", "hi")

        await mem.clear_short_term()
        msgs = mem.get_messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"

    @pytest.mark.asyncio
    async def test_context_window_usage(self, mem):
        mem.add_message("user", "hello world")
        stats = mem.get_context_window_usage()
        assert stats["message_count"] == 1
        assert stats["total_chars"] == 11
        assert stats["limit"] == 10

    @pytest.mark.asyncio
    async def test_tool_message(self, mem):
        mem.add_message(
            "assistant",
            "",
            tool_calls=[
                {
                    "id": "tc_123",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        )
        mem.add_message("tool", "file contents", name="read_file", tool_call_id="tc_123")
        msgs = mem.get_messages()
        assert msgs[1]["name"] == "read_file"
        assert msgs[1]["tool_call_id"] == "tc_123"


class TestWorkingMemory:
    @pytest.mark.asyncio
    async def test_set_get_working(self, mem):
        mem.set_working("task", {"id": "1"})
        assert mem.get_working("task") == {"id": "1"}
        assert mem.get_working("nonexistent", "default") == "default"

    @pytest.mark.asyncio
    async def test_clear_working(self, mem):
        mem.set_working("key", "value")
        mem.clear_working()
        assert mem.get_working("key") is None


class TestLongTermMemory:
    @pytest.mark.asyncio
    async def test_remember_and_recall(self, mem):
        await mem.remember("user_preference", {"theme": "dark"})
        results = await mem.recall("user_preference")
        assert len(results) == 1
        assert results[0]["value"] == {"theme": "dark"}

    @pytest.mark.asyncio
    async def test_remember_with_category(self, mem):
        await mem.remember("api_endpoint", "https://api.example.com", category="config")
        results = await mem.recall(category="config")
        assert len(results) == 1
        assert results[0]["category"] == "config"

    @pytest.mark.asyncio
    async def test_remember_updates_existing(self, mem):
        await mem.remember("key", "value1")
        await mem.remember("key", "value2")
        results = await mem.recall("key")
        assert len(results) == 1
        assert results[0]["value"] == "value2"

    @pytest.mark.asyncio
    async def test_forget(self, mem):
        await mem.remember("temp", "data")
        await mem.forget("temp")
        results = await mem.recall("temp")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_memory_stats(self, mem):
        await mem.remember("k1", "v1", category="config")
        await mem.remember("k2", "v2", category="config")
        await mem.remember("k3", "v3", category="notes")

        stats = await mem.get_memory_stats()
        assert stats["total"] == 3
        assert stats["categories"]["config"] == 2
        assert stats["categories"]["notes"] == 1

    @pytest.mark.asyncio
    async def test_recall_with_limit(self, mem):
        for i in range(10):
            await mem.remember(f"item_{i}", f"value_{i}")
        results = await mem.recall(limit=3)
        assert len(results) == 3


class TestSessionManagement:
    @pytest.mark.asyncio
    async def test_create_session(self, mem):
        sid = await mem.create_session("test session")
        assert len(sid) == 12
        assert mem.get_active_session() == sid

    @pytest.mark.asyncio
    async def test_list_sessions(self, mem):
        sid1 = await mem.create_session("session A")
        sid2 = await mem.create_session("session B")
        sessions = await mem.list_sessions()
        assert len(sessions) == 2
        ids = [s["id"] for s in sessions]
        assert sid1 in ids
        assert sid2 in ids

    @pytest.mark.asyncio
    async def test_load_session(self, mem):
        sid = await mem.create_session("test")
        mem.add_message("user", "hello in session")
        await mem._flush_pending()

        # Switch to a new session, verify old message not in short term
        sid2 = await mem.create_session("another")
        assert mem.get_active_session() == sid2
        msgs = [m for m in mem.short_term if m.role != "system"]
        assert len(msgs) == 0

        # Load back the original session
        ok = await mem.load_session(sid)
        assert ok is True
        msgs = [m for m in mem.short_term if m.role == "user"]
        assert len(msgs) == 1
        assert msgs[0].content == "hello in session"

    @pytest.mark.asyncio
    async def test_load_nonexistent_session(self, mem):
        ok = await mem.load_session("nonexistent")
        assert ok is False

    @pytest.mark.asyncio
    async def test_delete_session(self, mem):
        sid = await mem.create_session("to_delete")
        mem.add_message("user", "test message")
        await mem._flush_pending()

        ok = await mem.delete_session(sid)
        assert ok is True
        sessions = await mem.list_sessions()
        assert len(sessions) == 0
        assert mem.get_active_session() is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_session(self, mem):
        ok = await mem.delete_session("nonexistent")
        assert ok is False

    @pytest.mark.asyncio
    async def test_messages_have_session_id(self, mem):
        sid = await mem.create_session("with_msgs")
        mem.add_message("user", "user msg")
        mem.add_message("assistant", "assistant msg")
        await mem._flush_pending()

        # Load session in a new mem instance
        mem2 = Memory(mem.config, mem.agent_name)
        await mem2.init_db()
        await mem2.load_session(sid)
        try:
            msgs = mem2.get_messages()
            user_msgs = [m for m in msgs if m["role"] == "user"]
            assert len(user_msgs) == 1
            assert user_msgs[0]["content"] == "user msg"
        finally:
            await mem2.close()

    @pytest.mark.asyncio
    async def test_update_session_preview(self, mem):
        _sid = await mem.create_session()
        mem.add_message("user", "This is the first user message for preview")
        await mem._flush_pending()
        await mem.update_session_preview()

        sessions = await mem.list_sessions()
        assert sessions[0]["preview"] == "This is the first user message for preview"

    @pytest.mark.asyncio
    async def test_delete_active_session_resets(self, mem):
        sid = await mem.create_session("active")
        mem.add_message("system", "test system")
        mem.add_message("user", "test user")
        await mem._flush_pending()

        ok = await mem.delete_session(sid)
        assert ok is True
        assert mem.get_active_session() is None
        # Short term should only have system messages
        for m in mem.short_term:
            assert m.role == "system"


class TestDanglingToolCallPruning:
    def test_no_prune_on_clean_messages(self):
        """Messages with no tool calls or with matched tool calls are untouched."""
        from weather_agents.core.memory import Memory, MemoryConfig

        mem = Memory(MemoryConfig(db_path=":memory:", max_persisted_messages=100), "test")
        mem.short_term = [
            Message(role="system", content="base"),
            Message(role="user", content="hello"),
            Message(
                role="assistant",
                content="",
                tool_calls=[{"id": "tc_1", "function": {"name": "echo", "arguments": "{}"}}],
            ),
            Message(role="tool", content="result", tool_call_id="tc_1"),
            Message(role="assistant", content="done"),
        ]
        mem._prune_dangling_tool_calls()
        assert len(mem.short_term) == 5

    def test_prune_removes_orphaned_tool_calls(self):
        """Assistant message with tool_calls missing responses is pruned."""
        from weather_agents.core.memory import Memory, MemoryConfig

        mem = Memory(MemoryConfig(db_path=":memory:", max_persisted_messages=100), "test")
        mem.short_term = [
            Message(role="system", content="base"),
            Message(role="user", content="hello"),
            Message(
                role="assistant",
                content="",
                tool_calls=[{"id": "tc_dangle", "function": {"name": "echo", "arguments": "{}"}}],
            ),
            Message(role="user", content="next message"),
        ]
        mem._prune_dangling_tool_calls()
        assert len(mem.short_term) == 3  # assistant removed
        roles = [m.role for m in mem.short_term]
        assert roles == ["system", "user", "user"]

    def test_prune_partial_tool_calls(self):
        """Assistant with mix of responded + unresponded calls is fully removed."""
        from weather_agents.core.memory import Memory, MemoryConfig

        mem = Memory(MemoryConfig(db_path=":memory:", max_persisted_messages=100), "test")
        mem.short_term = [
            Message(role="system", content="base"),
            Message(role="user", content="run two"),
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    {"id": "tc_ok", "function": {"name": "read", "arguments": "{}"}},
                    {"id": "tc_bad", "function": {"name": "write", "arguments": "{}"}},
                ],
            ),
            Message(role="tool", content="ok", tool_call_id="tc_ok"),
            # tc_bad has no response → assistant should be pruned
            Message(role="assistant", content="after"),
        ]
        mem._prune_dangling_tool_calls()
        # The assistant with mixed calls is removed
        roles = [m.role for m in mem.short_term]
        assert "assistant" not in roles or all(
            m.role != "assistant" or not m.tool_calls for m in mem.short_term
        )

    def test_prune_duplicate_tool_call_ids(self):
        """Duplicate tool_call_ids across assistants — each needs its own response."""
        from weather_agents.core.memory import Memory, MemoryConfig

        mem = Memory(MemoryConfig(db_path=":memory:", max_persisted_messages=100), "test")
        mem.short_term = [
            Message(role="system", content="base"),
            Message(role="user", content="read file"),
            Message(
                role="assistant",
                content="",
                tool_calls=[{"id": "call_X", "function": {"name": "read_file", "arguments": "{}"}}],
            ),
            Message(role="tool", content="result", tool_call_id="call_X"),
            Message(role="assistant", content="Here is the result"),
            Message(role="user", content="write file"),
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    {"id": "call_X", "function": {"name": "write_file", "arguments": "{}"}}
                ],
            ),
            # No tool response for the second call_X → second assistant should be pruned
            Message(role="user", content="next"),
        ]
        mem._prune_dangling_tool_calls()
        roles = [m.role for m in mem.short_term]
        assert roles == ["system", "user", "assistant", "tool", "assistant", "user", "user"]


class TestMessageDataclass:
    def test_message_defaults(self):
        msg = Message(role="user", content="hello")
        assert msg.name is None
        assert msg.tool_call_id is None

    def test_message_with_tool_info(self):
        msg = Message(role="tool", content="result", name="read_file", tool_call_id="tc_1")
        assert msg.name == "read_file"


class TestGapTruncation:
    """Resume-time gap truncation — prevents the 乱拉取 bug."""

    def _row(self, role: str, content: str, ts: str) -> tuple:
        # Mirrors the 7-tuple shape from the messages SELECT in _load_short_term.
        # (role, content, name, tool_call_id, tool_calls, reasoning_content, created_at)
        return (role, content, None, None, None, None, ts)

    def test_no_gap_keeps_everything(self):
        rows = [
            self._row("assistant", "c", "2026-05-17 10:00:05"),
            self._row("user", "b", "2026-05-17 10:00:03"),
            self._row("assistant", "a", "2026-05-17 10:00:01"),
        ]
        kept = Memory._truncate_at_timestamp_gap(rows, gap_seconds=14400)
        assert len(kept) == 3

    def test_large_gap_truncates_older_messages(self):
        # Newest two are seconds apart; older two are from yesterday.
        rows = [
            self._row("assistant", "today2", "2026-05-17 10:05:00"),
            self._row("user", "today1", "2026-05-17 10:00:00"),
            self._row("assistant", "yesterday2", "2026-05-16 09:00:00"),
            self._row("user", "yesterday1", "2026-05-16 08:55:00"),
        ]
        kept = Memory._truncate_at_timestamp_gap(rows, gap_seconds=14400)  # 4h
        contents = [r[1] for r in kept]
        assert contents == ["today2", "today1"]
        assert "yesterday2" not in contents

    def test_gap_zero_disables_truncation(self):
        rows = [
            self._row("assistant", "now", "2026-05-17 10:00:00"),
            self._row("user", "ancient", "2020-01-01 00:00:00"),
        ]
        kept = Memory._truncate_at_timestamp_gap(rows, gap_seconds=0)
        assert len(kept) == 2

    def test_handles_missing_timestamps(self):
        rows = [
            self._row("assistant", "a", "2026-05-17 10:00:00"),
            self._row("user", "b", None),
        ]
        # Must not crash; behavior is best-effort (keep when ts is missing).
        kept = Memory._truncate_at_timestamp_gap(rows, gap_seconds=14400)
        assert len(kept) >= 1

    @pytest.mark.asyncio
    async def test_resume_does_not_drag_in_old_messages(self, memory_config, monkeypatch):
        """End-to-end: writing messages now + simulating an old message
        long ago should NOT load the old one on resume."""
        # Use 1-second gap so we don't need to actually wait hours.
        monkeypatch.setenv("WA_RESUME_GAP_SECONDS", "1")

        m1 = Memory(memory_config, "agent_a")
        await m1.init_db()
        sid = await m1.create_session("test")

        # Backdate one user message by 1 hour by writing directly to the DB.
        # (mimics yesterday's session)
        assert m1._db is not None
        await m1._db.execute(
            "INSERT INTO messages (agent, role, content, session_id, created_at) "
            "VALUES (?, ?, ?, ?, datetime('now', '-1 hour'))",
            ("agent_a", "user", "stale message from yesterday", sid),
        )
        # Recent message at current time.
        await m1._db.execute(
            "INSERT INTO messages (agent, role, content, session_id, created_at) "
            "VALUES (?, ?, ?, ?, datetime('now'))",
            ("agent_a", "user", "fresh message", sid),
        )
        await m1._db.commit()
        await m1.close()

        # Fresh Memory instance — simulating a new process resuming the session.
        m2 = Memory(memory_config, "agent_a")
        await m2.init_db()
        ok = await m2.load_session(sid)
        try:
            assert ok is True
            user_contents = [msg.content for msg in m2.short_term if msg.role == "user"]
            assert "fresh message" in user_contents
            assert "stale message from yesterday" not in user_contents
        finally:
            await m2.close()

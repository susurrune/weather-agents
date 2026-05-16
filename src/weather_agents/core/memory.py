"""Memory system: short-term context, long-term persistence, working memory."""

from __future__ import annotations

import asyncio
import contextlib
import json
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiosqlite

from weather_agents.core.config import MemoryConfig


class _RetryDB:
    """Wraps aiosqlite connection with automatic retry on SQLITE_BUSY.

    Multiple ``wa`` instances share the same SQLite file.  WAL mode allows
    concurrent reads but writes still serialise — when two processes write
    simultaneously one gets ``database is locked``.  SQLite's built-in
    ``busy_timeout`` handles retry at the C level; this wrapper adds a
    Python-level safety net so that even if the C handler fails, we retry
    gracefully instead of crashing the interactive session.
    """

    def __init__(self, db: aiosqlite.Connection) -> None:
        object.__setattr__(self, "_db", db)

    async def execute(self, sql: str, parameters: Any = None) -> Any:
        for attempt in range(5):
            try:
                if parameters is not None:
                    return await self._db.execute(sql, parameters)
                return await self._db.execute(sql)
            except sqlite3.OperationalError as e:
                if "database is locked" not in str(e):
                    raise
                if attempt < 4:
                    await asyncio.sleep(0.1 * (attempt + 1))
                else:
                    raise

    async def executemany(self, sql: str, parameters: Any) -> None:
        await self._db.executemany(sql, parameters)

    async def commit(self) -> None:
        for attempt in range(5):
            try:
                await self._db.commit()
                return
            except sqlite3.OperationalError as e:
                if "database is locked" not in str(e):
                    raise
                if attempt < 4:
                    await asyncio.sleep(0.1 * (attempt + 1))
                else:
                    raise

    async def close(self) -> None:
        await self._db.close()

    def __getattr__(self, name: str) -> Any:
        """Delegate undecorated attributes (Cursor returns, etc.) directly."""
        return getattr(self._db, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._db, name, value)


@dataclass
class Message:
    role: str
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict] | None = None
    reasoning_content: str | None = None


class Memory:
    """Three-layer memory for each agent with SQLite persistence.

    - Short-term: conversation context (persisted to SQLite)
    - Working: in-memory task-scoped state
    - Long-term: persistent key-value storage with search
    """

    def __init__(self, config: MemoryConfig, agent_name: str) -> None:
        self.config = config
        self.agent_name = agent_name
        self.short_term: list[Message] = []
        self.working: dict[str, Any] = {}
        self._db_path = Path(config.db_path).expanduser()
        self._db: _RetryDB | None = None
        self._loaded = False
        self._pending_persists: set[asyncio.Task] = set()
        self._active_session: str | None = None

    async def init_db(self) -> None:
        if self._db is not None:
            return
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        raw = await aiosqlite.connect(str(self._db_path))
        await raw.execute("PRAGMA journal_mode=WAL")
        await raw.execute("PRAGMA busy_timeout=5000")
        await raw.execute("PRAGMA auto_vacuum=INCREMENTAL")
        self._db = _RetryDB(raw)
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        # Migrate existing DBs that lack the category/updated_at columns
        with contextlib.suppress(Exception):
            await self._db.execute(
                "ALTER TABLE memories ADD COLUMN category TEXT DEFAULT 'general'"
            )
        with contextlib.suppress(Exception):
            await self._db.execute(
                "ALTER TABLE memories ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            )
        # Ensure unique index for agent+key (UPSERT support)
        with contextlib.suppress(Exception):
            await self._db.execute("DROP INDEX idx_agent_key")
        await self._db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_key ON memories(agent, key)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_category ON memories(agent, category)"
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                name TEXT,
                tool_call_id TEXT,
                tool_calls TEXT,
                reasoning_content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_agent ON messages(agent, created_at)"
        )
        with contextlib.suppress(Exception):
            await self._db.execute("ALTER TABLE messages ADD COLUMN tool_calls TEXT")
        with contextlib.suppress(Exception):
            await self._db.execute("ALTER TABLE messages ADD COLUMN reasoning_content TEXT")
        with contextlib.suppress(Exception):
            await self._db.execute("ALTER TABLE messages ADD COLUMN session_id TEXT")
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                agent TEXT NOT NULL,
                name TEXT,
                preview TEXT DEFAULT '',
                message_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_agent ON sessions(agent, updated_at DESC)"
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS working_data (
                agent TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (agent, key)
            )
            """
        )
        await self._db.commit()
        await self._load_short_term()
        await self._load_working()

    async def _load_short_term(self) -> None:
        if not self._db or self._loaded:
            return
        # ORDER BY id DESC (not created_at): id is strictly monotonic even when
        # multiple inserts land in the same second — avoids non-deterministic
        # ordering that could place a tool message before its assistant.
        if self._active_session:
            cursor = await self._db.execute(
                "SELECT role, content, name, tool_call_id, tool_calls, reasoning_content FROM messages "
                "WHERE agent = ? AND session_id = ? ORDER BY id DESC LIMIT ?",
                (self.agent_name, self._active_session, self.config.short_term_limit),
            )
        else:
            # No session — start with a clean slate.  Loading messages from
            # every prior session would present the LLM with a jumble of
            # unrelated conversation fragments and cause the confusion
            # described as "memory chaos".
            self._loaded = True
            self._prune_dangling_tool_calls()
            return
        rows = await cursor.fetchall()
        for row in reversed(list(rows)):
            tool_calls = None
            if len(row) > 4 and row[4]:
                with contextlib.suppress(json.JSONDecodeError, TypeError):
                    tool_calls = json.loads(row[4])
            reasoning_content = None
            if len(row) > 5 and row[5]:
                reasoning_content = row[5]
            self.short_term.append(
                Message(
                    role=row[0],
                    content=row[1],
                    name=row[2],
                    tool_call_id=row[3],
                    tool_calls=tool_calls,
                    reasoning_content=reasoning_content,
                )
            )
        self._loaded = True
        self._prune_dangling_tool_calls()

    def _prune_dangling_tool_calls(self) -> None:
        """Remove orphaned tool_calls/tool message pairs from short-term memory.

        The LLM API requires every 'tool' role message to be preceded by an
        'assistant' message whose tool_calls array contains the matching id.
        Truncation or compaction can break this invariant by removing an
        assistant message while leaving its tool responses behind.

        Uses position-aware matching: each tool message satisfies the *closest*
        preceding assistant that contains its tool_call_id.  This correctly
        handles duplicate tool_call_ids across different assistant messages,
        which the naive set-based approach would conflate.
        """
        if not self.short_term:
            return

        n = len(self.short_term)
        remove = [False] * n

        # ── Pass 1: position-aware matching ──
        # For each tool_call_id, a stack of assistant indices waiting for a
        # response. A tool message pops from the stack — satisfying the most
        # recent (closest) preceding assistant.
        waiting: dict[str, list[int]] = {}
        for i, msg in enumerate(self.short_term):
            if msg.role == "assistant" and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tid := tc.get("id"):
                        waiting.setdefault(tid, []).append(i)
            elif msg.role == "tool" and msg.tool_call_id:
                tid = msg.tool_call_id
                if tid in waiting and waiting[tid]:
                    waiting[tid].pop()  # consumed by closest assistant
                else:
                    remove[i] = True  # orphaned tool message

        # Any assistant indices still in waiting stacks are orphaned.
        for indices in waiting.values():
            for i in indices:
                remove[i] = True

        if not any(remove):
            return

        kept = [m for i, m in enumerate(self.short_term) if not remove[i]]

        # ── Pass 2: remove tool messages with no preceding assistant ──
        seen_tc_ids: set[str] = set()
        sanitized: list[Message] = []
        for msg in kept:
            if msg.role == "assistant" and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tid := tc.get("id"):
                        seen_tc_ids.add(tid)
            elif msg.role == "tool" and msg.tool_call_id and msg.tool_call_id not in seen_tc_ids:
                continue
            sanitized.append(msg)

        self.short_term = sanitized

    def prune_tool_messages(self) -> None:
        """Public wrapper around _prune_dangling_tool_calls."""
        self._prune_dangling_tool_calls()

    async def _flush_pending(self) -> None:
        if self._pending_persists:
            results = await asyncio.gather(*self._pending_persists, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    from weather_agents.core.logger import get_logger

                    get_logger("memory").warning(
                        "flush_persist_failed",
                        extra={"agent": self.agent_name, "error": str(r)},
                    )
            self._pending_persists.clear()
        if self._db:
            with contextlib.suppress(Exception):
                await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._flush_pending()
            await self._db.close()

    # -- Short-term memory (conversation context, persisted) --

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        msg = Message(role=role, content=content)
        if "name" in kwargs:
            msg.name = kwargs["name"]
        if "tool_call_id" in kwargs:
            msg.tool_call_id = kwargs["tool_call_id"]
        if "tool_calls" in kwargs and kwargs["tool_calls"]:
            msg.tool_calls = kwargs["tool_calls"]
        if "reasoning_content" in kwargs and kwargs["reasoning_content"]:
            msg.reasoning_content = kwargs["reasoning_content"]
        self.short_term.append(msg)

        if len(self.short_term) > self.config.short_term_limit:
            system_msgs = [m for m in self.short_term if m.role == "system"]
            other_msgs = [m for m in self.short_term if m.role != "system"]
            keep = max(0, self.config.short_term_limit - len(system_msgs))
            self.short_term = system_msgs + other_msgs[-keep:] if keep else system_msgs
            self._prune_dangling_tool_calls()

        if self._db and role != "system":
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No event loop — skip persistence rather than crash sync callers.
                return
            tool_calls_json = json.dumps(msg.tool_calls) if msg.tool_calls else None
            session_id = self._active_session
            task = loop.create_task(
                self._persist_message(
                    role,
                    content,
                    msg.name,
                    msg.tool_call_id,
                    tool_calls_json,
                    msg.reasoning_content,
                    session_id,
                )
            )
            self._pending_persists.add(task)
            task.add_done_callback(self._pending_persists.discard)

    async def _persist_message(
        self,
        role: str,
        content: str,
        name: str | None,
        tool_call_id: str | None,
        tool_calls: str | None = None,
        reasoning_content: str | None = None,
        session_id: str | None = None,
    ) -> None:
        if not self._db:
            return
        try:
            await self._db.execute(
                "INSERT INTO messages (agent, role, content, name, tool_call_id, tool_calls, reasoning_content, session_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    self.agent_name,
                    role,
                    content,
                    name,
                    tool_call_id,
                    tool_calls,
                    reasoning_content,
                    session_id,
                ),
            )
            if session_id:
                await self._db.execute(
                    "UPDATE sessions SET message_count = message_count + 1, "
                    "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (session_id,),
                )
            await self._db.commit()
            # Auto-prune old messages beyond max_persisted_messages
            max_persisted = getattr(self.config, "max_persisted_messages", 1000)
            cursor = await self._db.execute(
                "SELECT COUNT(*) FROM messages WHERE agent = ? AND role != 'system'",
                (self.agent_name,),
            )
            row = await cursor.fetchone()
            if row and row[0] > max_persisted:
                excess = row[0] - max_persisted
                await self._db.execute(
                    "DELETE FROM messages WHERE id IN ("
                    "SELECT id FROM messages WHERE agent = ? AND role != 'system' "
                    "ORDER BY id ASC LIMIT ?)",
                    (self.agent_name, excess),
                )
                await self._db.commit()
        except Exception as e:
            from weather_agents.core.logger import get_logger

            get_logger("memory").warning(
                "persist_message_failed",
                extra={"agent": self.agent_name, "error": str(e)},
            )

    def get_messages(self) -> list[dict]:
        self._prune_dangling_tool_calls()
        msgs = []
        for m in self.short_term:
            d: dict[str, Any] = {"role": m.role, "content": m.content}
            if m.name:
                d["name"] = m.name
            if m.tool_call_id:
                d["tool_call_id"] = m.tool_call_id
            if m.tool_calls:
                d["tool_calls"] = m.tool_calls
            if m.reasoning_content:
                d["reasoning_content"] = m.reasoning_content
            msgs.append(d)
        return msgs

    def get_context_window_usage(self) -> dict:
        """Return stats about current memory usage."""
        total_chars = sum(len(m.content) for m in self.short_term)
        cjk = sum(
            1 for m in self.short_term for c in m.content if "一" <= c <= "鿿" or "　" <= c <= "〿"
        )
        other = total_chars - cjk
        return {
            "message_count": len(self.short_term),
            "total_chars": total_chars,
            "estimated_tokens": max(1, cjk * 2 + other // 4),
            "limit": self.config.short_term_limit,
        }

    async def clear_short_term(self) -> None:
        """Clear in-memory short-term and delete the persisted rows for the
        *active* session only.

        Previously this DELETE had no ``session_id`` filter and wiped every
        saved session's messages for this agent — silent data loss when
        users meant to clear the current conversation. If no session is
        active, this falls back to the legacy behaviour but only for rows
        with NULL session_id so saved sessions are preserved.
        """
        system_msgs = [m for m in self.short_term if m.role == "system"]
        self.short_term = system_msgs
        if not self._db:
            return
        if self._active_session is not None:
            await self._db.execute(
                "DELETE FROM messages WHERE agent = ? AND role != 'system' AND session_id = ?",
                (self.agent_name, self._active_session),
            )
            # Reset the session's message_count so /list shows it correctly.
            await self._db.execute(
                "UPDATE sessions SET message_count = 0 WHERE id = ?",
                (self._active_session,),
            )
        else:
            await self._db.execute(
                "DELETE FROM messages WHERE agent = ? AND role != 'system' AND session_id IS NULL",
                (self.agent_name,),
            )
        await self._db.commit()

    # -- Working memory (task-scoped, persisted to SQLite) --

    async def _load_working(self) -> None:
        """Restore working memory from the database on startup."""
        if not self._db:
            return
        cursor = await self._db.execute(
            "SELECT key, value FROM working_data WHERE agent = ?",
            (self.agent_name,),
        )
        rows = await cursor.fetchall()
        for key, value in rows:
            with contextlib.suppress(json.JSONDecodeError):
                self.working[key] = json.loads(value)

    def _schedule_persist_working(self) -> None:
        """Fire-and-forget persist of the full working dict to SQLite."""
        if not self._db:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._persist_working())
        self._pending_persists.add(task)
        task.add_done_callback(self._pending_persists.discard)

    async def _persist_working(self) -> None:
        """Write all working data to the database (UPSERT)."""
        if not self._db:
            return
        try:
            await self._db.execute(
                "DELETE FROM working_data WHERE agent = ?",
                (self.agent_name,),
            )
            for key, value in self.working.items():
                await self._db.execute(
                    "INSERT INTO working_data (agent, key, value) VALUES (?, ?, ?)",
                    (self.agent_name, key, json.dumps(value, ensure_ascii=False)),
                )
            await self._db.commit()
        except Exception as e:
            try:
                from weather_agents.core.logger import get_logger

                get_logger("memory").warning(
                    "persist_working_failed",
                    extra={"agent": self.agent_name, "error": str(e)},
                )
            except ImportError:
                pass

    def set_working(self, key: str, value: Any) -> None:
        self.working[key] = value
        self._schedule_persist_working()

    def get_working(self, key: str, default: Any = None) -> Any:
        return self.working.get(key, default)

    def clear_working(self) -> None:
        self.working.clear()
        self._schedule_persist_working()

    # -- Long-term memory (persistent key-value with categories) --

    async def remember(self, key: str, value: Any, category: str = "general") -> None:
        if not self._db:
            return
        await self._db.execute(
            "INSERT INTO memories (agent, key, value, category) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(agent, key) DO UPDATE SET "
            "value = excluded.value, category = excluded.category, updated_at = CURRENT_TIMESTAMP",
            (self.agent_name, key, json.dumps(value, ensure_ascii=False), category),
        )
        await self._db.commit()

    async def recall(
        self,
        key: str | None = None,
        category: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        if not self._db:
            return []

        query = "SELECT key, value, category FROM memories WHERE agent = ?"
        params: list[Any] = [self.agent_name]

        if key:
            query += " AND key LIKE ?"
            params.append(f"%{key}%")
        if category:
            query += " AND category = ?"
            params.append(category)

        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)

        cursor = await self._db.execute(query, params)
        rows = await cursor.fetchall()
        return [{"key": r[0], "value": json.loads(r[1]), "category": r[2]} for r in rows]

    async def forget(self, key: str) -> None:
        if not self._db:
            return
        await self._db.execute(
            "DELETE FROM memories WHERE agent = ? AND key = ?",
            (self.agent_name, key),
        )
        await self._db.commit()

    # -- Session management --

    def get_active_session(self) -> str | None:
        return self._active_session

    async def create_session(self, name: str | None = None) -> str:
        session_id = uuid.uuid4().hex[:12]
        preview = name or ""
        if not self._db:
            return session_id
        await self._db.execute(
            "INSERT INTO sessions (id, agent, name, preview) VALUES (?, ?, ?, ?)",
            (session_id, self.agent_name, name, preview),
        )
        await self._db.commit()
        self._active_session = session_id
        self.short_term = [m for m in self.short_term if m.role == "system"]
        self._loaded = True
        return session_id

    async def list_sessions(self) -> list[dict]:
        if not self._db:
            return []
        cursor = await self._db.execute(
            "SELECT id, agent, name, preview, message_count, created_at, updated_at "
            "FROM sessions WHERE agent = ? ORDER BY updated_at DESC LIMIT 50",
            (self.agent_name,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": r[0],
                "agent": r[1],
                "name": r[2],
                "preview": r[3],
                "message_count": r[4],
                "created_at": r[5],
                "updated_at": r[6],
            }
            for r in rows
        ]

    async def load_session(self, session_id: str) -> bool:
        if not self._db:
            return False
        cursor = await self._db.execute(
            "SELECT id FROM sessions WHERE id = ? AND agent = ?",
            (session_id, self.agent_name),
        )
        if not await cursor.fetchone():
            return False
        self._active_session = session_id
        self.short_term = [m for m in self.short_term if m.role == "system"]
        self._loaded = False
        await self._load_short_term()
        return True

    async def delete_session(self, session_id: str) -> bool:
        if not self._db:
            return False
        cursor = await self._db.execute(
            "SELECT id FROM sessions WHERE id = ? AND agent = ?",
            (session_id, self.agent_name),
        )
        if not await cursor.fetchone():
            return False
        await self._db.execute(
            "DELETE FROM messages WHERE agent = ? AND session_id = ?",
            (self.agent_name, session_id),
        )
        await self._db.execute(
            "DELETE FROM sessions WHERE id = ? AND agent = ?",
            (session_id, self.agent_name),
        )
        await self._db.commit()
        if self._active_session == session_id:
            self._active_session = None
            self.short_term = [m for m in self.short_term if m.role == "system"]
            self._loaded = False
            await self._load_short_term()
        return True

    async def update_session_preview(self) -> None:
        """Set preview from the first user message in the active session."""
        if not self._db or not self._active_session:
            return
        cursor = await self._db.execute(
            "SELECT content FROM messages WHERE agent = ? AND session_id = ? AND role = 'user' "
            "ORDER BY id ASC LIMIT 1",
            (self.agent_name, self._active_session),
        )
        row = await cursor.fetchone()
        if row:
            preview = row[0][:80]
            await self._db.execute(
                "UPDATE sessions SET preview = ? WHERE id = ?",
                (preview, self._active_session),
            )
            await self._db.commit()

    async def get_memory_stats(self) -> dict:
        """Return statistics about long-term memory."""
        if not self._db:
            return {"total": 0, "categories": {}}
        cursor = await self._db.execute(
            "SELECT category, COUNT(*) FROM memories WHERE agent = ? GROUP BY category",
            (self.agent_name,),
        )
        rows = await cursor.fetchall()
        categories = {r[0]: r[1] for r in rows}
        return {"total": sum(categories.values()), "categories": categories}

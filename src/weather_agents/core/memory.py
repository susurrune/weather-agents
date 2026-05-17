"""Memory system: short-term context, long-term persistence, working memory."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
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
        # shared_working: cross-agent scratchpad keyed by session_id. Used by
        # multi-step pipelines so a downstream agent can pull the upstream
        # agent's full output via read_shared_memory tool instead of relying
        # on the orchestrator to splice it into the description (which gets
        # truncated at 500 chars).
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS shared_working (
                session_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                written_by TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (session_id, key)
            )
            """
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_shared_session ON shared_working(session_id, updated_at DESC)"
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
                "SELECT role, content, name, tool_call_id, tool_calls, "
                "reasoning_content, created_at FROM messages "
                "WHERE agent = ? AND session_id = ? ORDER BY id DESC LIMIT ?",
                (self.agent_name, self._active_session, self.config.short_term_limit),
            )
        else:
            # No session — start with a clean slate to avoid cross-session
            # leakage between processes. Callers wanting continuity should
            # explicitly call ``resume_latest_session()``.
            self._loaded = True
            self._prune_dangling_tool_calls()
            return

        rows = list(await cursor.fetchall())
        # Conversation-gap truncation: walking from the newest backward, stop
        # as soon as we see a timestamp gap larger than RESUME_GAP_SECONDS.
        # Pre-fix users complained that a fresh `wa chat` would drag in 50
        # messages spanning days of unrelated tasks (the "乱拉取" bug). A 4h
        # gap is a cheap proxy for "different work session"; nothing earlier
        # belongs in the immediate context.
        gap_seconds = float(os.environ.get("WA_RESUME_GAP_SECONDS", "14400"))  # 4h default
        rows = self._truncate_at_timestamp_gap(rows, gap_seconds)

        for row in reversed(rows):
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

    @staticmethod
    def _truncate_at_timestamp_gap(rows: list, gap_seconds: float) -> list:
        """Keep only the contiguous tail of rows where consecutive
        ``created_at`` values are within ``gap_seconds`` of each other.

        ``rows`` is ordered NEWEST first (DESC). We walk from index 0 forward
        looking for the first big gap; everything from that gap onward (older)
        is dropped. The created_at column is the LAST element of each row.
        """
        if not rows or gap_seconds <= 0:
            return rows
        from datetime import datetime

        def parse_ts(raw: Any) -> datetime | None:
            if not raw:
                return None
            if isinstance(raw, datetime):
                return raw
            with contextlib.suppress(ValueError, TypeError):
                # SQLite returns 'YYYY-MM-DD HH:MM:SS' for CURRENT_TIMESTAMP.
                return datetime.fromisoformat(str(raw).replace(" ", "T"))
            return None

        keep = [rows[0]]
        prev_ts = parse_ts(rows[0][-1])
        for row in rows[1:]:
            cur_ts = parse_ts(row[-1])
            if prev_ts and cur_ts:
                delta = abs((prev_ts - cur_ts).total_seconds())
                if delta > gap_seconds:
                    break
            keep.append(row)
            if cur_ts:
                prev_ts = cur_ts
        return keep

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

    # -- Shared working memory (cross-agent, session-scoped) --

    async def write_shared(
        self,
        key: str,
        value: Any,
        *,
        session_id: str | None = None,
    ) -> bool:
        """Store a value in the session-scoped shared scratchpad.

        Any agent in the same session can read it via ``read_shared``.
        Returns False when no session is active and none was supplied —
        shared memory is intentionally session-bound so cross-session
        bleed-through is impossible.
        """
        sid = session_id or self._active_session
        if not self._db or not sid:
            return False
        payload = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        await self._db.execute(
            "INSERT INTO shared_working (session_id, key, value, written_by) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(session_id, key) DO UPDATE SET "
            "value = excluded.value, written_by = excluded.written_by, "
            "updated_at = CURRENT_TIMESTAMP",
            (sid, key, payload, self.agent_name),
        )
        await self._db.commit()
        return True

    async def read_shared(
        self,
        key: str,
        *,
        session_id: str | None = None,
    ) -> Any | None:
        """Read a value from the shared scratchpad. None when missing."""
        sid = session_id or self._active_session
        if not self._db or not sid:
            return None
        cursor = await self._db.execute(
            "SELECT value FROM shared_working WHERE session_id = ? AND key = ?",
            (sid, key),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            return json.loads(row[0])
        return row[0]

    async def list_shared(
        self,
        *,
        session_id: str | None = None,
    ) -> list[dict]:
        """List all shared keys in this session with their writer + timestamp.

        Returns ``[]`` when no session is active. Values are NOT included —
        callers can ``read_shared`` for those once they pick a key.
        """
        sid = session_id or self._active_session
        if not self._db or not sid:
            return []
        cursor = await self._db.execute(
            "SELECT key, written_by, updated_at FROM shared_working "
            "WHERE session_id = ? ORDER BY updated_at DESC",
            (sid,),
        )
        rows = await cursor.fetchall()
        return [{"key": r[0], "written_by": r[1], "updated_at": r[2]} for r in rows]

    async def delete_shared(
        self,
        key: str,
        *,
        session_id: str | None = None,
    ) -> bool:
        """Delete a shared entry. Returns True iff a row was removed."""
        sid = session_id or self._active_session
        if not self._db or not sid:
            return False
        cursor = await self._db.execute(
            "DELETE FROM shared_working WHERE session_id = ? AND key = ?",
            (sid, key),
        )
        await self._db.commit()
        return (cursor.rowcount or 0) > 0

    # -- Retrieval-injection helpers (used by BaseAgent._messages_with_recall) --

    _TOKEN_RE = None  # type: ignore[assignment]  # populated lazily

    @staticmethod
    def _tokenize_for_recall(query: str) -> list[str]:
        """Split `query` into recall tokens.

        Strategy (no LLM, no embeddings):
        - ASCII words (length ≥ 2) come FIRST — they're typically the
          high-signal terms ("pnpm", "FastAPI") and must not get crowded
          out by a flood of CJK ngrams when callers apply a count cap.
        - CJK character runs → emit overlapping 2-grams and 3-grams. Plain
          run-as-token over-matches ("的依赖" instead of "依赖") and ends up
          missing facts stored under a sub-phrase.
        Returns unique tokens preserving the priority above.
        """
        import re

        out: dict[str, None] = {}
        # ASCII first → they're typically the most discriminative tokens.
        for w in re.findall(r"[A-Za-z][A-Za-z0-9_+-]+", query or ""):
            out.setdefault(w, None)
        for run in re.findall(r"[一-鿿]+", query or ""):
            for size in (3, 2):
                if len(run) < size:
                    continue
                for i in range(len(run) - size + 1):
                    out.setdefault(run[i : i + size], None)
        return list(out.keys())

    async def recall_for_injection(self, query: str, limit: int = 3) -> list[dict]:
        """Find long-term facts whose key OR value matches tokens from `query`.

        Goes direct to SQL rather than delegating to ``recall()`` because the
        public ``recall(key=...)`` only LIKEs the key column — facts stored
        under semantic keys (``pkg_mgr=pnpm``) would never be found by their
        value. Returns at most `limit` distinct facts ordered by recency.
        Empty list when the store is empty.

        Performance: assembles all tokens into a single SQL query rather
        than looping (one round-trip instead of N). Caps at 24 tokens to
        keep the LIKE chain from getting absurd on pasted essays.
        """
        if not self._db or not query:
            return []
        tokens = self._tokenize_for_recall(query)[:24]
        if not tokens:
            return []
        # Build a single (key LIKE ? OR value LIKE ?) chain.
        like_clauses = " OR ".join(["key LIKE ? OR value LIKE ?"] * len(tokens))
        params: list[Any] = [self.agent_name]
        for tok in tokens:
            pattern = f"%{tok}%"
            params.append(pattern)
            params.append(pattern)
        params.append(limit)
        cursor = await self._db.execute(
            f"SELECT key, value, category FROM memories "
            f"WHERE agent = ? AND ({like_clauses}) "
            f"ORDER BY updated_at DESC LIMIT ?",
            params,
        )
        rows = await cursor.fetchall()
        out: list[dict] = []
        for r in rows:
            try:
                val = json.loads(r[1])
            except (json.JSONDecodeError, TypeError):
                val = r[1]
            out.append({"key": r[0], "value": val, "category": r[2]})
        return out

    @staticmethod
    def format_facts_block(facts: list[dict]) -> str:
        """Render facts as a compact markdown block suitable for prompt injection.

        Empty / falsy input returns an empty string so callers can ``if block``.
        """
        if not facts:
            return ""
        lines = ["## 相关记忆"]
        for f in facts:
            v = f.get("value")
            if not isinstance(v, str):
                with contextlib.suppress(TypeError, ValueError):
                    v = json.dumps(v, ensure_ascii=False)
            lines.append(f"- **{f.get('key')}**: {v}")
        return "\n".join(lines)

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

    async def resume_latest_session(self) -> str | None:
        """Activate this agent's most recently updated session, if any.

        CLI entry points call this to restore continuity across `wa chat`
        invocations. Without it, every fresh process starts amnesic — which
        produced the "memory chaos" complaint where users felt their agents
        forgot everything between turns.
        """
        if not self._db or self._active_session:
            return self._active_session
        cursor = await self._db.execute(
            "SELECT id FROM sessions WHERE agent = ? ORDER BY updated_at DESC LIMIT 1",
            (self.agent_name,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        self._active_session = row[0]
        self._loaded = False
        # short_term might already hold a system prompt — keep it, reload the rest.
        self.short_term = [m for m in self.short_term if m.role == "system"]
        await self._load_short_term()
        return self._active_session

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

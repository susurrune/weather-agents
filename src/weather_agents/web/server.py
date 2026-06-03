"""WebSocket voice server for remote voice conversation with agents."""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import ssl
from pathlib import Path
from typing import Any

from aiohttp import web

from weather_agents.core.agent import BaseAgent
from weather_agents.core.factory import SystemContext
from weather_agents.core.logger import get_logger as _get_logger
from weather_agents.web.tts import DoubaoTTS

HERE = Path(__file__).parent

_HTML_PATH = HERE / "client.html"

_log = _get_logger("voice")


class VoiceServer:
    """aiohttp WebSocket server bridging browser voice ↔ agent chat_stream.

    Each WebSocket connection gets its own memory session so conversations
    are isolated.  HTTP ``GET /`` serves the single-page voice client HTML.
    HTTP ``GET /health`` returns a simple health-check JSON.

    Supports switching agents at runtime via WebSocket messages:
    ``{"type":"list_agents"}`` and ``{"type":"switch_agent","agent":"<name>"}``.
    """

    _html_content: str
    _html_etag: str
    _IDLE_TIMEOUT: int = 300  # close WS after 5 min of silence

    def __init__(
        self,
        agent_map: dict[str, BaseAgent],
        system_ctx: SystemContext,
        *,
        agent_name: str = "fair",
        host: str = "0.0.0.0",
        port: int = 8765,
        tts_engine: DoubaoTTS | None = None,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        self._agent_map = agent_map
        self._current_agent_name = agent_name
        self.ctx = system_ctx
        self.host = host
        self.port = port
        self.tts_engine = tts_engine
        self.ssl_context = ssl_context
        self._app = web.Application()

        # Serialize all _handle_ws calls so concurrent connections to the same
        # agent don't race on memory._active_session. Per-agent locks would be
        # more concurrent but break on agent switch (old lock still held when
        # accessing the new agent's memory in the switch handler).
        self._ws_lock = asyncio.Lock()

        # Cache HTML at startup — avoid disk I/O on every request.
        if _HTML_PATH.is_file():
            self._html_content = _HTML_PATH.read_text(encoding="utf-8")
            import hashlib

            # ETag — collision-resistance is the only requirement, no
            # security context. ``usedforsecurity=False`` satisfies FIPS
            # builds where md5 is disabled by default.
            self._html_etag = hashlib.md5(
                self._html_content.encode(), usedforsecurity=False
            ).hexdigest()[:16]
        else:
            self._html_content = "<h1>Voice client not found</h1>"
            self._html_etag = "not-found"

        self._app.router.add_get("/", self._handle_index)
        self._app.router.add_get("/health", self._handle_health)
        self._app.router.add_get("/qr", self._handle_qr)
        # PWA: installable "add to home screen" app on phones.
        self._app.router.add_get("/manifest.webmanifest", self._handle_manifest)
        self._app.router.add_get("/sw.js", self._handle_sw)
        self._app.router.add_get("/icon.svg", self._handle_icon)
        self._app.router.add_get("/ws", self._handle_ws)

    @property
    def agent(self) -> BaseAgent:
        """Return the currently active agent."""
        return self._agent_map[self._current_agent_name]

    def _build_agent_list(self) -> list[dict[str, str]]:
        """Build a list of all available agents with their metadata."""
        return [
            {
                "name": a.name,
                "display_name": a.display_name,
                "emoji": a.emoji,
                "specialty": a.specialty,
            }
            for a in self._agent_map.values()
        ]

    def _switch_agent(self, name: str) -> bool:
        """Switch the active agent. Returns True on success."""
        if name not in self._agent_map:
            return False
        self._current_agent_name = name
        return True

    async def _handle_index(self, request: web.Request) -> web.Response:
        """Serve the single-page voice client (cached in memory, ETag support)."""
        if request.headers.get("If-None-Match") == self._html_etag:
            return web.Response(status=304)
        return web.Response(
            text=self._html_content,
            content_type="text/html",
            charset="utf-8",
            headers={"ETag": self._html_etag},
        )

    async def _handle_health(self, _request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "agent": self.agent.name})

    async def _handle_qr(self, request: web.Request) -> web.Response:
        """Render a QR PNG for ?data=<text> (used by the desktop 'scan me'
        banner). Returns 404 if the optional qrcode lib isn't installed, so the
        page falls back to showing the URL as text."""
        data = request.query.get("data", "")
        if not data:
            return web.Response(status=400, text="missing data")
        try:
            import io

            import qrcode  # optional dep
            import qrcode.image.svg

            # SVG factory is pure-Python (no Pillow dependency) and scales
            # crisply in the <img> banner.
            img = qrcode.make(data, image_factory=qrcode.image.svg.SvgPathImage, border=2)
            buf = io.BytesIO()
            img.save(buf)
            return web.Response(body=buf.getvalue(), content_type="image/svg+xml")
        except Exception:
            return web.Response(status=404, text="qr unavailable")

    async def _handle_manifest(self, _request: web.Request) -> web.Response:
        """PWA manifest — makes the voice client installable to a phone's home
        screen as a full-screen app. start_url is relative so it works on the
        ephemeral Cloudflare hostname the phone was opened from."""
        manifest = {
            "name": "Weather Agents",
            "short_name": self.agent.display_name or "Agents",
            "description": "Voice companion — talk to your weather agents.",
            "start_url": ".",
            "scope": ".",
            "display": "standalone",
            "orientation": "portrait",
            "background_color": "#f3ead7",
            "theme_color": "#2a1f17",
            "icons": [
                {"src": "icon.svg", "sizes": "any", "type": "image/svg+xml", "purpose": "any"},
                {
                    "src": "icon.svg",
                    "sizes": "any",
                    "type": "image/svg+xml",
                    "purpose": "maskable",
                },
            ],
        }
        return web.json_response(manifest, content_type="application/manifest+json")

    async def _handle_sw(self, _request: web.Request) -> web.Response:
        """Minimal service worker — required for the install prompt. Network
        passthrough only (no caching), so it never serves a stale shell or
        interferes with the WebSocket."""
        js = (
            "self.addEventListener('install', e => self.skipWaiting());\n"
            "self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));\n"
            "self.addEventListener('fetch', e => { return; });\n"
        )
        return web.Response(
            text=js,
            content_type="application/javascript",
            headers={"Service-Worker-Allowed": "/", "Cache-Control": "no-cache"},
        )

    async def _handle_icon(self, _request: web.Request) -> web.Response:
        """App icon for the manifest / home screen — the central sun mark on a
        dark rounded tile (matches the favicon family)."""
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
            '<rect width="64" height="64" rx="14" fill="#2a1f17"/>'
            '<circle cx="32" cy="32" r="12" fill="#d4a056"/>'
            '<g stroke="#d4a056" stroke-width="2.5" stroke-linecap="round">'
            '<line x1="32" y1="8" x2="32" y2="16"/><line x1="32" y1="48" x2="32" y2="56"/>'
            '<line x1="8" y1="32" x2="16" y2="32"/><line x1="48" y1="32" x2="56" y2="32"/>'
            '<line x1="15" y1="15" x2="21" y2="21"/><line x1="43" y1="43" x2="49" y2="49"/>'
            '<line x1="49" y1="15" x2="43" y2="21"/><line x1="21" y1="43" x2="15" y2="49"/>'
            "</g></svg>"
        )
        return web.Response(body=svg.encode("utf-8"), content_type="image/svg+xml")

    async def _activate_session(self, agent_name: str, session_id: str) -> None:
        """Ensure ``agent_name``'s memory is positioned on ``session_id``.

        Multiple WebSocket connections can coexist now that the lock is
        per-message rather than per-connection. Each WS owns a session_id;
        before any agent operation we re-point the agent's memory at that
        session so two clients don't see each other's short_term history.
        ``load_session`` does the reload from SQLite; if the row was
        deleted out-of-band (e.g. user manually wiped state) we silently
        recreate a fresh session.
        """
        agent = self._agent_map.get(agent_name)
        if not agent:
            return
        mem = agent.memory
        if mem.get_active_session() == session_id:
            return
        ok = await mem.load_session(session_id)
        if not ok:
            await mem.create_session()

    # Cap replayed turns so reopening a long conversation can't blow the
    # context window (and to bound the work). Most recent turns are kept.
    _MAX_CONTEXT_TURNS: int = 40

    def _load_context(self, agent_name: str, messages: list[dict[str, Any]]) -> int:
        """Seed ``agent_name``'s short-term memory from a saved conversation.

        The browser stores conversations client-side; the server session is
        separate. When the user reopens one, we replay its user/agent turns so
        the agent can continue with context. We clear short-term first (keeping
        only the freshly-rebuilt system prompt) to avoid duplicating turns on a
        re-load, then append the turns as plain user/assistant messages — no
        LLM call, no tool replay.
        """
        agent = self._agent_map.get(agent_name)
        if not agent:
            return 0
        mem = agent.memory
        # Drop everything but the system prompt, then re-stamp it fresh.
        mem.short_term = [m for m in mem.short_term if m.role == "system"]
        loaded = 0
        for m in messages[-self._MAX_CONTEXT_TURNS :]:
            role = "assistant" if m.get("role") == "agent" else str(m.get("role") or "")
            text = (m.get("text") or "").strip()
            if role in ("user", "assistant") and text:
                mem.add_message(role, text)
                loaded += 1
        return loaded

    async def _handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        # Create this WS's session under the lock so the agent's
        # memory._active_session isn't being concurrently mutated by another
        # connection. The lock is released as soon as creation finishes —
        # subsequent operations grab it again, briefly, per message.
        async with self._ws_lock:
            await self.agent.init()
            await self.agent.memory.create_session()
            session_id = self.agent.memory.get_active_session()
        my_agent_name = self._current_agent_name
        # Track all sessions created during this WS lifecycle so they
        # are reliably cleaned up on every exit path.
        _open_sessions: list[tuple[str, str]] = [(my_agent_name, session_id)] if session_id else []
        # The in-flight turn (so a `stop` message can cancel it). Runs as a
        # background task rather than inline, keeping the receive loop free.
        current_task: asyncio.Task | None = None

        _log.info("voice_ws_open session=%s", session_id)

        try:
            while True:
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=self._IDLE_TIMEOUT)
                except TimeoutError:
                    _log.info("voice_ws_idle_timeout session=%s", session_id)
                    with contextlib.suppress(Exception):
                        await ws.send_json({"type": "error", "text": "idle timeout"})
                    break
                except (RuntimeError, ConnectionResetError, ConnectionError):
                    # aiohttp raises RuntimeError("WebSocket connection is closed.")
                    # when the peer dropped abruptly and the receive coroutine
                    # is still pending. Treat as a normal close — no point
                    # logging at warning level.
                    _log.info("voice_ws_peer_dropped session=%s", session_id)
                    break

                if msg.type == web.WSMsgType.TEXT:
                    try:
                        data: dict[str, Any] = json.loads(msg.data)
                    except json.JSONDecodeError:
                        with contextlib.suppress(Exception):
                            await ws.send_json({"type": "error", "text": "invalid json"})
                        continue

                    msg_type = data.get("type", "")
                    if msg_type == "speech":
                        text = (data.get("text") or "").strip()
                        if text and session_id:
                            # Run as a cancelable background task so the receive
                            # loop stays free to handle `stop`. A new turn
                            # supersedes any still-running one.
                            if current_task and not current_task.done():
                                current_task.cancel()
                            current_task = asyncio.create_task(
                                self._run_turn(ws, my_agent_name, session_id, text)
                            )
                    elif msg_type == "stop":
                        # User hit pause — cancel the in-flight turn and tell the
                        # client to settle (partial output already streamed stays).
                        if current_task and not current_task.done():
                            current_task.cancel()
                            with contextlib.suppress(Exception):
                                await ws.send_json(
                                    {"type": "done", "full_text": "", "interrupted": True}
                                )
                    elif msg_type == "ping":
                        with contextlib.suppress(Exception):
                            await ws.send_json({"type": "pong"})
                    elif msg_type == "load_context":
                        # Browser reopened a saved conversation: replay its turns
                        # into the agent's short-term memory so the next message
                        # continues with full context (the client store and the
                        # server session are otherwise independent).
                        if session_id:
                            async with self._ws_lock:
                                await self._activate_session(my_agent_name, session_id)
                                count = self._load_context(
                                    my_agent_name, data.get("messages") or []
                                )
                            with contextlib.suppress(Exception):
                                await ws.send_json({"type": "context_loaded", "count": count})
                    elif msg_type == "list_agents":
                        with contextlib.suppress(Exception):
                            await ws.send_json(
                                {
                                    "type": "agent_list",
                                    "agents": self._build_agent_list(),
                                    "current": my_agent_name,
                                }
                            )
                    elif msg_type == "switch_agent":
                        name = data.get("agent", "")
                        async with self._ws_lock:
                            if name not in self._agent_map:
                                with contextlib.suppress(Exception):
                                    await ws.send_json(
                                        {"type": "error", "text": f"unknown agent: {name}"}
                                    )
                                continue
                            if name == my_agent_name:
                                continue
                            # Init the new agent FIRST — if it fails we keep
                            # the current agent and session intact.
                            try:
                                await self._agent_map[name].init()
                                await self._agent_map[name].memory.create_session()
                            except Exception as exc:
                                _log.warning(
                                    "voice_ws_switch_agent_failed agent=%s err=%s", name, exc
                                )
                                with contextlib.suppress(Exception):
                                    await ws.send_json(
                                        {
                                            "type": "error",
                                            "text": f"failed to switch to {name}",
                                        }
                                    )
                                continue
                            # New agent ready — teardown old session and switch.
                            old_agent = self._agent_map.get(my_agent_name)
                            if old_agent and session_id:
                                with contextlib.suppress(Exception):
                                    await old_agent.memory.delete_session(session_id)
                                _open_sessions = [
                                    (a, s) for a, s in _open_sessions if s != session_id
                                ]
                            self._current_agent_name = name
                            my_agent_name = name
                            new_sid = self._agent_map[name].memory.get_active_session()
                            if new_sid:
                                session_id = new_sid
                                _open_sessions.append((my_agent_name, new_sid))
                            with contextlib.suppress(Exception):
                                await ws.send_json(
                                    {
                                        "type": "agent_switched",
                                        "agent": name,
                                        "display_name": self._agent_map[name].display_name,
                                        "emoji": self._agent_map[name].emoji,
                                        "specialty": self._agent_map[name].specialty,
                                        "session_id": new_sid,
                                    }
                                )
                elif msg.type == web.WSMsgType.ERROR:
                    _log.warning("voice_ws_error session=%s err=%s", session_id, ws.exception())
                    break
                elif msg.type in (
                    web.WSMsgType.CLOSE,
                    web.WSMsgType.CLOSING,
                    web.WSMsgType.CLOSED,
                ):
                    break
        except asyncio.CancelledError:
            pass
        finally:
            # Cancel any in-flight turn so it doesn't outlive the connection.
            if current_task and not current_task.done():
                current_task.cancel()
            _log.info("voice_ws_close sessions=%s", [s for _, s in _open_sessions])
            for agent_name, sid in _open_sessions:
                agent = self._agent_map.get(agent_name)
                if agent:
                    async with self._ws_lock:
                        with contextlib.suppress(Exception):
                            await agent.memory.delete_session(sid)

        return ws

    async def _run_turn(
        self, ws: web.WebSocketResponse, agent_name: str, session_id: str, text: str
    ) -> None:
        """Run one speech turn under the WS lock, as a cancelable task.

        Cancellation (a `stop` message) propagates as CancelledError, which
        unwinds the ``async with`` and releases the lock; the receive loop has
        already told the client to settle, so we just let it end.
        """
        try:
            async with self._ws_lock:
                await self._activate_session(agent_name, session_id)
                await self._handle_speech(ws, text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — a turn error shouldn't kill the socket
            _log.warning("voice_turn_error %s", exc)
            with contextlib.suppress(Exception):
                await self._safe_send(ws, {"type": "error", "text": f"error: {exc}"})

    async def _safe_send(self, ws: web.WebSocketResponse, data: dict[str, Any]) -> bool:
        try:
            await ws.send_json(data)
            return True
        except (ConnectionResetError, ConnectionError, OSError):
            return False

    async def _handle_speech(self, ws: web.WebSocketResponse, text: str) -> None:
        """Stream user speech through the agent and pipe events back via WS."""
        await self._safe_send(ws, {"type": "start"})

        full_text = ""
        try:
            async for event in self.agent.chat_stream(text):
                ev_type = event.get("type", "")

                if ev_type == "content":
                    chunk = event.get("text", "")
                    full_text += chunk
                    if not await self._safe_send(ws, {"type": "content", "text": chunk}):
                        return
                elif ev_type == "reasoning":
                    pass
                elif ev_type == "tool_status":
                    if not await self._safe_send(
                        ws, {"type": "status", "label": event.get("label", "")}
                    ):
                        return
                elif ev_type == "done":
                    break
        except Exception as exc:
            _log.warning("voice_speech_error %s", exc)
            await self._safe_send(ws, {"type": "error", "text": f"error: {exc}"})
            return

        clean = _strip_markdown(full_text)
        done_msg: dict[str, Any] = {"type": "done", "full_text": clean, "raw_text": full_text}
        if self.tts_engine and clean:
            done_msg["tts"] = "doubao"
        if not await self._safe_send(ws, done_msg):
            return

        if self.tts_engine and clean:
            await self._synthesize_audio(ws, clean)

    async def _synthesize_audio(self, ws: web.WebSocketResponse, text: str) -> None:
        """Synthesize text to audio and forward chunks via WebSocket as they arrive.

        Uses the streaming API to forward base64 audio chunks directly
        from the TTS response to the browser without buffering or
        decode/re-encode overhead.
        """
        assert self.tts_engine is not None
        try:
            if not await self._safe_send(
                ws, {"type": "audio_start", "format": self.tts_engine.encoding}
            ):
                return
            sent = 0
            async for chunk_b64 in self.tts_engine.synthesize_stream(text):
                sent += 1
                if not await self._safe_send(ws, {"type": "audio_chunk", "data": chunk_b64}):
                    return
            if sent:
                await self._safe_send(ws, {"type": "audio_end"})
            else:
                _log.warning("tts_empty_audio")
                await self._safe_send(ws, {"type": "audio_end", "error": "empty"})
        except Exception as exc:
            _log.warning("tts_synthesis_error %s", exc)
            await self._safe_send(ws, {"type": "audio_end", "error": str(exc)})

    async def run(self, on_started: Any = None) -> None:
        """Start the aiohttp server and run until cancelled.

        ``on_started`` (optional async callable) is awaited once the socket is
        accepting connections — the desktop app uses it to open the Cloudflare
        tunnel only after the local server is actually up.
        """
        runner = web.AppRunner(self._app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port, ssl_context=self.ssl_context)
        await site.start()

        _log.info("voice_server_started host=%s port=%s", self.host, self.port)

        if on_started is not None:
            with contextlib.suppress(Exception):
                await on_started()

        try:
            await asyncio.Event().wait()  # run forever
        finally:
            _log.info("voice_server_shutdown")
            await runner.cleanup()


async def run_voice_server(
    *,
    host: str = "0.0.0.0",
    port: int = 8765,
    agent_name: str = "fair",
    ssl_context: ssl.SSLContext | None = None,
    on_started: Any = None,
) -> None:
    """Create system context, init the target agent, and start the voice server."""
    from weather_agents.core.config import load_config
    from weather_agents.core.factory import create_system_context

    cfg = load_config()
    ctx = create_system_context()

    if agent_name not in ctx.agent_map:
        msg = f"unknown agent: {agent_name}"
        raise ValueError(msg)

    # Init the default agent so system prompt / memory / session are ready.
    await ctx.agent_map[agent_name].init()

    # Create TTS engine if configured
    tts_engine = None
    if cfg.tts.enabled:
        tts_engine = DoubaoTTS(
            access_token=cfg.tts.access_token or None,
            api_key=cfg.tts.api_key or None,
            app_id=cfg.tts.app_id or None,
            resource_id=cfg.tts.resource_id,
            voice_type=cfg.tts.voice_type,
            encoding=cfg.tts.encoding,
            speed_ratio=cfg.tts.speed_ratio,
            volume_ratio=cfg.tts.volume_ratio,
            pitch_ratio=cfg.tts.pitch_ratio,
            emotion=cfg.tts.emotion,
        )

    server = VoiceServer(
        ctx.agent_map,
        ctx,
        agent_name=agent_name,
        host=host,
        port=port,
        tts_engine=tts_engine,
        ssl_context=ssl_context,
    )
    try:
        await server.run(on_started=on_started)
    finally:
        if tts_engine:
            await tts_engine.close()
        await ctx.close_all()


def _strip_markdown(text: str) -> str:
    """Remove common markdown syntax for clean TTS output."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^>\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

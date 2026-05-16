"""WebSocket voice server for remote voice conversation with agents."""

from __future__ import annotations

import asyncio
import base64
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
    """

    def __init__(
        self,
        agent: BaseAgent,
        system_ctx: SystemContext,
        *,
        host: str = "0.0.0.0",
        port: int = 8765,
        tts_engine: DoubaoTTS | None = None,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        self.agent = agent
        self.ctx = system_ctx
        self.host = host
        self.port = port
        self.tts_engine = tts_engine
        self.ssl_context = ssl_context
        self._app = web.Application()

        self._app.router.add_get("/", self._handle_index)
        self._app.router.add_get("/health", self._handle_health)
        self._app.router.add_get("/ws", self._handle_ws)

    async def _handle_index(self, _request: web.Request) -> web.Response:
        """Serve the single-page voice client."""
        if _HTML_PATH.is_file():
            html = _HTML_PATH.read_text(encoding="utf-8")
        else:
            html = "<h1>Voice client not found</h1>"
        return web.Response(text=html, content_type="text/html", charset="utf-8")

    async def _handle_health(self, _request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "agent": self.agent.name})

    async def _handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        # Each voice connection gets its own session for isolation.
        await self.agent.memory.create_session()
        session_id = self.agent.memory.get_active_session()

        _log.info("voice_ws_open session=%s", session_id)

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        data: dict[str, Any] = json.loads(msg.data)
                    except json.JSONDecodeError:
                        await ws.send_json({"type": "error", "text": "invalid json"})
                        continue

                    msg_type = data.get("type", "")
                    if msg_type == "speech":
                        text = (data.get("text") or "").strip()
                        if text:
                            await self._handle_speech(ws, text)
                    elif msg_type == "ping":
                        await ws.send_json({"type": "pong"})
                elif msg.type == web.WSMsgType.ERROR:
                    _log.warning("voice_ws_error session=%s err=%s", session_id, ws.exception())
        except asyncio.CancelledError:
            pass
        finally:
            _log.info("voice_ws_close session=%s", session_id)

        return ws

    async def _handle_speech(self, ws: web.WebSocketResponse, text: str) -> None:
        """Stream user speech through the agent and pipe events back via WS."""
        await ws.send_json({"type": "start"})

        full_text = ""
        try:
            async for event in self.agent.chat_stream(text):
                ev_type = event.get("type", "")

                if ev_type == "content":
                    chunk = event.get("text", "")
                    full_text += chunk
                    await ws.send_json({"type": "content", "text": chunk})
                elif ev_type == "reasoning":
                    pass  # skip reasoning in voice mode
                elif ev_type == "tool_status":
                    await ws.send_json({"type": "status", "label": event.get("label", "")})
                elif ev_type == "done":
                    break
        except Exception as exc:
            _log.warning("voice_speech_error %s", exc)
            await ws.send_json({"type": "error", "text": f"error: {exc}"})
            return

        # Strip markdown for client-side TTS consumption
        clean = _strip_markdown(full_text)
        done_msg: dict[str, Any] = {"type": "done", "full_text": clean, "raw_text": full_text}
        if self.tts_engine and clean:
            done_msg["tts"] = "doubao"
        await ws.send_json(done_msg)

        # Synthesize audio via Doubao TTS if engine is available
        if self.tts_engine and clean:
            await self._synthesize_audio(ws, clean)

    async def _synthesize_audio(self, ws: web.WebSocketResponse, text: str) -> None:
        """Synthesize text to audio and send chunks via WebSocket."""
        assert self.tts_engine is not None  # guarded at call site
        try:
            audio_data = await self.tts_engine.synthesize(text)
            if not audio_data:
                _log.warning("tts_empty_audio")
                await ws.send_json({"type": "audio_end", "error": "empty"})
                return
            audio_b64 = base64.b64encode(audio_data).decode("ascii")
            await ws.send_json({"type": "audio_start", "format": self.tts_engine.encoding})
            # Send large audio in ~32KB base64 chunks
            chunk_size = 48000  # 36 KB base64 ≈ 27 KB raw per chunk
            for i in range(0, len(audio_b64), chunk_size):
                await ws.send_json({"type": "audio_chunk", "data": audio_b64[i : i + chunk_size]})
            await ws.send_json({"type": "audio_end"})
        except Exception as exc:
            _log.warning("tts_synthesis_error %s", exc)
            await ws.send_json({"type": "audio_end", "error": str(exc)})

    async def run(self) -> None:
        """Start the aiohttp server and run until cancelled."""
        runner = web.AppRunner(self._app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port, ssl_context=self.ssl_context)
        await site.start()

        _log.info("voice_server_started host=%s port=%s", self.host, self.port)

        try:
            await asyncio.Event().wait()  # run forever
        finally:
            _log.info("voice_server_shutdown")
            await runner.cleanup()


async def run_voice_server(
    *,
    host: str = "0.0.0.0",
    port: int = 8765,
    agent_name: str = "sunshine",
    ssl_context: ssl.SSLContext | None = None,
) -> None:
    """Create system context, init the target agent, and start the voice server."""
    from weather_agents.core.config import load_config
    from weather_agents.core.factory import create_system_context

    cfg = load_config()
    ctx = create_system_context()

    agent = ctx.agent_map.get(agent_name)
    if agent is None:
        msg = f"unknown agent: {agent_name}"
        raise ValueError(msg)

    # Init the agent so system prompt / memory / session are ready.
    await agent.init()

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
        agent, ctx, host=host, port=port, tts_engine=tts_engine, ssl_context=ssl_context
    )
    try:
        await server.run()
    finally:
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

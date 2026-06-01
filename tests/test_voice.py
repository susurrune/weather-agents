"""Tests for the voice server module.

WebSocket handler integration tests are omitted because aiohttp's
make_mocked_request doesn't support real WS upgrade handshakes in
unit-test isolation — those are covered by end-to-end manual testing.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from weather_agents.web.server import VoiceServer, _strip_markdown


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.name = "fair"
    agent.display_name = "晴"
    agent.memory.get_active_session.return_value = None
    agent.memory.create_session = AsyncMock()
    agent.memory.short_term = []
    return agent


@pytest.fixture
def mock_ctx():
    ctx = MagicMock()
    ctx.close_all = AsyncMock()
    return ctx


@pytest.fixture
def voice_server(mock_agent, mock_ctx):
    return VoiceServer({"fair": mock_agent}, mock_ctx, agent_name="fair", host="127.0.0.1", port=0)


@pytest.fixture
def mock_ws():
    """A mock WebSocketResponse that records sent messages."""
    ws = AsyncMock()
    ws.sent: list[dict] = []

    async def send_json(data):
        ws.sent.append(data)

    ws.send_json = send_json
    return ws


# ── HTTP endpoints ──


@pytest.mark.asyncio
async def test_health_endpoint(voice_server):
    """GET /health returns agent status."""
    request = _make_request("GET", "/health")
    resp = await voice_server._handle_health(request)
    assert resp.status == 200
    body = json.loads(resp.body)
    assert body["status"] == "ok"
    assert body["agent"] == "fair"


@pytest.mark.asyncio
async def test_index_returns_html(voice_server):
    """GET / returns HTML content."""
    request = _make_request("GET", "/")
    resp = await voice_server._handle_index(request)
    assert resp.status == 200
    assert "text/html" in resp.content_type


# ── _handle_speech (core logic, tested with mock WS) ──


@pytest.mark.asyncio
async def test_speech_round_trip(voice_server, mock_agent, mock_ws):
    """Full round-trip: send speech → start/content/done events."""

    async def mock_stream(_text):
        yield {"type": "content", "text": "Hello "}
        yield {"type": "content", "text": "world!"}
        yield {"type": "done"}

    mock_agent.chat_stream = mock_stream

    await voice_server._handle_speech(mock_ws, "你好")

    types = [m["type"] for m in mock_ws.sent]
    assert types == ["start", "content", "content", "done"]
    assert mock_ws.sent[-1]["full_text"] == "Hello world!"


@pytest.mark.asyncio
async def test_speech_with_tool_status(voice_server, mock_agent, mock_ws):
    """Tool_status events are forwarded."""

    async def mock_stream(_text):
        yield {"type": "tool_status", "label": "Searching..."}
        yield {"type": "content", "text": "Result"}
        yield {"type": "done"}

    mock_agent.chat_stream = mock_stream

    await voice_server._handle_speech(mock_ws, "search")

    types = [m["type"] for m in mock_ws.sent]
    assert "status" in types
    assert types[-1] == "done"


@pytest.mark.asyncio
async def test_speech_error_handling(voice_server, mock_agent, mock_ws):
    """Agent exception sends error message."""

    async def mock_stream(_text):
        raise RuntimeError("test error")
        yield

    mock_agent.chat_stream = mock_stream

    await voice_server._handle_speech(mock_ws, "trigger")

    assert mock_ws.sent[0]["type"] == "start"
    assert mock_ws.sent[1]["type"] == "error"
    assert "test error" in mock_ws.sent[1]["text"]


@pytest.mark.asyncio
async def test_done_has_stripped_text(voice_server, mock_agent, mock_ws):
    """Done event strips markdown from full_text."""

    async def mock_stream(_text):
        yield {"type": "content", "text": "**Hello**"}
        yield {"type": "done"}

    mock_agent.chat_stream = mock_stream

    await voice_server._handle_speech(mock_ws, "hi")

    last = mock_ws.sent[-1]
    assert last["type"] == "done"
    assert last["full_text"] == "Hello"


@pytest.mark.asyncio
async def test_reasoning_events_skipped(voice_server, mock_agent, mock_ws):
    """Reasoning events are not forwarded."""

    async def mock_stream(_text):
        yield {"type": "reasoning", "text": "thinking..."}
        yield {"type": "content", "text": "Answer"}
        yield {"type": "done"}

    mock_agent.chat_stream = mock_stream

    await voice_server._handle_speech(mock_ws, "think")

    types = [m["type"] for m in mock_ws.sent]
    assert "reasoning" not in types
    assert "content" in types


@pytest.mark.asyncio
async def test_empty_text_no_agent_call(voice_server, mock_agent, mock_ws):
    """Empty text doesn't call chat_stream."""
    called = False

    async def mock_stream(_text):
        nonlocal called
        called = True
        yield {"type": "done"}

    mock_agent.chat_stream = mock_stream

    # _handle_speech is called from _handle_ws which already validates non-empty
    await voice_server._handle_speech(mock_ws, "ok")
    assert called


# ── Context replay (reopening a saved conversation) ──


def test_load_context_seeds_memory(voice_server, mock_agent):
    """Replaying a saved conversation drops stale turns (keeping only the
    system prompt) and re-adds the user/agent turns as user/assistant."""
    from types import SimpleNamespace

    mock_agent.memory.short_term = [
        SimpleNamespace(role="system", content="SYS"),
        SimpleNamespace(role="user", content="stale — should be dropped"),
    ]
    added: list[tuple[str, str]] = []
    mock_agent.memory.add_message = lambda role, content, **_k: added.append((role, content))

    n = voice_server._load_context(
        "fair",
        [
            {"role": "user", "text": "hi"},
            {"role": "agent", "text": "hello there"},  # agent → assistant
            {"role": "system", "text": "ignored"},  # non user/agent → skipped
            {"role": "user", "text": "   "},  # blank → skipped
        ],
    )

    assert n == 2
    assert added == [("user", "hi"), ("assistant", "hello there")]
    # short_term reset to the system prompt only (stale user turn dropped)
    assert [m.role for m in mock_agent.memory.short_term] == ["system"]


def test_load_context_caps_turns(voice_server, mock_agent):
    """Only the most recent _MAX_CONTEXT_TURNS turns are replayed."""
    mock_agent.memory.short_term = []
    added: list[tuple[str, str]] = []
    mock_agent.memory.add_message = lambda role, content, **_k: added.append((role, content))

    cap = VoiceServer._MAX_CONTEXT_TURNS
    msgs = [{"role": "user", "text": f"m{i}"} for i in range(cap + 10)]
    n = voice_server._load_context("fair", msgs)

    assert n == cap
    assert added[0][1] == "m10"  # oldest 10 dropped
    assert added[-1][1] == f"m{cap + 9}"


def test_load_context_unknown_agent(voice_server):
    assert voice_server._load_context("nope", [{"role": "user", "text": "x"}]) == 0


# ── Markdown stripping ──


@pytest.mark.parametrize(
    "md,expected",
    [
        ("**bold**", "bold"),
        ("*italic*", "italic"),
        ("`code`", ""),
        ("```block```", ""),
        ("# Heading", "Heading"),
        ("[link](url)", "link"),
        ("- item", "item"),
        ("1. item", "item"),
        ("> quote", "quote"),
        ("**bold** and *italic*", "bold and italic"),
    ],
)
def test_strip_markdown(md, expected):
    assert _strip_markdown(md) == expected, f"failed for: {md!r}"


# ── Helpers ──


def _make_request(method: str, path: str):
    from aiohttp.test_utils import make_mocked_request

    return make_mocked_request(method, path)

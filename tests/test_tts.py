"""Tests for Doubao TTS V3 HTTP Unidirectional API."""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from weather_agents.web.tts import DoubaoTTS

# ── Auth / init ──


def test_auth_with_api_key():
    """API key auth uses V3 new console headers."""
    tts = DoubaoTTS(api_key="key-123")
    headers = tts._get_http_headers()
    assert headers["X-Api-Key"] == "key-123"
    assert headers["X-Api-Resource-Id"] == "seed-tts-2.0"


def test_auth_with_legacy():
    """Legacy console uses app_id + access_token."""
    tts = DoubaoTTS(app_id="app-456", access_token="tok-789")
    headers = tts._get_http_headers()
    assert headers["X-Api-App-Id"] == "app-456"
    assert headers["X-Api-Access-Key"] == "tok-789"


def test_auth_raises_without_credentials():
    """Missing credentials raises ValueError."""
    tts = DoubaoTTS()
    with pytest.raises(ValueError, match="requires api_key"):
        tts._get_http_headers()


def test_is_available_with_api_key():
    assert DoubaoTTS(api_key="k").is_available


def test_is_available_with_legacy():
    assert DoubaoTTS(app_id="a", access_token="t").is_available


def test_not_available_without_auth():
    assert not DoubaoTTS().is_available


def test_default_params():
    """Default params match V3 API spec."""
    tts = DoubaoTTS(api_key="k")
    assert tts.resource_id == "seed-tts-2.0"
    assert tts.voice_type == "zh_female_sajiaoxuemei_uranus_bigtts"
    assert tts.encoding == "mp3"


# ── Rate mapping ──


@pytest.mark.parametrize(
    "value,expected",
    [
        (1.0, 0),
        (2.0, 100),
        (0.5, -50),
        (1.5, 50),
        (0.75, -25),
        (0.0, -50),
    ],
)
def test_map_speech_rate(value, expected):
    assert DoubaoTTS._map_rate(value, 0.5, 2.0, -50, 100) == expected


# ── Request body ──


def test_request_body_defaults():
    """Request body has correct structure with V3 params."""
    tts = DoubaoTTS(api_key="k", voice_type="zh_female_xiaoyun_bigtts")
    body = tts._make_request_body("test text")

    assert body["user"]["uid"] == "wa_voice"
    assert body["req_params"]["text"] == "test text"
    assert body["req_params"]["speaker"] == "zh_female_xiaoyun_bigtts"
    assert body["req_params"]["audio_params"]["format"] == "mp3"
    assert body["req_params"]["audio_params"]["sample_rate"] == 24000


def test_request_body_custom_speed():
    """Non-default speed ratio is mapped to speech_rate."""
    tts = DoubaoTTS(api_key="k", speed_ratio=1.5)
    body = tts._make_request_body("hi")
    assert body["req_params"]["audio_params"]["speech_rate"] == 50


def test_request_body_default_speed_omitted():
    """Default speed (1.0) omits speech_rate from payload."""
    tts = DoubaoTTS(api_key="k", speed_ratio=1.0)
    body = tts._make_request_body("hi")
    assert "speech_rate" not in body["req_params"]["audio_params"]


# ── Stream helpers ──


@pytest.fixture
def tts() -> DoubaoTTS:
    return DoubaoTTS(api_key="k")


def _stream_ctx(resp_status: int, resp_text: str) -> MagicMock:
    """Build a mock for ``client.stream()`` return value.

    Returns a ``MagicMock`` that acts as an async context manager.
    ``__aenter__`` yields a response mock with ``status_code`` and
    ``aiter_text()`` returning *resp_text* as a single chunk.
    """

    async def _single_chunk() -> AsyncMock:
        yield resp_text

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = resp_status
    mock_resp.aiter_text = _single_chunk
    if resp_status != 200:
        mock_resp.aread = AsyncMock(return_value=resp_text.encode())

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_resp)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


def _json_response(chunks: list[bytes]) -> str:
    """Build multi-line JSON response text from audio chunks."""
    lines = []
    for c in chunks:
        b64 = base64.b64encode(c).decode()
        lines.append(f'{{"code":0,"message":"","data":"{b64}"}}')
    lines.append('{"code":20000000,"message":"OK"}')
    return "\n".join(lines)


@pytest.fixture
def mock_stream():
    with patch.object(httpx.AsyncClient, "stream") as m:
        yield m


# ── Full synthesize flow ──


@pytest.mark.asyncio
async def test_synthesize_empty_text():
    """Empty text returns empty bytes without connecting."""
    tts = DoubaoTTS(api_key="k")
    assert await tts.synthesize("") == b""
    assert await tts.synthesize("  ") == b""


@pytest.mark.asyncio
async def test_synthesize_success():
    """Successful synthesize decodes base64 audio from multi-chunk JSON."""
    tts = DoubaoTTS(api_key="test-key")
    chunk1, chunk2 = b"fake_mp3_data_chunk1", b"fake_mp3_data_chunk2"
    fake_audio = chunk1 + chunk2
    json_text = _json_response([chunk1, chunk2])
    ctx = _stream_ctx(200, json_text)

    with patch.object(httpx.AsyncClient, "stream", return_value=ctx) as mock_stream:
        result = await tts.synthesize("hello world")

    assert result == fake_audio
    mock_stream.assert_called_once()
    args, kwargs = mock_stream.call_args
    assert args[0] == "POST"
    assert args[1] == "https://openspeech.bytedance.com/api/v3/tts/unidirectional"
    assert kwargs["headers"]["X-Api-Key"] == "test-key"
    assert kwargs["headers"]["X-Api-Resource-Id"] == "seed-tts-2.0"


@pytest.mark.asyncio
async def test_synthesize_single_chunk():
    """Single chunk + end-of-stream still works."""
    tts = DoubaoTTS(api_key="k")
    ctx = _stream_ctx(200, _json_response([b"audio_data"]))
    with patch.object(httpx.AsyncClient, "stream", return_value=ctx):
        result = await tts.synthesize("hi")
    assert result == b"audio_data"


@pytest.mark.asyncio
async def test_synthesize_http_error():
    """Non-200 status code raises RuntimeError."""
    tts = DoubaoTTS(api_key="k")
    ctx = _stream_ctx(401, "unauthorized")
    with (
        patch.object(httpx.AsyncClient, "stream", return_value=ctx),
        pytest.raises(RuntimeError, match="401"),
    ):
        await tts.synthesize("test")


@pytest.mark.asyncio
async def test_synthesize_api_error():
    """API returns JSON error (code != 0) with 200 status."""
    tts = DoubaoTTS(api_key="k")
    err_json = '{"reqid":"","code":55000000,"message":"resource ID is mismatched"}'
    ctx = _stream_ctx(200, err_json)
    with (
        patch.object(httpx.AsyncClient, "stream", return_value=ctx),
        pytest.raises(RuntimeError, match="55000000"),
    ):
        await tts.synthesize("test")


@pytest.mark.asyncio
async def test_synthesize_empty_response():
    """Empty response body raises RuntimeError."""
    tts = DoubaoTTS(api_key="k")
    ctx = _stream_ctx(200, "")
    with (
        patch.object(httpx.AsyncClient, "stream", return_value=ctx),
        pytest.raises(RuntimeError, match="missing data"),
    ):
        await tts.synthesize("test")


# ── synthesize_stream ──


@pytest.mark.asyncio
async def test_synthesize_stream_empty_text():
    """Empty text yields nothing without connecting."""
    tts = DoubaoTTS(api_key="k")
    got = [c async for c in tts.synthesize_stream("")]
    assert got == []
    got = [c async for c in tts.synthesize_stream("  ")]
    assert got == []


@pytest.mark.asyncio
async def test_synthesize_stream_yields_b64():
    """synthesize_stream yields raw base64 strings from the API."""
    tts = DoubaoTTS(api_key="k")
    chunk1, chunk2 = b"\x00\x01\x02", b"\x03\x04\x05"
    b64_1 = base64.b64encode(chunk1).decode()
    b64_2 = base64.b64encode(chunk2).decode()
    json_text = _json_response([chunk1, chunk2])
    ctx = _stream_ctx(200, json_text)

    with patch.object(httpx.AsyncClient, "stream", return_value=ctx):
        got = [c async for c in tts.synthesize_stream("hello")]

    assert got == [b64_1, b64_2]


@pytest.mark.asyncio
async def test_synthesize_stream_http_error():
    """Non-200 status propagates from stream."""
    tts = DoubaoTTS(api_key="k")
    ctx = _stream_ctx(403, "forbidden")
    with (
        patch.object(httpx.AsyncClient, "stream", return_value=ctx),
        pytest.raises(RuntimeError, match="403"),
    ):
        async for _ in tts.synthesize_stream("hi"):
            pass

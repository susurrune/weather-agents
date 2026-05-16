"""Tests for Doubao TTS V3 HTTP Unidirectional API."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from unittest.mock import MagicMock

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
    assert tts.voice_type == "zh_female_vv_uranus_bigtts"
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


# ── Full synthesize flow ──


@pytest.mark.asyncio
async def test_synthesize_empty_text():
    """Empty text returns empty bytes without connecting."""
    tts = DoubaoTTS(api_key="k")
    assert await tts.synthesize("") == b""
    assert await tts.synthesize("  ") == b""


@pytest.mark.asyncio
async def test_synthesize_success():
    """Successful synthesize returns audio bytes."""
    tts = DoubaoTTS(api_key="test-key")
    fake_audio = b"fake_mp3_data"

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.content = fake_audio
    mock_response.headers = {"content-type": "audio/mpeg"}
    mock_response.json = MagicMock(side_effect=ValueError("not json"))

    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response)) as mock_post:
        result = await tts.synthesize("hello world")

    assert result == fake_audio
    mock_post.assert_awaited_once()

    # Verify the endpoint URL and auth headers
    call_args = mock_post.call_args
    assert call_args[0][0] == "https://openspeech.bytedance.com/api/v3/tts/unidirectional"
    assert call_args[1]["headers"]["X-Api-Key"] == "test-key"
    assert call_args[1]["headers"]["X-Api-Resource-Id"] == "seed-tts-2.0"


@pytest.mark.asyncio
async def test_synthesize_http_error():
    """Non-200 status code raises RuntimeError."""
    tts = DoubaoTTS(api_key="k")

    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.status_code = 401
    mock_response.text = "unauthorized"

    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response)):
        with pytest.raises(RuntimeError, match="401"):
            await tts.synthesize("test")


@pytest.mark.asyncio
async def test_synthesize_api_error():
    """API returns JSON error with 200 status."""
    tts = DoubaoTTS(api_key="k")

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.content = b'{"reqid":"","code":55000000,"message":"resource ID is mismatched"}'
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json = MagicMock(return_value={"reqid": "", "code": 55000000, "message": "resource ID is mismatched"})

    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response)):
        with pytest.raises(RuntimeError, match="55000000"):
            await tts.synthesize("test")


@pytest.mark.asyncio
async def test_synthesize_no_content_type_json_check():
    """When content-type isn't available, small responses may be treated as error."""
    tts = DoubaoTTS(api_key="k")

    err_body = b'{"code":55000000,"message":"error"}'
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.content = err_body
    mock_response.headers = {}
    mock_response.json = MagicMock(return_value={"code": 55000000, "message": "error"})

    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response)):
        with pytest.raises(RuntimeError, match="55000000"):
            await tts.synthesize("test")

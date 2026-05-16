"""Tests for Doubao TTS V3 WebSocket bidirectional protocol."""

from __future__ import annotations

import json
import struct
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weather_agents.web.tts import (
    EVENT_CONNECTION_FAILED,
    EVENT_CONNECTION_STARTED,
    EVENT_SESSION_FAILED,
    EVENT_SESSION_FINISHED,
    EVENT_SESSION_STARTED,
    EVENT_START_CONNECTION,
    EVENT_START_SESSION,
    EVENT_TTS_RESPONSE,
    MSG_AUDIO_ONLY_RESPONSE,
    MSG_ERROR,
    MSG_FULL_CLIENT_REQUEST,
    MSG_FULL_SERVER_RESPONSE,
    DoubaoTTS,
    _build_request_frame,
    _parse_response_frame,
)

# ── Binary protocol helpers ──


def test_build_request_frame_minimal():
    """Frame with just event + payload has correct structure."""
    frame = _build_request_frame(MSG_FULL_CLIENT_REQUEST, EVENT_START_CONNECTION, {})
    assert len(frame) >= 12
    # Byte 0: v1 + 4-byte header
    assert frame[0] == 0x11
    # Byte 1: client-req + has_event
    assert frame[1] == 0x14
    # Bytes 4-7: event code (1 = StartConnection)
    assert struct.unpack(">i", frame[4:8])[0] == EVENT_START_CONNECTION
    # Payload length = 2 (for "{}")
    payload_len = struct.unpack(">I", frame[8:12])[0]
    assert payload_len == 2


def test_build_request_frame_with_session_id():
    """Frame with session_id includes it after the event code."""
    sid = "abc123"
    payload = {"text": "hello"}
    frame = _build_request_frame(
        MSG_FULL_CLIENT_REQUEST, EVENT_START_SESSION, payload, session_id=sid
    )
    # Event
    assert struct.unpack(">i", frame[4:8])[0] == EVENT_START_SESSION
    # Session ID length
    off = 8
    sid_len = struct.unpack(">I", frame[off : off + 4])[0]
    assert sid_len == len(sid)
    off += 4
    assert frame[off : off + sid_len].decode() == sid
    off += sid_len
    # Payload
    plen = struct.unpack(">I", frame[off : off + 4])[0]
    off += 4
    payload_decoded = json.loads(frame[off : off + plen].decode())
    assert payload_decoded["text"] == "hello"


def test_build_request_frame_dict_payload():
    """Dict payload is serialized as JSON."""
    frame = _build_request_frame(1, 1, {"key": "value"})
    plen = struct.unpack(">I", frame[8:12])[0]
    payload = json.loads(frame[12 : 12 + plen].decode())
    assert payload["key"] == "value"


def test_parse_connection_started():
    """Parse a ConnectionStarted response frame."""
    conn_id = b"testcid"
    payload = b"{}"
    data = _make_response_frame(
        MSG_FULL_SERVER_RESPONSE,
        EVENT_CONNECTION_STARTED,
        conn_id=conn_id,
        payload=payload,
    )
    result = _parse_response_frame(data)
    assert result["event"] == EVENT_CONNECTION_STARTED
    assert result["conn_id"] == "testcid"
    assert result["payload"] == {}


def test_parse_connection_failed():
    """Parse a ConnectionFailed response frame."""
    payload = json.dumps({"status_code": 45000001, "message": "unauthorized"}).encode()
    data = _make_response_frame(
        MSG_FULL_SERVER_RESPONSE,
        EVENT_CONNECTION_FAILED,
        conn_id=b"x",
        payload=payload,
    )
    result = _parse_response_frame(data)
    assert result["event"] == EVENT_CONNECTION_FAILED
    assert result["payload"]["status_code"] == 45000001


def test_parse_session_started():
    """Parse a SessionStarted response frame."""
    sid = b"testsession12"
    payload = b"{}"
    data = _make_response_frame(
        MSG_FULL_SERVER_RESPONSE,
        EVENT_SESSION_STARTED,
        session_id=sid,
        payload=payload,
    )
    result = _parse_response_frame(data)
    assert result["event"] == EVENT_SESSION_STARTED
    assert result["session_id"] == "testsession12"


def test_parse_audio_only_response():
    """Parse an Audio-only TTSResponse frame with binary audio data."""
    audio = b"\xff\xf3\x00\x01fake_mp3_data"
    data = _make_response_frame(
        MSG_AUDIO_ONLY_RESPONSE,
        EVENT_TTS_RESPONSE,
        session_id=b"testid1234567",
        payload=audio,
        serialization=0,
    )
    result = _parse_response_frame(data)
    assert result["event"] == EVENT_TTS_RESPONSE
    assert result["msg_type"] == MSG_AUDIO_ONLY_RESPONSE
    assert result["payload_raw"] == audio


def test_parse_short_frame():
    """Frame shorter than header returns safe defaults."""
    result = _parse_response_frame(b"")
    assert result["event"] is None


def test_parse_error_frame():
    """Error frame (msg_type=0xF) has no event but has error code."""
    data = bytes(
        [
            0x11,
            0xF0,
            0x10,
            0x00,
            0x00,
            0x00,
            0x45,
            0x00,  # error at byte 4
            0x00,
            0x00,
            0x10,  # payload_len
            0x62,
            0x61,
            0x64,
            0x5F,
            0x65,
            0x72,
            0x72,
            0x6F,
            0x72,
            0x5F,
            0x6D,
            0x73,
            0x67,
            0x00,
            0x00,
            0x00,
        ]
    )
    # This may or may not have event flag depending on specific_flags
    result = _parse_response_frame(data)
    assert result["msg_type"] == MSG_ERROR


# ── Auth / init ──


def test_auth_with_api_key():
    """API key auth uses V3 new console headers."""
    tts = DoubaoTTS(api_key="key-123")
    headers = tts._get_ws_headers()
    assert headers["X-Api-Key"] == "key-123"
    assert headers["X-Api-Resource-Id"] == "seed-tts-2.0"


def test_auth_with_legacy():
    """Legacy console uses app_id + access_token."""
    tts = DoubaoTTS(app_id="app-456", access_token="tok-789")
    headers = tts._get_ws_headers()
    assert headers["X-Api-App-Id"] == "app-456"
    assert headers["X-Api-Access-Key"] == "tok-789"


def test_auth_raises_without_credentials():
    """Missing credentials raises ValueError."""
    tts = DoubaoTTS()
    with pytest.raises(ValueError, match="requires api_key"):
        tts._get_ws_headers()


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
    assert tts.voice_type == "zh_female_cancan_mars_bigtts"
    assert tts.encoding == "mp3"
    assert tts.WS_ENDPOINT == "wss://openspeech.bytedance.com/api/v3/tts/bidirection"


# ── Rate mapping ──


@pytest.mark.parametrize(
    "value,expected",
    [
        (1.0, 0),  # normal speed → 0
        (2.0, 100),  # max → 100
        (0.5, -50),  # min → -50
        (1.5, 50),  # middle → 50
        (0.75, -25),  # between min and normal
        (0.0, -50),  # below range → clamp to min
    ],
)
def test_map_speech_rate(value, expected):
    assert DoubaoTTS._map_rate(value, 0.5, 2.0, -50, 100) == expected


# ── Session payload ──


def test_session_payload_defaults():
    """StartSession payload has correct structure with V3 params."""
    tts = DoubaoTTS(api_key="k", voice_type="zh_female_xiaoyun_bigtts")
    payload = tts._make_session_payload("test text")

    assert payload["user"]["uid"] == "wa_voice"
    assert payload["event"] == EVENT_START_SESSION
    assert payload["req_params"]["text"] == "test text"
    assert payload["req_params"]["speaker"] == "zh_female_xiaoyun_bigtts"
    assert payload["req_params"]["audio_params"]["format"] == "mp3"
    assert payload["req_params"]["audio_params"]["sample_rate"] == 24000


def test_session_payload_custom_speed():
    """Non-default speed ratio is mapped to speech_rate."""
    tts = DoubaoTTS(api_key="k", speed_ratio=1.5)
    payload = tts._make_session_payload("hi")
    assert payload["req_params"]["audio_params"]["speech_rate"] == 50


def test_session_payload_default_speed_omitted():
    """Default speed (1.0) omits speech_rate from payload."""
    tts = DoubaoTTS(api_key="k", speed_ratio=1.0)
    payload = tts._make_session_payload("hi")
    assert "speech_rate" not in payload["req_params"]["audio_params"]


# ── Full synthesize flow ──


@pytest.mark.asyncio
async def test_synthesize_empty_text():
    """Empty text returns empty bytes without connecting."""
    tts = DoubaoTTS(api_key="k")
    assert await tts.synthesize("") == b""
    assert await tts.synthesize("  ") == b""


@pytest.mark.asyncio
async def test_synthesize_full_flow():
    """Full synthesize flow with mocked WebSocket."""
    tts = DoubaoTTS(api_key="test-key")
    fake_audio = b"fake_mp3_data"

    mock_ws = AsyncMock()
    # Return frames in sequence: ConnectionStarted, SessionStarted,
    # TTSResponse (audio), SessionFinished
    mock_ws.recv = AsyncMock(
        side_effect=[
            _make_response_frame(
                MSG_FULL_SERVER_RESPONSE,
                EVENT_CONNECTION_STARTED,
                conn_id=b"cid12345",
                payload=b"{}",
            ),
            _make_response_frame(
                MSG_FULL_SERVER_RESPONSE,
                EVENT_SESSION_STARTED,
                session_id=b"testid123456",
                payload=b"{}",
            ),
            _make_response_frame(
                MSG_AUDIO_ONLY_RESPONSE,
                EVENT_TTS_RESPONSE,
                session_id=b"testid123456",
                payload=fake_audio,
                serialization=0,
            ),
            _make_response_frame(
                MSG_FULL_SERVER_RESPONSE,
                EVENT_SESSION_FINISHED,
                session_id=b"testid123456",
                payload=json.dumps({"status_code": 20000000, "message": "ok"}).encode(),
            ),
        ]
    )

    with patch("websockets.connect", MagicMock()) as mock_connect:
        mock_connect.return_value.__aenter__.return_value = mock_ws
        result = await tts.synthesize("hello world")

    assert result == fake_audio
    # Verify the correct frames were sent
    sent_frames = mock_ws.send.call_args_list
    assert len(sent_frames) >= 2  # StartConnection + StartSession


@pytest.mark.asyncio
async def test_synthesize_connection_failed():
    """ConnectionFailed raises RuntimeError."""
    tts = DoubaoTTS(api_key="k")

    mock_ws = AsyncMock()
    mock_ws.recv = AsyncMock(
        side_effect=[
            _make_response_frame(
                MSG_FULL_SERVER_RESPONSE,
                EVENT_CONNECTION_FAILED,
                conn_id=b"x",
                payload=json.dumps({"status_code": 45000001, "message": "unauthorized"}).encode(),
            ),
        ]
    )

    with patch("websockets.connect", MagicMock()) as mock_connect:
        mock_connect.return_value.__aenter__.return_value = mock_ws
        with pytest.raises(RuntimeError, match="unauthorized"):
            await tts.synthesize("test")


@pytest.mark.asyncio
async def test_synthesize_session_failed():
    """SessionFailed raises RuntimeError."""
    tts = DoubaoTTS(api_key="k")

    mock_ws = AsyncMock()
    mock_ws.recv = AsyncMock(
        side_effect=[
            _make_response_frame(
                MSG_FULL_SERVER_RESPONSE,
                EVENT_CONNECTION_STARTED,
                conn_id=b"cid12345",
                payload=b"{}",
            ),
            _make_response_frame(
                MSG_FULL_SERVER_RESPONSE,
                EVENT_SESSION_STARTED,
                session_id=b"testid123456",
                payload=b"{}",
            ),
            _make_response_frame(
                MSG_FULL_SERVER_RESPONSE,
                EVENT_SESSION_FAILED,
                session_id=b"testid123456",
                payload=json.dumps({"status_code": 45000001, "message": "bad request"}).encode(),
            ),
        ]
    )

    with patch("websockets.connect", MagicMock()) as mock_connect:
        mock_connect.return_value.__aenter__.return_value = mock_ws
        with pytest.raises(RuntimeError, match="bad request"):
            await tts.synthesize("test")


@pytest.mark.asyncio
async def test_synthesize_multiple_audio_chunks():
    """Multiple TTSResponse audio chunks are concatenated."""
    tts = DoubaoTTS(api_key="k")
    chunks = [b"chunk1", b"chunk2", b"chunk3"]

    mock_ws = AsyncMock()
    mock_ws.recv = AsyncMock(
        side_effect=[
            _make_response_frame(
                MSG_FULL_SERVER_RESPONSE,
                EVENT_CONNECTION_STARTED,
                conn_id=b"cid12345",
                payload=b"{}",
            ),
            _make_response_frame(
                MSG_FULL_SERVER_RESPONSE,
                EVENT_SESSION_STARTED,
                session_id=b"testid123456",
                payload=b"{}",
            ),
            *[
                _make_response_frame(
                    MSG_AUDIO_ONLY_RESPONSE,
                    EVENT_TTS_RESPONSE,
                    session_id=b"testid123456",
                    payload=c,
                    serialization=0,
                )
                for c in chunks
            ],
            _make_response_frame(
                MSG_FULL_SERVER_RESPONSE,
                EVENT_SESSION_FINISHED,
                session_id=b"testid123456",
                payload=json.dumps({"status_code": 20000000, "message": "ok"}).encode(),
            ),
        ]
    )

    with patch("websockets.connect", MagicMock()) as mock_connect:
        mock_connect.return_value.__aenter__.return_value = mock_ws
        result = await tts.synthesize("hello")

    assert result == b"chunk1chunk2chunk3"


# ── Helpers ──


def _make_response_frame(
    msg_type: int,
    event: int,
    *,
    conn_id: bytes | None = None,
    session_id: bytes | None = None,
    payload: bytes = b"{}",
    serialization: int = 1,
) -> bytes:
    """Build a binary response frame for test mocks."""
    header = bytearray([0x11, (msg_type << 4) | 0x04, (serialization << 4), 0x00])
    body = bytearray()

    body.extend(struct.pack(">i", event))

    if conn_id is not None:
        body.extend(struct.pack(">I", len(conn_id)))
        body.extend(conn_id)
    elif session_id is not None:
        body.extend(struct.pack(">I", len(session_id)))
        body.extend(session_id)

    body.extend(struct.pack(">I", len(payload)))
    body.extend(payload)

    return bytes(header) + bytes(body)

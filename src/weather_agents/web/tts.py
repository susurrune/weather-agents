"""Doubao (Volcano Engine) TTS integration using V3 WebSocket bidirectional API.

The V3 protocol uses a custom binary frame format over WebSocket.
Session lifecycle: StartConnection → StartSession → TaskRequest → FinishSession.
"""

from __future__ import annotations

import json
import struct
import uuid
from typing import TYPE_CHECKING, Any

import websockets

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

_log = None


def _get_log():
    global _log
    if _log is None:
        from weather_agents.core.logger import get_logger

        _log = get_logger("tts")
    return _log


# ── Protocol constants ─────────────────────────────────────────────────────

# Message type (upstream)
MSG_FULL_CLIENT_REQUEST = 0x01

# Message type (downstream)
MSG_FULL_SERVER_RESPONSE = 0x09
MSG_AUDIO_ONLY_RESPONSE = 0x0B
MSG_ERROR = 0x0F

# Event codes
EVENT_START_CONNECTION = 1
EVENT_FINISH_CONNECTION = 2
EVENT_CONNECTION_STARTED = 50
EVENT_CONNECTION_FAILED = 51
EVENT_CONNECTION_FINISHED = 52
EVENT_START_SESSION = 100
EVENT_CANCEL_SESSION = 101
EVENT_FINISH_SESSION = 102
EVENT_SESSION_STARTED = 150
EVENT_SESSION_CANCELED = 151
EVENT_SESSION_FINISHED = 152
EVENT_SESSION_FAILED = 153
EVENT_TASK_REQUEST = 200
EVENT_TTS_SENTENCE_START = 350
EVENT_TTS_SENTENCE_END = 351
EVENT_TTS_RESPONSE = 352


# ── Binary protocol helpers ────────────────────────────────────────────────


def _build_request_frame(
    msg_type: int,
    event: int,
    payload_json: str | dict,
    session_id: str | None = None,
) -> bytes:
    """Build a binary TTS protocol request frame.

    Frame format:
      Byte 0: [v1(4)][header_size(4)] = 0x11
      Byte 1: [msg_type(4)][flags(4)]  — flags: bit2=has_event
      Byte 2: [serialization(4)][compression(4)]
      Byte 3: reserved = 0x00
      Bytes 4-7: event_code (int32 big-endian)
      [optionally: session_id_len(uint32) + session_id(bytes)]
      [payload_size(uint32) + payload(bytes)]
    """
    if isinstance(payload_json, dict):
        payload = json.dumps(payload_json, ensure_ascii=False).encode("utf-8")
    else:
        payload = payload_json.encode("utf-8")

    # v1, 4-byte header, client-req + has_event, JSON + no compression
    header = bytes([0x11, 0x14, 0x10, 0x00])
    body = bytearray()

    body.extend(struct.pack(">i", event))

    if session_id is not None:
        sid = session_id.encode("ascii")
        body.extend(struct.pack(">I", len(sid)))
        body.extend(sid)

    body.extend(struct.pack(">I", len(payload)))
    body.extend(payload)

    return header + bytes(body)


def _parse_response_frame(data: bytes) -> dict[str, Any]:
    """Parse a binary TTS protocol response frame."""
    if len(data) < 4:
        return {"msg_type": 0, "event": None}

    msg_type = data[1] >> 4
    specific_flags = data[1] & 0x0F
    serialization = data[2] >> 4

    offset = 4
    result: dict[str, Any] = {
        "msg_type": msg_type,
        "specific_flags": specific_flags,
        "serialization": serialization,
    }

    event = None
    if specific_flags & 0x04:
        event = struct.unpack(">i", data[offset : offset + 4])[0]
        offset += 4
    result["event"] = event

    # Optional ID fields (depends on event type)
    if event in (
        EVENT_CONNECTION_STARTED,
        EVENT_CONNECTION_FAILED,
        EVENT_CONNECTION_FINISHED,
    ):
        if offset + 4 <= len(data):
            id_len = struct.unpack(">I", data[offset : offset + 4])[0]
            offset += 4
            if id_len > 0 and offset + id_len <= len(data):
                result["conn_id"] = data[offset : offset + id_len].decode("utf-8", errors="replace")
                offset += id_len
    elif event in (  # noqa: SIM102
        EVENT_SESSION_STARTED,
        EVENT_SESSION_CANCELED,
        EVENT_SESSION_FINISHED,
        EVENT_SESSION_FAILED,
        EVENT_TTS_SENTENCE_START,
        EVENT_TTS_SENTENCE_END,
        EVENT_TTS_RESPONSE,
    ):
        if offset + 4 <= len(data):
            id_len = struct.unpack(">I", data[offset : offset + 4])[0]
            offset += 4
            if id_len > 0 and offset + id_len <= len(data):
                result["session_id"] = data[offset : offset + id_len].decode(
                    "utf-8", errors="replace"
                )
                offset += id_len

    # Payload
    if offset + 4 <= len(data):
        payload_len = struct.unpack(">I", data[offset : offset + 4])[0]
        offset += 4
        if payload_len > 0 and offset + payload_len <= len(data):
            raw = data[offset : offset + payload_len]
            result["payload_size"] = payload_len
            if serialization == 1:  # JSON
                try:
                    result["payload"] = json.loads(raw.decode("utf-8"))
                except Exception:
                    result["payload_raw"] = raw
            else:
                result["payload_raw"] = raw

    return result


# ── Main client ────────────────────────────────────────────────────────────


class DoubaoTTS:
    """Doubao TTS V3 client using WebSocket bidirectional streaming API.

    Converts text to speech via the Volcano Engine TTS service.
    Uses the WSS endpoint with a custom binary protocol.

    Requires either ``api_key`` (new console, recommended) or
    ``app_id`` + ``access_token`` (legacy console).
    """

    WS_ENDPOINT = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"

    def __init__(
        self,
        *,
        access_token: str | None = None,
        api_key: str | None = None,
        resource_id: str = "seed-tts-2.0",
        voice_type: str = "zh_female_cancan_mars_bigtts",
        encoding: str = "mp3",
        speed_ratio: float = 1.0,
        volume_ratio: float = 1.0,
        pitch_ratio: float = 1.0,
        emotion: str = "happy",
        app_id: str | None = None,
    ) -> None:
        self.access_token = access_token
        self.api_key = api_key
        self.resource_id = resource_id
        self.voice_type = voice_type
        self.encoding = encoding
        self.speed_ratio = speed_ratio
        self.volume_ratio = volume_ratio
        self.pitch_ratio = pitch_ratio
        self.emotion = emotion
        self.app_id = app_id

    # ── Public API ──

    async def synthesize(self, text: str) -> bytes:
        """Convert text to audio bytes via the V3 WebSocket API.

        Connects, creates a session, sends the text, and collects all
        audio frames before returning the complete MP3 (or configured
        encoding) bytes.
        """
        text = text.strip()
        if not text:
            return b""

        headers = self._get_ws_headers()
        session_id = uuid.uuid4().hex[:12]
        audio_chunks: list[bytes] = []

        async with websockets.connect(
            self.WS_ENDPOINT,
            additional_headers=headers,
            max_size=None,
        ) as ws:
            await self._start_connection(ws)
            await self._start_session(ws, session_id, text)
            await self._collect_audio(ws, session_id, audio_chunks)

        return b"".join(audio_chunks)

    # ── Connection helpers ──

    def _get_ws_headers(self) -> dict[str, str]:
        """Build WebSocket upgrade headers for authentication."""
        if self.api_key:
            # New console: X-Api-Key + X-Api-Resource-Id
            return {
                "X-Api-Key": self.api_key,
                "X-Api-Resource-Id": self.resource_id,
            }
        if self.app_id and self.access_token:
            # Old console: X-Api-App-Id + X-Api-Access-Key + X-Api-Resource-Id
            return {
                "X-Api-App-Id": self.app_id,
                "X-Api-Access-Key": self.access_token,
                "X-Api-Resource-Id": self.resource_id,
            }
        msg = "DoubaoTTS requires api_key (new console) or app_id+access_token (legacy)"
        raise ValueError(msg)

    @staticmethod
    def _map_rate(
        value: float,
        src_min: float,
        src_max: float,
        dst_min: int,
        dst_max: int,
    ) -> int:
        """Map a float ratio to the API's integer range."""
        if value <= 0:
            return dst_min
        src_range = src_max - src_min or 1
        norm = (value - src_min) / src_range
        norm = max(0.0, min(1.0, norm))
        return dst_min + round(norm * (dst_max - dst_min))

    def _make_session_payload(self, text: str) -> dict[str, Any]:
        """Build the StartSession payload with TTS parameters."""
        speed = self._map_rate(self.speed_ratio, 0.5, 2.0, -50, 100)
        loudness = self._map_rate(self.volume_ratio, 0.5, 2.0, -50, 100)

        audio_params: dict[str, Any] = {
            "format": self.encoding,
            "sample_rate": 24000,
        }
        if speed != 0:
            audio_params["speech_rate"] = speed
        if loudness != 0:
            audio_params["loudness_rate"] = loudness

        payload: dict[str, Any] = {
            "user": {"uid": "wa_voice"},
            "event": EVENT_START_SESSION,
            "req_params": {
                "text": text,
                "speaker": self.voice_type,
                "audio_params": audio_params,
            },
        }
        return payload

    async def _start_connection(self, ws: ClientConnection) -> None:
        """Send StartConnection and wait for ConnectionStarted."""
        await ws.send(_build_request_frame(MSG_FULL_CLIENT_REQUEST, EVENT_START_CONNECTION, {}))
        resp = await ws.recv()
        assert isinstance(resp, bytes), "TTS protocol expects binary frames"
        frame = _parse_response_frame(resp)
        if frame.get("event") == EVENT_CONNECTION_FAILED:
            err = frame.get("payload", {})
            msg_text = err.get("message") if isinstance(err, dict) else str(err)
            raise RuntimeError(f"TTS connection failed: {msg_text}")
        if frame.get("event") != EVENT_CONNECTION_STARTED:
            raise RuntimeError(f"TTS unexpected event {frame.get('event')} after StartConnection")

    async def _start_session(self, ws: ClientConnection, session_id: str, text: str) -> None:
        """Send StartSession with TTS params."""
        payload = self._make_session_payload(text)
        await ws.send(
            _build_request_frame(
                MSG_FULL_CLIENT_REQUEST,
                EVENT_START_SESSION,
                payload,
                session_id=session_id,
            )
        )

    async def _collect_audio(
        self,
        ws: ClientConnection,
        session_id: str,
        audio_chunks: list[bytes],
    ) -> None:
        """Receive events and collect audio chunks until session ends.

        Handles:
          - TTSResponse (352): audio data (binary or JSON-wrapped)
          - TTSSentenceStart / TTSSentenceEnd: metadata (skipped)
          - SessionFinished / SessionCanceled: end of session
          - SessionFailed / Error: raises
        """
        while True:
            resp = await ws.recv()
            assert isinstance(resp, bytes), "TTS protocol expects binary frames"
            frame = _parse_response_frame(resp)
            event = frame.get("event")
            msg_type = frame.get("msg_type")

            if event == EVENT_SESSION_FAILED:
                err = frame.get("payload", {})
                msg_text = err.get("message") if isinstance(err, dict) else str(err)
                raise RuntimeError(f"TTS session failed: {msg_text}")

            if event in (EVENT_SESSION_FINISHED, EVENT_SESSION_CANCELED):
                break

            if event == EVENT_TTS_RESPONSE:
                # Audio-only frames carry raw binary payload
                if msg_type == MSG_AUDIO_ONLY_RESPONSE:
                    raw = frame.get("payload_raw")
                    if raw:
                        audio_chunks.append(raw)
                # Full-server response might carry base64 data
                elif msg_type == MSG_FULL_SERVER_RESPONSE:
                    payload = frame.get("payload", {})
                    data_b64 = payload.get("data") if isinstance(payload, dict) else None
                    if data_b64:
                        import base64

                        audio_chunks.append(base64.b64decode(data_b64))

            if event == EVENT_TTS_SENTENCE_START:
                pass  # metadata only, skip
            if event == EVENT_TTS_SENTENCE_END:
                pass  # metadata only, skip

    @property
    def is_available(self) -> bool:
        return bool(self.api_key) or (bool(self.app_id) and bool(self.access_token))

"""Doubao (Volcano Engine) TTS integration using V3 HTTP Unidirectional API.

Uses POST to https://openspeech.bytedance.com/api/v3/tts/unidirectional
to convert text to speech. Returns audio bytes (MP3 by default).
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx

_log = None


def _get_log():
    global _log
    if _log is None:
        from weather_agents.core.logger import get_logger

        _log = get_logger("tts")
    return _log


HTTP_ENDPOINT = "https://openspeech.bytedance.com/api/v3/tts/unidirectional"

# ── Voice catalog ────────────────────────────────────────────────────────────

VOICE_CATALOG: list[dict[str, str]] = [
    {
        "key": "xiaohe",
        "name": "小河",
        "desc": "温柔自然女声",
        "voice_type": "zh_female_xiaohe_uranus_bigtts",
    },
    {
        "key": "qingxinnvsheng",
        "name": "清新女声",
        "desc": "清澈自然女声",
        "voice_type": "zh_female_qingxinnvsheng_uranus_bigtts",
    },
    {
        "key": "cancan",
        "name": "灿灿",
        "desc": "活力甜美少女音",
        "voice_type": "zh_female_cancan_uranus_bigtts",
    },
    {
        "key": "sajiaoxuemei",
        "name": "撒娇雪梅",
        "desc": "甜美撒娇少女音",
        "voice_type": "zh_female_sajiaoxuemei_uranus_bigtts",
    },
    {
        "key": "meilinvyou",
        "name": "魅力女游",
        "desc": "温柔魅力女声",
        "voice_type": "zh_female_meilinvyou_uranus_bigtts",
    },
    {
        "key": "uranus",
        "name": "乌拉努斯",
        "desc": "大气知性女声",
        "voice_type": "zh_female_vv_uranus_bigtts",
    },
    {
        "key": "tianmeitaozi",
        "name": "甜美桃子",
        "desc": "甜美软萌少女音",
        "voice_type": "zh_female_tianmeitaozi_uranus_bigtts",
    },
]


def get_voice_by_key(key: str) -> dict[str, str] | None:
    """Look up a voice by its short key."""
    for v in VOICE_CATALOG:
        if v["key"] == key:
            return v
    return None


def get_voice_type(key_or_type: str) -> str:
    """Resolve a key or voice_type string to a voice_type ID.

    If the input matches a catalog key, returns the corresponding
    voice_type.  Otherwise returns the input unchanged (assumed to
    already be a voice_type ID).
    """
    entry = get_voice_by_key(key_or_type)
    return entry["voice_type"] if entry else key_or_type


class DoubaoTTS:
    """Doubao TTS V3 client using HTTP Unidirectional API.

    Converts text to speech via the Volcano Engine TTS service.
    Uses HTTP POST to send text and receive audio directly.

    Requires either ``api_key`` (new console, recommended) or
    ``app_id`` + ``access_token`` (legacy console).
    """

    def __init__(
        self,
        *,
        access_token: str | None = None,
        api_key: str | None = None,
        resource_id: str = "seed-tts-2.0",
        voice_type: str = "zh_female_sajiaoxuemei_uranus_bigtts",
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
        """Convert text to audio bytes via the V3 HTTP Unidirectional API.

        Posts the text to the TTS endpoint and returns the raw audio
        bytes (MP3 by default).

        The API streams audio as multiple JSON lines, each containing
        a base64 chunk of the audio data.  All chunks are concatenated.
        """
        text = text.strip()
        if not text:
            return b""

        headers = self._get_http_headers()
        body = self._make_request_body(text)

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(HTTP_ENDPOINT, json=body, headers=headers)

        if resp.status_code != 200:
            raise RuntimeError(f"TTS HTTP {resp.status_code}: {resp.text[:200]}")

        if not resp.content:
            raise RuntimeError("TTS empty response")

        # The API streams audio as multiple JSON lines:
        #   {"code":0,"message":"","data":"<base64_audio_chunk>"}
        #   {"code":0,"message":"","data":"<base64_audio_chunk>"}
        #   ...
        # Each line is a separate chunk. Concatenate all data fields,
        # then decode once.
        audio_chunks: list[bytes] = []
        for line in resp.text.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            code = payload.get("code", 0)
            if code == 20000000:
                # End-of-stream signal, no more data
                break
            if code != 0:
                msg = payload.get("message", "unknown error")
                raise RuntimeError(f"TTS API error ({code}): {msg}")

            raw_data = payload.get("data")
            if isinstance(raw_data, str):
                audio_chunks.append(base64.b64decode(raw_data))
            elif isinstance(raw_data, dict):
                audio_b64 = raw_data.get("audio")
                if audio_b64:
                    audio_chunks.append(base64.b64decode(audio_b64))

        if not audio_chunks:
            raise RuntimeError("TTS response missing data field")

        return b"".join(audio_chunks)

    # ── Request helpers ──

    def _get_http_headers(self) -> dict[str, str]:
        """Build HTTP headers for authentication."""
        if self.api_key:
            return {
                "X-Api-Key": self.api_key,
                "X-Api-Resource-Id": self.resource_id,
            }
        if self.app_id and self.access_token:
            return {
                "X-Api-App-Id": self.app_id,
                "X-Api-Access-Key": self.access_token,
                "X-Api-Resource-Id": self.resource_id,
            }
        msg = "DoubaoTTS requires api_key (new console) or app_id+access_token (legacy)"
        raise ValueError(msg)

    def _make_request_body(self, text: str) -> dict[str, Any]:
        """Build the JSON request body with TTS parameters."""
        audio_params: dict[str, Any] = {
            "format": self.encoding,
            "sample_rate": 24000,
        }
        speed = self._map_rate(self.speed_ratio, 0.5, 2.0, -50, 100)
        loudness = self._map_rate(self.volume_ratio, 0.5, 2.0, -50, 100)
        if speed != 0:
            audio_params["speech_rate"] = speed
        if loudness != 0:
            audio_params["loudness_rate"] = loudness

        return {
            "user": {"uid": "wa_voice"},
            "req_params": {
                "text": text,
                "speaker": self.voice_type,
                "audio_params": audio_params,
            },
        }

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

    @property
    def is_available(self) -> bool:
        return bool(self.api_key) or (bool(self.app_id) and bool(self.access_token))

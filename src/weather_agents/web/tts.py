"""Doubao (Volcano Engine) TTS integration using V3 HTTP Unidirectional API.

Uses POST to https://openspeech.bytedance.com/api/v3/tts/unidirectional
to convert text to speech. Returns audio bytes (MP3 by default).
"""

from __future__ import annotations

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

        # The API returns JSON errors with 200 status (code != 0 means error)
        content_type = resp.headers.get("content-type", "")
        if "json" in content_type or (len(resp.content) < 200 and "audio" not in content_type):
            try:
                err_data = json.loads(resp.content)
                code = err_data.get("code", 0)
                if code != 0:
                    msg = err_data.get("message", "unknown error")
                    raise RuntimeError(f"TTS API error ({code}): {msg}")
            except (json.JSONDecodeError, ValueError):
                pass  # not JSON, treat as audio

        return resp.content

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

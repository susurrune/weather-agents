"""Additional TTS providers — Edge, OpenAI, Azure, ElevenLabs, Fish Audio.

Each implements synthesize_stream(text), synthesize(text), is_available,
close().  Use ``create_provider(cfg)`` to instantiate.  All via HTTP REST —
no vendor SDKs.

This is intentionally a SEPARATE file from ``tts.py`` so the original
DoubaoTTS code path in ``server.py`` is never touched by multi-provider
logic.  If ``cfg.tts.provider`` is ``"doubao"`` (or unset), server.py
uses DoubaoTTS directly — exactly as it always has.
"""

from __future__ import annotations

import base64
import os as _os

import httpx


class EdgeTTS:
    """Microsoft Edge TTS — free, no API key, high-quality Chinese voices."""

    def __init__(
        self,
        *,
        voice: str = "zh-CN-XiaoxiaoNeural",
        rate: str = "+0%",
        volume: str = "+0%",
        **_: object,
    ) -> None:
        self.voice = voice
        self.rate = rate
        self.volume = volume

    @property
    def is_available(self) -> bool:
        try:
            import edge_tts  # noqa: F401

            return True
        except ImportError:
            return False

    async def synthesize_stream(self, text: str):
        import edge_tts

        text = text.strip()
        if not text:
            return
        comm = edge_tts.Communicate(text, self.voice, rate=self.rate, volume=self.volume)
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                yield base64.b64encode(chunk["data"]).decode()

    async def synthesize(self, text: str) -> bytes:
        chunks: list[bytes] = []
        async for b64 in self.synthesize_stream(text):
            chunks.append(base64.b64decode(b64))
        return b"".join(chunks)

    async def close(self) -> None:
        pass


class OpenAITTS:
    """OpenAI tts-1 / tts-1-hd. Needs api_key."""

    def __init__(
        self,
        *,
        api_key: str = "",
        voice: str = "nova",
        model: str = "tts-1",
        speed: float = 1.0,
        **_: object,
    ) -> None:
        self.api_key = api_key or _os.environ.get("OPENAI_API_KEY", "")
        self.voice = voice
        self.model = model
        self.speed = speed
        self._client: httpx.AsyncClient | None = None

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    async def synthesize_stream(self, text: str):
        text = text.strip()
        if not text or not self.is_available:
            return
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30)
        resp = await self._client.post(
            "https://api.openai.com/v1/audio/speech",
            json={
                "model": self.model,
                "input": text,
                "voice": self.voice,
                "speed": self.speed,
                "response_format": "mp3",
            },
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"OpenAI TTS HTTP {resp.status_code}: {resp.text[:200]}")
        yield base64.b64encode(resp.content).decode()

    async def synthesize(self, text: str) -> bytes:
        async for b64 in self.synthesize_stream(text):
            return base64.b64decode(b64)
        return b""

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


class AzureTTS:
    """Azure Cognitive Services Speech. Needs api_key + region."""

    def __init__(
        self,
        *,
        api_key: str = "",
        region: str = "eastasia",
        voice: str = "zh-CN-XiaoxiaoNeural",
        **_: object,
    ) -> None:
        self.api_key = api_key or _os.environ.get("AZURE_SPEECH_KEY", "")
        self.region = region or _os.environ.get("AZURE_SPEECH_REGION", "eastasia")
        self.voice = voice
        self._client: httpx.AsyncClient | None = None

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    async def synthesize_stream(self, text: str):
        text = text.strip()
        if not text or not self.is_available:
            return
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30)
        ssml = f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="zh-CN"><voice name="{self.voice}">{text}</voice></speak>'
        resp = await self._client.post(
            f"https://{self.region}.tts.speech.microsoft.com/cognitiveservices/v1",
            content=ssml.encode("utf-8"),
            headers={
                "Ocp-Apim-Subscription-Key": self.api_key,
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": "audio-16khz-128kbitrate-mono-mp3",
            },
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Azure TTS HTTP {resp.status_code}: {resp.text[:200]}")
        yield base64.b64encode(resp.content).decode()

    async def synthesize(self, text: str) -> bytes:
        async for b64 in self.synthesize_stream(text):
            return base64.b64decode(b64)
        return b""

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


class ElevenLabsTTS:
    """ElevenLabs — highest voice quality. Needs api_key."""

    def __init__(
        self, *, api_key: str = "", voice: str = "21m00Tcm4TlvDq8ikWAM", **_: object
    ) -> None:
        self.api_key = api_key or _os.environ.get("ELEVENLABS_API_KEY", "")
        self.voice = voice
        self._client: httpx.AsyncClient | None = None

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    async def synthesize_stream(self, text: str):
        text = text.strip()
        if not text or not self.is_available:
            return
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30)
        resp = await self._client.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice}",
            json={
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            },
            headers={"xi-api-key": self.api_key},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"ElevenLabs TTS HTTP {resp.status_code}: {resp.text[:200]}")
        yield base64.b64encode(resp.content).decode()

    async def synthesize(self, text: str) -> bytes:
        async for b64 in self.synthesize_stream(text):
            return base64.b64decode(b64)
        return b""

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


class FishAudioTTS:
    """Fish Audio — great Chinese + Japanese. Needs api_key from fish.audio."""

    def __init__(
        self, *, api_key: str = "", voice: str = "753e7755e5be920fe8d4305a273af908", **_: object
    ) -> None:
        self.api_key = api_key or _os.environ.get("FISH_AUDIO_API_KEY", "")
        self.voice = voice
        self._client: httpx.AsyncClient | None = None

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    async def synthesize_stream(self, text: str):
        text = text.strip()
        if not text or not self.is_available:
            return
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30)
        resp = await self._client.post(
            "https://api.fish.audio/v1/tts",
            json={"text": text, "reference_id": self.voice, "format": "mp3"},
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Fish Audio HTTP {resp.status_code}: {resp.text[:200]}")
        yield base64.b64encode(resp.content).decode()

    async def synthesize(self, text: str) -> bytes:
        async for b64 in self.synthesize_stream(text):
            return base64.b64decode(b64)
        return b""

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


_PROVIDER_MAP: dict[str, type] = {
    "edge": EdgeTTS,
    "openai": OpenAITTS,
    "azure": AzureTTS,
    "elevenlabs": ElevenLabsTTS,
    "fishaudio": FishAudioTTS,
}


def create_provider(cfg) -> object | None:
    """Build a non-Doubao TTS engine from config, or None."""
    provider = cfg.tts.provider
    cls = _PROVIDER_MAP.get(provider)
    if cls is None:
        return None
    keys: dict = getattr(cfg.tts, "api_keys", {}) or {}
    if not isinstance(keys, dict):
        keys = {}
    api_key = keys.get(provider, "")
    kwargs: dict = {"api_key": api_key}
    if provider == "azure":
        kwargs["region"] = keys.get("azure_region", "eastasia")
    voice_cfg = getattr(cfg.tts, "voice_type", "")
    if voice_cfg:
        kwargs["voice"] = voice_cfg
    try:
        return cls(**kwargs)
    except Exception:
        return None

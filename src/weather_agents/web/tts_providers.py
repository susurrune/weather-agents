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
import json
import os as _os
import time as _time

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


# ═══════════════════════════════════════════════════════════════════════════
# Chinese-vendor TTS — need api_key + api_secret, use signature auth
# ═══════════════════════════════════════════════════════════════════════════


class IflytekTTS:
    """科大讯飞 — best Chinese voice quality. Needs APPID + APISecret from xfyun.cn."""

    _HOST = "tts-api.xfyun.cn"

    def __init__(
        self, *, api_key: str = "", api_secret: str = "", voice: str = "xiaoyan", **_: object
    ) -> None:
        self.app_id = api_key or _os.environ.get("IFLYTEK_APP_ID", "")
        self.api_secret = api_secret or _os.environ.get("IFLYTEK_API_SECRET", "")
        self.voice = voice
        self._client: httpx.AsyncClient | None = None

    @property
    def is_available(self) -> bool:
        return bool(self.app_id) and bool(self.api_secret)

    async def synthesize_stream(self, text: str):
        import hmac as _hmac
        import urllib.parse as _uparse

        text = text.strip()
        if not text or not self.is_available:
            return
        ts = _time.strftime("%a, %d %b %Y %H:%M:%S GMT", _time.gmtime())
        raw = f"host: {self._HOST}\ndate: {ts}\nGET /v2/tts HTTP/1.1"
        sig = base64.b64encode(
            _hmac.new(self.api_secret.encode(), raw.encode(), "sha256").digest()
        ).decode()
        auth = f'api_key="{self.app_id}", algorithm="hmac-sha256", headers="host date request-line", signature="{sig}"'
        params = {
            "text": base64.b64encode(text.encode()).decode(),
            "voice_name": self.voice,
            "audio_format": "mp3",
            "sample_rate": "16000",
        }
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30)
        resp = await self._client.get(
            f"https://{self._HOST}/v2/tts?{_uparse.urlencode(params)}",
            headers={"Authorization": auth},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"讯飞 TTS HTTP {resp.status_code}")
        ct = resp.headers.get("content-type", "")
        if "audio" in ct:
            yield base64.b64encode(resp.content).decode()
        else:
            data = resp.json()
            if data.get("code") != 0:
                raise RuntimeError(f"讯飞 TTS {data.get('code')}: {data.get('message', '')}")

    async def synthesize(self, text: str) -> bytes:
        async for b64 in self.synthesize_stream(text):
            return base64.b64decode(b64)
        return b""

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


class TencentTTS:
    """腾讯云语音合成 — needs SecretId + SecretKey from cloud.tencent.com."""

    _ENDPOINT = "tts.tencentcloudapi.com"

    def __init__(
        self, *, api_key: str = "", api_secret: str = "", voice: str = "101001", **_: object
    ) -> None:
        self.secret_id = api_key or _os.environ.get("TENCENT_SECRET_ID", "")
        self.secret_key = api_secret or _os.environ.get("TENCENT_SECRET_KEY", "")
        self.voice = voice
        self._client: httpx.AsyncClient | None = None

    @property
    def is_available(self) -> bool:
        return bool(self.secret_id) and bool(self.secret_key)

    async def synthesize_stream(self, text: str):
        import hashlib as _hl
        import hmac as _hmac
        import uuid as _uuid

        text = text.strip()
        if not text or not self.is_available:
            return
        ts = int(_time.time())
        payload = json.dumps(
            {
                "Text": text,
                "SessionId": str(_uuid.uuid4()),
                "ModelType": 1,
                "VoiceType": int(self.voice) if self.voice.isdigit() else 101001,
                "Codec": "mp3",
            }
        )
        svc, host, algo = "tts", self._ENDPOINT, "TC3-HMAC-SHA256"
        date = _time.strftime("%Y-%m-%d", _time.gmtime(ts))
        sd = _hmac.new(("TC3" + self.secret_key).encode(), date.encode(), "sha256").digest()
        ss = _hmac.new(sd, svc.encode(), "sha256").digest()
        sk = _hmac.new(ss, b"tc3_request", "sha256").digest()
        ct = "application/json; charset=utf-8"
        ch = f"content-type:{ct}\nhost:{host}\nx-tc-action:texttospeech\n"
        cr = f"POST\n/\n\n{ch}\ncontent-type;host;x-tc-action\n{_hl.sha256(payload.encode()).hexdigest()}"
        cs = f"{date}/{svc}/tc3_request"
        sts = f"{algo}\n{ts}\n{cs}\n{_hl.sha256(cr.encode()).hexdigest()}"
        sig = _hmac.new(sk, sts.encode(), "sha256").hexdigest()
        auth = f"{algo} Credential={self.secret_id}/{cs}, SignedHeaders=content-type;host;x-tc-action, Signature={sig}"
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30)
        resp = await self._client.post(
            f"https://{host}",
            content=payload,
            headers={
                "Authorization": auth,
                "Content-Type": ct,
                "Host": host,
                "X-TC-Action": "TextToSpeech",
                "X-TC-Version": "2019-08-23",
                "X-TC-Timestamp": str(ts),
                "X-TC-Region": "ap-guangzhou",
            },
        )
        if resp.status_code != 200:
            raise RuntimeError(f"腾讯云 TTS HTTP {resp.status_code}")
        data = resp.json()
        if "Response" in data and "Error" in data["Response"]:
            e = data["Response"]["Error"]
            raise RuntimeError(f"腾讯云 TTS {e.get('Code')}: {e.get('Message')}")
        b64 = data.get("Response", {}).get("Audio", "")
        if b64:
            yield b64

    async def synthesize(self, text: str) -> bytes:
        async for b64 in self.synthesize_stream(text):
            return base64.b64decode(b64)
        return b""

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


class AliyunTTS:
    """阿里云智能语音 — needs AccessKey ID + Secret from ram.console.aliyun.com."""

    def __init__(
        self, *, api_key: str = "", api_secret: str = "", voice: str = "xiaoyun", **_: object
    ) -> None:
        self.access_id = api_key or _os.environ.get("ALIBABA_ACCESS_KEY_ID", "")
        self.access_secret = api_secret or _os.environ.get("ALIBABA_ACCESS_KEY_SECRET", "")
        self.voice = voice
        self._token = ""
        self._tok_exp = 0.0
        self._client: httpx.AsyncClient | None = None

    @property
    def is_available(self) -> bool:
        return bool(self.access_id) and bool(self.access_secret)

    async def synthesize_stream(self, text: str):
        text = text.strip()
        if not text or not self.is_available:
            return
        if not self._token or _time.time() >= self._tok_exp - 60:
            if self._client is None:
                self._client = httpx.AsyncClient(timeout=30)
            resp = await self._client.post(
                "https://nls-meta.cn-shanghai.aliyuncs.com/pop/2018-05-18/tokens",
                json={"AccessKeyId": self.access_id, "Version": "2019-02-28"},
            )
            data = resp.json()
            if data.get("ErrMsg") != "Successful":
                raise RuntimeError(f"阿里云 token: {data}")
            self._token = data["Token"]["Id"]
            self._tok_exp = _time.time() + 1800
        body = json.dumps(
            {
                "payload": {"text": text},
                "parameters": {"voice_name": self.voice, "format": "mp3", "volume": 70},
                "context": {"device_id": "skyloom"},
            }
        )
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30)
        resp = await self._client.post(
            "https://nls-gateway-cn-shanghai.aliyuncs.com/stream/v1/tts",
            content=body,
            headers={"X-NLS-Token": self._token, "Content-Type": "application/json"},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"阿里云 TTS HTTP {resp.status_code}")
        yield base64.b64encode(resp.content).decode()

    async def synthesize(self, text: str) -> bytes:
        async for b64 in self.synthesize_stream(text):
            return base64.b64decode(b64)
        return b""

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


class BaiduTTS:
    """百度语音合成 — needs API Key + Secret Key from console.bce.baidu.com."""

    def __init__(
        self, *, api_key: str = "", api_secret: str = "", voice: str = "0", **_: object
    ) -> None:
        self.api_key = api_key or _os.environ.get("BAIDU_TTS_APP_ID", "")
        self.api_secret = api_secret or _os.environ.get("BAIDU_TTS_APP_KEY", "")
        self.voice = voice
        self._token = ""
        self._tok_exp = 0.0
        self._client: httpx.AsyncClient | None = None

    @property
    def is_available(self) -> bool:
        return bool(self.api_key) and bool(self.api_secret)

    async def synthesize_stream(self, text: str):
        text = text.strip()
        if not text or not self.is_available:
            return
        if not self._token or _time.time() >= self._tok_exp - 60:
            if self._client is None:
                self._client = httpx.AsyncClient(timeout=30)
            resp = await self._client.get(
                f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={self.api_key}&client_secret={self.api_secret}"
            )
            if resp.status_code != 200:
                raise RuntimeError(f"百度 token HTTP {resp.status_code}")
            data = resp.json()
            self._token = data["access_token"]
            self._tok_exp = _time.time() + data.get("expires_in", 2592000) - 300
        body = json.dumps(
            {
                "tex": text,
                "tok": self._token,
                "cuid": "skyloom",
                "ctp": 1,
                "lan": "zh",
                "spd": 5,
                "pit": 5,
                "vol": 5,
                "per": int(self.voice) if self.voice.isdigit() else 0,
                "aue": 3,
            }
        )
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30)
        resp = await self._client.post(
            "https://tsn.baidu.com/text2audio",
            content=body,
            headers={"Content-Type": "application/json"},
        )
        ct = resp.headers.get("content-type", "")
        if "audio" in ct:
            yield base64.b64encode(resp.content).decode()
        else:
            data = resp.json()
            raise RuntimeError(f"百度 TTS {data.get('err_no')}: {data.get('err_msg', '')}")

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
    "iflytek": IflytekTTS,
    "tencent": TencentTTS,
    "aliyun": AliyunTTS,
    "baidu": BaiduTTS,
}

_NEEDS_SECRET = {"iflytek", "tencent", "aliyun", "baidu"}


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
    # Chinese vendors need api_secret alongside api_key
    if provider in _NEEDS_SECRET:
        secrets: dict = getattr(cfg.tts, "api_secrets", {}) or {}
        if isinstance(secrets, dict):
            kwargs["api_secret"] = secrets.get(provider, "")
    if provider == "azure":
        kwargs["region"] = keys.get("azure_region", "eastasia")
    voice_cfg = getattr(cfg.tts, "voice_type", "")
    if voice_cfg:
        kwargs["voice"] = voice_cfg
    try:
        return cls(**kwargs)
    except Exception:
        return None

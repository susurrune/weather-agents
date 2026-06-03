"""Cloudflare Quick Tunnel — expose the local voice server on a public URL.

Runs ``cloudflared tunnel --url http://127.0.0.1:<port>``, which needs no
Cloudflare account or config and returns an ephemeral
``https://<random>.trycloudflare.com`` address. A phone opens that URL to reach
the desktop over the internet; Cloudflare's HTTPS edge also unlocks the browser
microphone (blocked on a plain ``http://<LAN-IP>`` page).

``cloudflared`` is an optional external binary — everything here degrades
gracefully when it isn't installed.
"""

from __future__ import annotations

import asyncio
import re
import shutil

# trycloudflare hands out URLs like https://foo-bar-baz.trycloudflare.com
_TUNNEL_URL_RE = re.compile(r"https://[a-z0-9][a-z0-9-]*\.trycloudflare\.com")


def extract_tunnel_url(text: str) -> str | None:
    """Pull the first trycloudflare URL out of cloudflared's log output."""
    m = _TUNNEL_URL_RE.search(text)
    return m.group(0) if m else None


def cloudflared_path() -> str | None:
    """Absolute path to the ``cloudflared`` binary, or None if not installed."""
    return shutil.which("cloudflared")


class CloudflareTunnel:
    """Manage a ``cloudflared`` Quick Tunnel subprocess.

    Usage::

        t = CloudflareTunnel(port=8765)
        url = await t.start()      # https://...trycloudflare.com or None
        ...
        await t.close()
    """

    def __init__(self, port: int, host: str = "127.0.0.1") -> None:
        self.port = port
        self.host = host
        self.url: str | None = None
        self._proc: asyncio.subprocess.Process | None = None

    @staticmethod
    def is_available() -> bool:
        return cloudflared_path() is not None

    async def start(self, timeout: float = 30.0) -> str | None:
        """Launch the tunnel and return its public URL (or None on failure).

        Reads cloudflared's stdout/stderr until the trycloudflare URL appears
        or ``timeout`` elapses. The process keeps running until ``close()``.
        """
        exe = cloudflared_path()
        if not exe:
            return None
        self._proc = await asyncio.create_subprocess_exec(
            exe,
            "tunnel",
            "--url",
            f"http://{self.host}:{self.port}",
            "--no-autoupdate",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            self.url = await asyncio.wait_for(self._read_url(), timeout=timeout)
        except TimeoutError:
            self.url = None
        return self.url

    async def _read_url(self) -> str | None:
        assert self._proc is not None and self._proc.stdout is not None
        while True:
            raw = await self._proc.stdout.readline()
            if not raw:  # process exited before printing a URL
                return None
            url = extract_tunnel_url(raw.decode("utf-8", errors="ignore"))
            if url:
                return url

    async def close(self) -> None:
        """Terminate the tunnel subprocess (best effort)."""
        proc = self._proc
        if proc is None or proc.returncode is not None:
            return
        try:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except TimeoutError:
                proc.kill()
        except ProcessLookupError:
            pass

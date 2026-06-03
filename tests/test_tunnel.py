"""Tests for the Cloudflare Quick Tunnel helper (pure logic, no subprocess)."""

from __future__ import annotations

import pytest

from weather_agents.web.tunnel import CloudflareTunnel, extract_tunnel_url


class TestExtractTunnelUrl:
    def test_finds_url_in_log_line(self):
        line = "2026-06-01 INF |  https://misty-fog-1234.trycloudflare.com  |"
        assert extract_tunnel_url(line) == "https://misty-fog-1234.trycloudflare.com"

    def test_returns_none_without_url(self):
        assert extract_tunnel_url("starting tunnel, registering connection...") is None

    def test_ignores_other_https(self):
        assert extract_tunnel_url("see https://example.com for docs") is None

    def test_first_match_wins(self):
        text = "a https://one.trycloudflare.com b https://two.trycloudflare.com"
        assert extract_tunnel_url(text) == "https://one.trycloudflare.com"


class TestCloudflareTunnel:
    @pytest.mark.asyncio
    async def test_start_returns_none_when_binary_missing(self, monkeypatch):
        import weather_agents.web.tunnel as t

        monkeypatch.setattr(t, "cloudflared_path", lambda: None)
        tunnel = CloudflareTunnel(port=8765)
        assert await tunnel.start(timeout=1.0) is None
        assert tunnel.url is None

    @pytest.mark.asyncio
    async def test_close_is_safe_when_never_started(self):
        await CloudflareTunnel(port=8765).close()  # must not raise

    def test_is_available_reflects_path(self, monkeypatch):
        import weather_agents.web.tunnel as t

        monkeypatch.setattr(t, "cloudflared_path", lambda: "/usr/bin/cloudflared")
        assert CloudflareTunnel.is_available() is True
        monkeypatch.setattr(t, "cloudflared_path", lambda: None)
        assert CloudflareTunnel.is_available() is False

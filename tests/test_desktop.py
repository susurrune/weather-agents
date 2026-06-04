"""Tests for the desktop app helpers (pure logic — no window/subprocess)."""

from __future__ import annotations

from weather_agents.web.desktop import _splash_url, share_target


def test_share_target_without_public_url_is_local():
    assert share_target("http://127.0.0.1:8765", None) == "http://127.0.0.1:8765"


def test_share_target_encodes_public_url():
    out = share_target("http://127.0.0.1:8765", "https://misty-fog.trycloudflare.com")
    assert out.startswith("http://127.0.0.1:8765/?share=")
    # URL is percent-encoded so it survives as a single query value
    assert "https%3A%2F%2Fmisty-fog.trycloudflare.com" in out


def test_splash_url_is_self_contained_data_uri():
    url = _splash_url()
    assert url.startswith("data:text/html;charset=utf-8,")
    # The branding + spinner must be present so the window renders without a server.
    decoded = url[len("data:text/html;charset=utf-8,") :]
    from urllib.parse import unquote

    html = unquote(decoded)
    assert "SKYLOOM" in html
    assert "@keyframes spin" in html

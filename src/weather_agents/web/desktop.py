"""Desktop app — a native window running the voice client, with an optional
Cloudflare tunnel so a phone can join over a public URL.

Architecture: the aiohttp voice server runs in a background thread's event
loop; once it's listening, an ``on_started`` hook opens the Cloudflare tunnel
and records the public URL. The native window (pywebview) must own the main
thread, so it's created here after the server signals ready. Everything is
optional — without pywebview we fall back to the default browser; without
``cloudflared`` we just serve locally + on the LAN.
"""

from __future__ import annotations

import threading
import urllib.parse


def share_target(local_url: str, public_url: str | None) -> str:
    """Window URL — carries the public link as ?share= so the page can show a
    'scan me' banner with a QR for the phone."""
    if not public_url:
        return local_url
    return f"{local_url}/?share={urllib.parse.quote(public_url, safe='')}"


def _print_qr(url: str) -> None:
    """Print an ASCII QR of ``url`` to the terminal, if the qrcode lib is present."""
    try:
        import qrcode  # optional dep

        qr = qrcode.QRCode(border=1)
        qr.add_data(url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
    except Exception:
        pass


def _print_banner(local_url: str, public_url: str | None, tunnel_enabled: bool) -> None:
    print(f"\n  🖥  桌面端已启动  →  {local_url}")
    if public_url:
        print(f"  📱  手机访问（公网 HTTPS）→  {public_url}")
        print("      用手机浏览器打开上面的网址，或扫码：\n")
        _print_qr(public_url)
    elif tunnel_enabled:
        print("  📱  未检测到 cloudflared，暂无法生成公网地址。")
        print("      安装 cloudflared 后重开即可手机访问：")
        print(
            "      https://developers.cloudflare.com/cloudflare-one/"
            "connections/connect-apps/install-and-setup/installation/"
        )
    print()


def run_desktop_app(
    agent_name: str = "fair",
    port: int = 8765,
    tunnel: bool = True,
    window: bool = True,
) -> None:
    """Launch the desktop voice app. Blocks until the window/server is closed."""
    import asyncio

    from weather_agents.web.server import run_voice_server
    from weather_agents.web.tunnel import CloudflareTunnel

    state: dict = {"public_url": None, "tunnel": None, "loop": None, "error": None}
    ready = threading.Event()

    async def _on_started() -> None:
        # The local server is now accepting connections.
        try:
            if tunnel and CloudflareTunnel.is_available():
                t = CloudflareTunnel(port=port)
                state["tunnel"] = t
                state["public_url"] = await t.start(timeout=30.0)
        finally:
            ready.set()

    def _bg() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        state["loop"] = loop
        try:
            loop.run_until_complete(
                run_voice_server(
                    host="127.0.0.1",
                    port=port,
                    agent_name=agent_name,
                    on_started=_on_started,
                )
            )
        except Exception as exc:  # noqa: BLE001 — surface to the main thread
            state["error"] = exc
            ready.set()

    threading.Thread(target=_bg, daemon=True).start()

    if not ready.wait(timeout=45.0):
        print("  ⚠ 服务器启动超时。")
    if state["error"]:
        print(f"  ⚠ 启动失败: {state['error']}")
        return

    local_url = f"http://127.0.0.1:{port}"
    _print_banner(local_url, state["public_url"], tunnel)
    target = share_target(local_url, state["public_url"])

    def _close_tunnel() -> None:
        import contextlib

        t, loop = state["tunnel"], state["loop"]
        if t and loop:
            with contextlib.suppress(Exception):
                asyncio.run_coroutine_threadsafe(t.close(), loop).result(timeout=6)

    if window:
        try:
            import webview  # optional dep (pywebview)

            webview.create_window("Weather Agents", target, width=920, height=760)
            try:
                webview.start()
            finally:
                _close_tunnel()
            return
        except ImportError:
            print(
                "  pywebview 未安装，改用默认浏览器打开。"
                "（pip install 'weather-agents[desktop]' 可获得原生窗口）"
            )

    # Fallback: open in the default browser, keep serving until Ctrl-C.
    import webbrowser

    webbrowser.open(target)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        _close_tunnel()

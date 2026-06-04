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

# Shown the instant the window opens, while the server + Cloudflare tunnel are
# still coming up (can take tens of seconds on a cold cloudflared). Without it
# the user stares at an empty window — or, in the old flow, at nothing at all
# until startup finished. Pure inline HTML so it needs no server to render.
_SPLASH_HTML = """<!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Skyloom</title><style>
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:100%}
body{background:#2a1f17;color:#f3ead7;font-family:system-ui,-apple-system,sans-serif;
display:flex;flex-direction:column;align-items:center;justify-content:center;gap:26px}
.mark{position:relative;width:96px;height:96px}
.sun{position:absolute;inset:0;margin:auto;width:40px;height:40px;border-radius:50%;
background:radial-gradient(circle at 50% 45%,#f0be6e,#d4a056);
box-shadow:0 0 30px rgba(212,160,86,.5);animation:pulse 2.4s ease-in-out infinite}
.ring{position:absolute;inset:0;border:1.5px solid rgba(212,160,86,.25);border-top-color:#d4a056;
border-radius:50%;animation:spin 1.4s linear infinite}
h1{font-size:24px;font-weight:300;letter-spacing:.42em;padding-left:.42em}
.tip{font-size:13px;color:rgba(243,234,215,.45);letter-spacing:.04em}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes pulse{0%,100%{transform:scale(1);opacity:1}50%{transform:scale(.86);opacity:.8}}
</style></head><body>
<div class="mark"><div class="ring"></div><div class="sun"></div></div>
<h1>SKYLOOM</h1>
<div class="tip">正在升起…</div>
</body></html>"""


def _splash_url() -> str:
    """A data: URL carrying the splash page so the window has something to show
    before the local server is reachable."""
    return "data:text/html;charset=utf-8," + urllib.parse.quote(_SPLASH_HTML)


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

    local_url = f"http://127.0.0.1:{port}"

    def _close_tunnel() -> None:
        import contextlib

        t, loop = state["tunnel"], state["loop"]
        if t and loop:
            with contextlib.suppress(Exception):
                asyncio.run_coroutine_threadsafe(t.close(), loop).result(timeout=6)

    if window:
        try:
            import webview  # optional dep (pywebview)

            # Open the window NOW with the splash, so the user sees Skyloom
            # rising while the server + tunnel come up in the background.
            win = webview.create_window(
                "Skyloom",
                _splash_url(),
                width=960,
                height=780,
                min_size=(640, 480),
                text_select=False,  # keep the page-like feel — no text cursor
            )

            def _load_when_ready() -> None:
                # Runs in a pywebview worker thread after the GUI loop starts.
                import contextlib

                if not ready.wait(timeout=45.0):
                    print("  ⚠ 服务器启动超时。")
                if state["error"]:
                    print(f"  ⚠ 启动失败: {state['error']}")
                    return
                _print_banner(local_url, state["public_url"], tunnel)
                target = share_target(local_url, state["public_url"])
                # The window may be closing as we navigate — ignore that race.
                with contextlib.suppress(Exception):
                    if win is not None:
                        win.load_url(target)

            try:
                webview.start(_load_when_ready)
            finally:
                _close_tunnel()
            return
        except ImportError:
            print(
                "  pywebview 未安装，改用默认浏览器打开。"
                "（pip install 'skyloom[desktop]' 可获得原生窗口）"
            )

    # Fallback: open in the default browser, keep serving until Ctrl-C.
    if not ready.wait(timeout=45.0):
        print("  ⚠ 服务器启动超时。")
    if state["error"]:
        print(f"  ⚠ 启动失败: {state['error']}")
        return
    _print_banner(local_url, state["public_url"], tunnel)
    target = share_target(local_url, state["public_url"])

    import webbrowser

    webbrowser.open(target)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        _close_tunnel()

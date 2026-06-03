"""PyInstaller entry point for the Weather Agents desktop app.

Built into a standalone executable by ``packaging/wa-desktop.spec``. Running the
exe is equivalent to ``wa app`` — a native window + Cloudflare tunnel for phone
access. ``--selftest`` imports the heavy dependency graph and exits, so a CI/
build smoke test can confirm the bundle is complete without opening a window.
"""

from __future__ import annotations

import sys


def main() -> int:
    if "--selftest" in sys.argv:
        # Touch the modules the frozen app needs so a broken bundle fails loudly
        # here instead of when the user double-clicks the exe.
        import litellm  # noqa: F401
        import qrcode  # noqa: F401
        import webview  # noqa: F401

        from weather_agents.core import factory  # noqa: F401
        from weather_agents.web import desktop, server, tunnel  # noqa: F401

        print("selftest ok")
        return 0

    from weather_agents.web.desktop import run_desktop_app

    run_desktop_app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Packaging — installable apps for every platform

Skyloom ships as a standalone app on **Windows / macOS / Linux** desktop, plus an installable **PWA** for phones (iOS + Android). Every release is built automatically by GitHub Actions and published to [Releases](https://github.com/susurrune/skyloom/releases).

## Download (users)

Go to [Releases](https://github.com/susurrune/skyloom/releases), pick the latest, download the asset for your platform:

| Platform | Asset | Notes |
|----------|-------|-------|
| Windows 10/11 | `Skyloom-Windows-*.zip` | Unzip → double-click `Skyloom.exe`. Runs the voice client in a native window. |
| macOS (Apple Silicon / Intel) | `Skyloom-macOS-*.tar.gz` | Untar → run `Skyloom`. May need `xattr -cr` on first launch. |
| Linux | `Skyloom-Linux-*.tar.gz` | Untar → `./Skyloom`. Needs GTK / WebKit2GTK for the native window. |
| Phone (iOS / Android) | — | Run `sky app` on the desktop → scan the QR (or open the URL), then **Add to Home Screen**. Installs as a PWA with its own icon. |

### How it works

The desktop app bundles Python + all dependencies (litellm, pywebview, aiohttp, etc.) into a single portable folder. **`cloudflared` is NOT bundled** — install it separately (`winget install Cloudflare.cloudflared` or `brew install cloudflared`) if you want the public phone-accessible URL. Without cloudflared the app still works locally and over the LAN.

On the phone there is no separate app to sideload: the web client ships a full PWA (manifest + service worker + icon), so "Add to Home Screen" gives you a standalone, full-screen Skyloom on both iOS and Android — same experience as the desktop window, zero install friction.

## Build (developers)

### Desktop

```bash
pip install -e ".[desktop]"     # pywebview + qrcode
pip install pyinstaller
pyinstaller packaging/sky-desktop.spec --noconfirm
```

Output: `dist/Skyloom/Skyloom` (macOS/Linux) or `dist/Skyloom/Skyloom.exe` (Windows).
Smoke-test: `dist/Skyloom/Skyloom --selftest` → prints "selftest ok", exits 0.

### App icon

`packaging/generate_icons.py` paints the Skyloom icon (a dark tile, central amber sun, six coloured agent dots) and writes `skyloom.png` + a multi-resolution `skyloom.ico` using pure Python — no Pillow needed. The release workflow runs it before PyInstaller so the bundled `.exe` carries the icon; the committed PNG/ICO let local builds skip the step.

### Mobile (PWA)

There is no separate mobile build step. The web client (`src/weather_agents/web/client.html`) already serves a manifest, service worker, and icon, so any phone can "Add to Home Screen" for a standalone install.

### Release automation

Push a tag `v*` (e.g. `v1.1.0`) and the [release workflow](https://github.com/susurrune/skyloom/actions/workflows/release.yml) builds all three desktop assets and publishes them as a GitHub Release.

## What's bundled / not

- **Bundled:** the whole `weather_agents` package + its data (config, skills, icons, `web/client.html`), litellm + tiktoken, pywebview, aiohttp, aiosqlite, qrcode, the app icon.
- **Not bundled:** `cloudflared` (external binary). API keys and conversation history live in `~/.skyloom/` (outside the bundle), so they persist across rebuilds and updates.

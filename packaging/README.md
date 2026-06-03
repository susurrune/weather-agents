# Packaging — installable apps for every platform

Skyloom ships as a standalone app on **Windows / macOS / Linux** desktop and as an **Android APK** for phones. Every release is built automatically by GitHub Actions and published to [Releases](https://github.com/susurrune/skyloom/releases).

## Download (users)

Go to [Releases](https://github.com/susurrune/skyloom/releases), pick the latest, download the asset for your platform:

| Platform | Asset | Notes |
|----------|-------|-------|
| Windows 10/11 | `Skyloom-Windows-*.zip` | Unzip → double-click `Skyloom.exe`. Runs the voice client in a native window. |
| macOS (Apple Silicon / Intel) | `Skyloom-macOS-*.tar.gz` | Untar → run `Skyloom.app`. May need `xattr -cr` on first launch. |
| Linux | `Skyloom-Linux-*.tar.gz` | Untar → `./Skyloom`. Needs GTK / WebKit2GTK for the native window. |
| Android | `Skyloom-android.apk` | Sideload install. Opens a launcher → paste the `sky app` URL from your desktop. |

### How it works

The desktop app bundles Python + all dependencies (litellm, pywebview, aiohttp, etc.) into a single portable folder. **`cloudflared` is NOT bundled** — install it separately (`winget install Cloudflare.cloudflared` or `brew install cloudflared`) if you want the public phone-accessible URL. Without cloudflared the app still works locally and over the LAN.

The Android APK is a Capacitor wrapper around the voice client launcher. It asks for the server URL (either a Cloudflare tunnel or a LAN `https://<IP>:8765`) and opens the full Skyloom web app — same experience as the desktop window.

## Build (developers)

### Desktop

```bash
pip install -e ".[desktop]"     # pywebview + qrcode
pip install pyinstaller
pyinstaller packaging/sky-desktop.spec --noconfirm
```

Output: `dist/Skyloom/Skyloom` (macOS/Linux) or `dist/Skyloom/Skyloom.exe` (Windows).
Smoke-test: `dist/Skyloom/Skyloom --selftest` → prints "selftest ok", exits 0.

### Android APK

```bash
npm install -g @capacitor/cli @capacitor/core @capacitor/android
cd mobile/skyloom-app
npx cap sync android
cd android && ./gradlew assembleDebug
# APK at app/build/outputs/apk/debug/app-debug.apk
```

### Release automation

Push a tag `v*` (e.g. `v1.1.0`) and the [release workflow](https://github.com/susurrune/skyloom/actions/workflows/release.yml) builds all four platform assets and publishes them as a GitHub Release.

## What's bundled / not

- **Bundled:** the whole `weather_agents` package + its data (config, skills, icons, `web/client.html`), litellm + tiktoken, pywebview, aiohttp, aiosqlite, qrcode.
- **Not bundled:** `cloudflared` (external binary). API keys and conversation history live in `~/.skyloom/` (outside the bundle), so they persist across rebuilds and updates.

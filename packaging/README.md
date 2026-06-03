# Packaging — standalone desktop executable

Turn `wa app` into a double-clickable desktop app (no Python install needed).

## Build (Windows)

```bash
pip install -e ".[desktop]"     # pywebview + qrcode
pip install pyinstaller
pyinstaller packaging/wa-desktop.spec --noconfirm
```

Output: `dist/WeatherAgents/WeatherAgents.exe` (one-folder, ~150 MB — litellm
and its model providers dominate the size). Zip the `dist/WeatherAgents/`
folder to distribute.

Smoke-test the bundle without opening a window:

```bash
dist/WeatherAgents/WeatherAgents.exe --selftest   # prints "selftest ok", exits 0
```

## What's bundled / not

- **Bundled:** the whole `weather_agents` package + its data (config, skills,
  icons, `web/client.html`), litellm + tiktoken, pywebview, qrcode.
- **Not bundled:** `cloudflared`. It's an external binary for the public phone
  URL; install it separately (e.g. `winget install Cloudflare.cloudflared`).
  Without it the app still runs locally and over the LAN — just no public URL.

## Notes

- macOS / Linux: the same spec works; PyInstaller produces a native bundle for
  the OS it runs on (build on each target platform).
- The exe is GUI-mode (`console=False`). Run `--selftest` from a terminal to see
  output.
- API keys and conversation history live in `~/.weather-agents/` (outside the
  bundle), so they persist across rebuilds and updates.

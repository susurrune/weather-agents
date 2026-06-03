"""Generate Skyloom app icons (PNG + ICO) from raw pixels — no Pillow needed.

Produces:
  - packaging/skyloom.png  256x256 RGBA — source tile for the OS / dock
  - packaging/skyloom.ico  multi-resolution Windows icon (16,32,48,256 px)
Falls back gracefully: if Pillow IS available we get a proper .ico; otherwise
a single-resolution PNG-based .ico that works on modern Windows.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SIZE = 256
# Agent mark positions on a 256×256 canvas (six dots around a central sun).
# The dots trace the six weather agents positioned around the edge.
_DOTS = [
    (128, 34, 7),    # top-center — fair (sun)
    (54, 58, 5),     # top-left   — fog
    (198, 62, 5),    # top-right  — rain
    (208, 128, 5),   # right      — frost
    (196, 198, 5),   # bottom-rgt — snow
    (60, 200, 5),    # bottom-lft — dew
]


def _make_rgba_pixels(w: int, h: int) -> bytearray:
    """Paint the Skyloom icon — a dark rounded tile with a warm central sun and
    six coloured agent-marks around it, then return raw RGBA bytes."""
    # Colours (same palette as the web client's per-agent seal colours)
    BG = (42, 31, 23, 255)  # #2a1f17 — dark paper
    SUN = (212, 160, 86, 255)  # #d4a056 — amber gold
    # Agent mark colours (fog, rain, frost, snow, dew)
    DOT_COL = [
        (90, 126, 138, 255),  # fog   — muted teal
        (58, 82, 152, 255),   # rain  — blue
        (42, 112, 136, 255),  # frost — cyan
        (74, 74, 128, 255),   # snow  — periwinkle
        (58, 122, 50, 255),   # dew   — green
    ]
    RADIUS = w // 2
    pixels = bytearray(w * h * 4)
    for y in range(h):
        dy = y - RADIUS
        dy2 = dy * dy
        for x in range(w):
            dx = x - RADIUS
            d2 = dx * dx + dy2
            off = (y * w + x) * 4
            if d2 > (RADIUS - 2) ** 2:
                # Outside the circle → transparent
                pixels[off + 3] = 0
                continue
            # Inside — dark tile background
            pixels[off : off + 4] = BG

    # Central sun — warm amber glow
    _fill_circle(pixels, w, h, 128, 128, 44, SUN)
    _fill_circle(pixels, w, h, 128, 128, 36, SUN)
    # Lighter centre so the sun reads as one solid mark
    core = (240, 190, 110, 255)
    _fill_circle(pixels, w, h, 128, 128, 24, core)

    # Six agent dots + rays from the sun to the agents (fog skip — mist at edge)
    for i, (cx, cy, r) in enumerate(_DOTS):
        col = DOT_COL[i - 1] if i > 0 else SUN  # i==0 is fair (already drawn)
        if i == 0:
            continue  # fair = the sun itself
        _fill_circle(pixels, w, h, cx, cy, r, col)
        # Thin ray connecting each agent dot to the central sun
        _line(pixels, w, 128 + 18, 128 + 18, cx + r // 2, cy + r // 2, col[:3] + (90,))

    return pixels


def _fill_circle(pixels: bytearray, w: int, h: int, cx: int, cy: int, r: int, rgba: tuple) -> None:
    """Fill a circle of radius *r* at (cx, cy) with *rgba*."""
    for y in range(max(0, cy - r), min(h, cy + r + 1)):
        dy = y - cy
        dx_limit = int((r * r - dy * dy) ** 0.5)
        for x in range(max(0, cx - dx_limit), min(w, cx + dx_limit + 1)):
            off = (y * w + x) * 4
            pixels[off : off + 4] = rgba


def _line(pixels: bytearray, w: int, x0: int, y0: int, x1: int, y1: int, rgba: tuple) -> None:
    """Bresenham line with alpha blending."""
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    r, g, b, a = rgba
    while True:
        off = (y0 * w + x0) * 4
        # alpha-blend with existing pixel
        ea = pixels[off + 3]
        if ea > 0:
            blend = a / 255
            pixels[off] = int(pixels[off] * (1 - blend) + r * blend)
            pixels[off + 1] = int(pixels[off + 1] * (1 - blend) + g * blend)
            pixels[off + 2] = int(pixels[off + 2] * (1 - blend) + b * blend)
            pixels[off + 3] = min(255, ea + a // 2)
        else:
            pixels[off : off + 4] = rgba
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


# ── PNG writer (pure Python, no Pillow) ──────────────────────────────────


def _write_png(path: Path, w: int, h: int, rgba: bytes) -> None:
    """Write a minimal RGBA PNG."""

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    raw = bytearray()
    # Filter byte 0 per row
    for y in range(h):
        raw.append(0)  # no filter
        raw += rgba[y * w * 4 : (y + 1) * w * 4]

    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
    png += _chunk(b"IDAT", zlib.compress(bytes(raw)))
    png += _chunk(b"IEND", b"")
    path.write_bytes(png)


# ── ICO writer (PNG-in-ICO, works on modern Windows) ────────────────────


def _write_ico(path: Path, sizes: list[tuple[int, int, bytes]]) -> None:
    """Write a multi-resolution ICO containing PNG images (Vista+ icon format)."""
    count = len(sizes)
    # ICO header: reserved(2) + type=1(2) + count(2)
    header = struct.pack("<HHH", 0, 1, count)
    offset = 6 + 16 * count
    dir_entries = bytearray()
    img_data = bytearray()
    for w, h, png_bytes in sizes:
        dir_entries += struct.pack(
            "<BBBBHHII", w if w < 256 else 0, h if h < 256 else 0, 0, 0, 1, 32, len(png_bytes), offset
        )
        img_data += png_bytes
        offset += len(png_bytes)
    path.write_bytes(header + dir_entries + img_data)


def generate_icons() -> None:
    """Generate skyloom.png + skyloom.ico in the packaging/ directory."""
    rgba = _make_rgba_pixels(_SIZE, _SIZE)
    png_path = _HERE / "skyloom.png"
    _write_png(png_path, _SIZE, _SIZE, bytes(rgba))
    print(f"  [OK] {png_path}  ({_SIZE}×{_SIZE})")

    # Down-sample for ICO resolutions via a cheap box filter.
    ico_path = _HERE / "skyloom.ico"
    pngs: list[tuple[int, int, bytes]] = []
    for sz in (16, 32, 48, 256):
        small = _resize_rgba(rgba, _SIZE, _SIZE, sz, sz)
        buf = bytearray()
        _write_png_to_buf(bytes(small), sz, sz, buf)
        pngs.append((sz, sz, bytes(buf)))
    _write_ico(ico_path, pngs)
    print(f"  [OK] {ico_path}  (resolutions: {', '.join(f'{s}×{s}' for s, _, _ in pngs)})")


def _write_png_to_buf(rgba: bytes, w: int, h: int, buf: bytearray) -> None:
    path = _HERE / "_tmp.png"
    _write_png(path, w, h, rgba)
    data = path.read_bytes()
    path.unlink()
    buf += data


def _resize_rgba(src: bytearray, sw: int, sh: int, dw: int, dh: int) -> bytearray:
    """Simple box-filter downscale of RGBA pixel data."""
    dst = bytearray(dw * dh * 4)
    x_ratio = sw / dw
    y_ratio = sh / dh
    for dy in range(dh):
        sy_start = int(dy * y_ratio)
        sy_end = int((dy + 1) * y_ratio)
        sy_end = min(sy_end, sh)
        for dx in range(dw):
            sx_start = int(dx * x_ratio)
            sx_end = int((dx + 1) * x_ratio)
            sx_end = min(sx_end, sw)
            rs, gs, bs, as_, count = 0, 0, 0, 0, 0
            for sy in range(sy_start, sy_end):
                row_off = sy * sw * 4
                for sx in range(sx_start, sx_end):
                    off = row_off + sx * 4
                    rs += src[off]
                    gs += src[off + 1]
                    bs += src[off + 2]
                    as_ += src[off + 3]
                    count += 1
            if count:
                d_off = (dy * dw + dx) * 4
                dst[d_off] = rs // count
                dst[d_off + 1] = gs // count
                dst[d_off + 2] = bs // count
                dst[d_off + 3] = as_ // count
    return dst


if __name__ == "__main__":
    generate_icons()
    print("  done — icons ready for PyInstaller.")

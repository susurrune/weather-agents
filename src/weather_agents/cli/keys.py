"""Cross-platform raw keyboard input for the interactive REPL.

Reads single keypresses (returning named tokens like ``up`` / ``enter`` /
``shift_tab``) and offers a non-blocking Esc poll used to interrupt streaming.
Pure terminal I/O — no dependency on the rest of the CLI — so it lives on its
own as the lowest layer the REPL, pickers, and wizard all build on.

Windows uses ``msvcrt`` + ``GetAsyncKeyState``; POSIX uses ``termios`` + ``tty``
raw mode with a short ``select`` timeout to drain escape sequences.
"""

from __future__ import annotations

import sys

if sys.platform == "win32":
    import msvcrt as _msvcrt
else:
    import termios as _termios
    import tty as _tty

# Pushback buffer: when a paste arrives mixed with Enter, the extra characters
# are stashed here and returned by subsequent _get_key() calls. Private to this
# module — callers only ever go through get_key().
_key_buffer: list[str] = []


def get_key() -> str:
    """Read a single keypress. Returns named tokens for special keys."""
    if _key_buffer:
        return _key_buffer.pop(0)

    if sys.platform == "win32":
        ch = _msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            ch2 = _msvcrt.getwch()
            # Scan code Z (0x5A) = Shift+Tab on Windows console
            if ch2 == "Z":
                return "shift_tab"
            return {"H": "up", "P": "down", "K": "left", "M": "right"}.get(ch2, ch2)
        if ch == "\x03":
            raise KeyboardInterrupt
        if ch == "\x1b":
            if _msvcrt.kbhit():
                nxt = _msvcrt.getwch()
                if nxt == "[" and _msvcrt.kbhit():
                    nxt2 = _msvcrt.getwch()
                    if nxt2 == "Z":
                        return "shift_tab"
                return "esc"
            return "esc"
        if ch == "\r":
            # Drain all immediately queued chars to distinguish paste from Enter.
            # Enter at normal typing speed never has extra chars queued.
            extra = []
            while _msvcrt.kbhit():
                n = _msvcrt.getwch()
                extra.append(n)
                if len(extra) >= 16:
                    break

            if extra:
                # \r\n from Enter → all extra chars are just \n, discard them
                if all(c == "\n" for c in extra):
                    pass  # consume the \n, fall through to Shift check
                else:
                    # Has real content → paste, stash for next reads
                    _key_buffer.extend(extra)
                    return "shift_enter"

            # Shift+Enter: check physical Shift key state
            try:
                import ctypes as _ct

                if _ct.windll.user32.GetAsyncKeyState(0x10) & 0x8000:  # VK_SHIFT
                    return "shift_enter"
            except Exception:
                pass
            return "enter"
        if ch == "\n":
            # From paste, treat as newline insert
            return "shift_enter"
        if ch == "\x08":
            return "backspace"
        if ch == "\t":
            # Some Windows terminals pass Shift+Tab as \t (same as Tab).
            # Use GetAsyncKeyState to check if Shift is held.
            try:
                import ctypes as _ct

                SHIFT_MASK = 0x8000
                if _ct.windll.user32.GetAsyncKeyState(0x10) & SHIFT_MASK:  # VK_SHIFT
                    return "shift_tab"
            except Exception:
                pass
            return "tab"
        return ch
    else:
        fd = sys.stdin.fileno()
        old = _termios.tcgetattr(fd)
        try:
            import select

            _tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                # Drain the full escape sequence with a short timeout
                seq = "\x1b"
                while True:
                    r, _, _ = select.select([fd], [], [], 0.05)
                    if not r:
                        break
                    more = sys.stdin.read(1)
                    if not more:
                        break
                    seq += more
                    # CSI sequences end with a byte in 0x40–0x7E
                    if ord(more) in range(0x40, 0x7F):
                        break
                if seq == "\x1b":
                    return "esc"
                if seq.startswith("\x1b["):
                    final = seq[-1]
                    if final == "A":
                        return "up"
                    if final == "B":
                        return "down"
                    if final == "C":
                        return "right"
                    if final == "D":
                        return "left"
                    if final == "Z":
                        return "shift_tab"
                    return "esc"
                if seq.startswith("\x1bO") and seq[-1] in "PQ":
                    return "tab"
                return "esc"
            if ch == "\x03":
                raise KeyboardInterrupt
            if ch in ("\r", "\n"):
                # Consume trailing \n from \r\n sent by some IMEs
                r, _, _ = select.select([fd], [], [], 0.01)
                if r:
                    nxt = sys.stdin.read(1)
                    if nxt == "\n":
                        pass  # consumed
                return "enter"
            if ch in ("\x7f", "\x08"):
                return "backspace"
            if ch == "\t":
                return "tab"
            return ch
        finally:
            _termios.tcsetattr(fd, _termios.TCSADRAIN, old)


def poll_esc() -> bool:
    """Non-blocking check for Esc keypress (Windows).

    On Windows uses ``GetAsyncKeyState`` — a non-consumptive API that
    checks physical key state without draining the console input buffer.
    User keystrokes typed while the agent is streaming are preserved
    for the next input prompt.

    Masks both ``0x8000`` (key currently down) and ``0x0001`` (pressed
    since last call) so a quick tap of Esc between poll cycles is
    still detected.

    Only Esc triggers interrupt — Ctrl+C is left alone so users can
    copy text without stopping the agent. Python's built-in
    KeyboardInterrupt (SIGINT) still works as a force-quit fallback.
    """
    if sys.platform == "win32":
        try:
            import ctypes as _ct

            KEYEVENT = 0x8001  # 0x8000 (currently down) | 0x0001 (pressed since last call)
            if _ct.windll.user32.GetAsyncKeyState(0x1B) & KEYEVENT:  # VK_ESCAPE
                return True
        except Exception:
            pass
    return False

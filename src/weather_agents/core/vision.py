"""Image preprocessing for multimodal LLM calls.

Embeds image references as ``<<IMG:path>>`` markers in message content
strings. Before LLM calls, these markers are expanded into OpenAI-format
multimodal content blocks that LiteLLM handles natively.
"""

from __future__ import annotations

import base64
import mimetypes
import re
import tempfile
import uuid
from pathlib import Path

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".svg"}

_MARKER_RE = re.compile(r"<<IMG:(.*?)>>")

_upload_dir: Path | None = None


def _get_upload_dir() -> Path:
    global _upload_dir
    if _upload_dir is None:
        _upload_dir = Path(tempfile.gettempdir()) / "wa-images"
        _upload_dir.mkdir(parents=True, exist_ok=True)
    return _upload_dir


def make_marker(path: str) -> str:
    return f"<<IMG:{path}>>"


def has_images(text: str) -> bool:
    return "<<IMG:" in text


def save_upload(data: bytes, ext: str = ".png") -> Path:
    if not ext.startswith("."):
        ext = "." + ext
    dest = _get_upload_dir() / f"{uuid.uuid4().hex[:12]}{ext}"
    dest.write_bytes(data)
    return dest


def _read_as_data_url(path: str) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    mime, _ = mimetypes.guess_type(p.name)
    if not mime or not mime.startswith("image/"):
        mime = "image/png"
    raw = p.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _expand_content(content: str) -> list[dict]:
    """Convert a content string with ``<<IMG:path>>`` markers into
    a list of OpenAI-format content blocks."""
    parts: list[dict] = []
    last = 0
    for m in _MARKER_RE.finditer(content):
        text_before = content[last : m.start()].strip()
        if text_before:
            parts.append({"type": "text", "text": text_before})
        img_path = m.group(1)
        data_url = _read_as_data_url(img_path)
        if data_url:
            parts.append({"type": "image_url", "image_url": {"url": data_url}})
        else:
            parts.append({"type": "text", "text": f"[image not found: {img_path}]"})
        last = m.end()
    trailing = content[last:].strip()
    if trailing:
        parts.append({"type": "text", "text": trailing})
    return parts if parts else [{"type": "text", "text": content}]


def preprocess_messages_for_vision(messages: list[dict]) -> list[dict]:
    """Return a copy of *messages* with image markers expanded.

    Only user-role messages are processed — system and assistant messages
    are left untouched.  Returns the original list unmodified when no
    markers are found (fast path).
    """
    if not any(isinstance(m.get("content"), str) and has_images(m["content"]) for m in messages):
        return messages
    out: list[dict] = []
    for m in messages:
        content = m.get("content", "")
        if m.get("role") == "user" and isinstance(content, str) and has_images(content):
            expanded = _expand_content(content)
            out.append({**m, "content": expanded})
        else:
            out.append(m)
    return out


def detect_image_paths(text: str) -> list[str]:
    """Find file paths in *text* that look like images (by extension).

    Handles both Unix and Windows paths, quoted or unquoted.
    """
    candidates: list[str] = []
    # Quoted paths: "path" or 'path'
    for m in re.finditer(r"""["']([^"']+?)["']""", text):
        p = m.group(1)
        if Path(p).suffix.lower() in _IMAGE_EXTS:
            candidates.append(p)
    # Unquoted tokens — split on whitespace, check each
    for token in text.split():
        token = token.strip("\"'(),;")
        if (
            Path(token).suffix.lower() in _IMAGE_EXTS
            and token not in candidates
            and ("/" in token or "\\" in token or Path(token).is_file())
        ):
            candidates.append(token)
    return [p for p in candidates if Path(p).is_file()]


def inject_markers(text: str, paths: list[str]) -> str:
    """Replace detected image file paths in *text* with ``<<IMG:path>>`` markers."""
    for p in paths:
        marker = make_marker(str(Path(p).resolve()))
        text = text.replace(p, marker)
    return text

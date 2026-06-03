"""Structured logging for Skyloom.

Usage:
    from weather_agents.core.logger import get_logger
    log = get_logger("fog")
    log.info("chat_request", extra={"user_message": "hello", "agent": "fog"})
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_configured = False
_loggers: dict[str, logging.Logger] = {}
_request_id: str | None = None


def set_request_id(request_id: str | None) -> None:
    """Set a global request ID for the current operation."""
    global _request_id
    _request_id = request_id


def get_request_id() -> str | None:
    return _request_id


_STD_LOG_RECORD_ATTRS: set[str] = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


class StructuredFormatter(logging.Formatter):
    """JSON-lines formatter for machine-parseable logs."""

    def format(self, record: logging.LogRecord) -> str:
        obj: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if _request_id:
            obj["request_id"] = _request_id
        # Capture extra_fields set by log_event()
        if hasattr(record, "extra_fields") and record.extra_fields:
            obj.update(record.extra_fields)
        # Also capture keys from standard logging's extra={} kwarg
        for key, val in record.__dict__.items():
            if key not in _STD_LOG_RECORD_ATTRS and not key.startswith("_"):
                obj[key] = val
        obj.pop("extra_fields", None)
        return json.dumps(obj, ensure_ascii=False, default=str)


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    json_output: bool = True,
) -> None:
    """Configure root logger with structured output.

    Args:
        level: One of DEBUG, INFO, WARNING, ERROR
        log_file: Optional file path; if not set, logs go to stderr.
        json_output: If False, use plain text format for development.
    """
    global _configured
    if _configured:
        return

    root = logging.getLogger("sky")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    if json_output:
        fmt: logging.Formatter = StructuredFormatter()
    else:
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-5s %(name)s - %(message)s",
            datefmt="%H:%M:%S",
        )

    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    if log_file:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(p))
        fh.setFormatter(fmt)
        root.addHandler(fh)

    _configured = True


def _ensure_default_setup() -> None:
    """Configure default file-only logging so warnings don't pollute the CLI UI.

    Runs at most once. If WA_DEBUG=1 is set, also stream to stderr.
    """
    global _configured
    if _configured:
        return
    root = logging.getLogger("sky")
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.propagate = False  # don't leak to Python's root logger -> stderr

    # File handler — quiet by default. Best-effort: if file isn't writable,
    # fall through to NullHandler so the CLI never crashes on logging.
    log_path = Path.home() / ".skyloom" / "sky.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_path), encoding="utf-8")
        fh.setFormatter(StructuredFormatter())
        root.addHandler(fh)
    except OSError:
        root.addHandler(logging.NullHandler())

    # Opt-in stderr stream for debugging.
    import os as _os

    if _os.environ.get("WA_DEBUG") == "1":
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)-5s %(name)s — %(message)s", datefmt="%H:%M:%S"
            )
        )
        root.addHandler(sh)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Get a structured logger for the given component name."""
    _ensure_default_setup()
    if name not in _loggers:
        logger = logging.getLogger(f"sky.{name}")
        _loggers[name] = logger
    return _loggers[name]


def log_event(logger: logging.Logger, event_type: str, **fields: Any) -> None:
    """Log a structured event with extra fields.

    Usage: log_event(log, "tool_call", tool="read_file", duration_ms=42)
    """
    record = logger.makeRecord(
        logger.name,
        logging.INFO,
        "(unknown)",
        0,
        event_type,
        (),
        None,
    )
    record.extra_fields = fields  # type: ignore[attr-defined]
    logger.handle(record)

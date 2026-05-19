"""Lightweight structured output schemas for LLM response validation.

Why: LLM JSON output is inherently fragile — models emit markdown fences,
trailing commas, unquoted keys, or hallucinated fields. Rather than layering
heuristic repair (which silently passes corrupted data), we define typed
schemas and validate on ingress. Parsing failures surface immediately so the
caller can retry with a corrected prompt instead of propagating garbage.

Zero external dependencies: uses only ``dataclasses`` + ``json``.
"""

from __future__ import annotations

import json
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from typing import Any, cast, get_type_hints


class SchemaValidationError(ValueError):
    """Raised when an LLM response fails schema validation.

    Carries both a human-readable message and the raw text so callers
    can log / retry with full context.
    """

    def __init__(self, message: str, raw: str = "") -> None:
        super().__init__(message)
        self.raw = raw


# ── Schema models ────────────────────────────────────────────────────────


@dataclass
class TaskStepSchema:
    """One step in a task plan (mirrors PipelineStep / Task)."""

    id: str | int
    description: str
    agent: str = "rain"
    depends_on: list[str] = field(default_factory=list)
    priority: str = "medium"


@dataclass
class TaskPlanSchema:
    """Full task plan output from Snow's orchestrator."""

    goal: str
    steps: list[TaskStepSchema] = field(default_factory=list)


@dataclass
class FactSchema:
    """A single extracted fact for long-term memory."""

    key: str
    value: str
    category: str = "auto_extracted"


@dataclass
class ExtractionResultSchema:
    """Structured fact-extraction output from the LLM."""

    facts: list[FactSchema] = field(default_factory=list)


VALID_AGENTS: frozenset = frozenset({"fog", "rain", "frost", "snow", "dew", "fair"})


# ── Validator ────────────────────────────────────────────────────────────


def _coerce_type(value: Any, target: type) -> Any:
    """Best-effort type coercion for schema fields."""
    if target is Any or value is None:
        return value
    origin = getattr(target, "__origin__", None)
    if origin is list:
        if not isinstance(value, list):
            return []
        elem_type = getattr(target, "__args__", (Any,))[0]
        return [_coerce_type(v, elem_type) for v in value]
    if target is str:
        return str(value) if not isinstance(value, str) else value
    if target is int:
        return int(value)
    return value


def parse_schema(raw: str, schema_type: type) -> Any:
    """Parse a raw LLM response string into a typed dataclass instance.

    Handles markdown fences, leading/trailing text, and minor JSON quirks.
    Raises ``SchemaValidationError`` on failure.
    """
    if not raw or not raw.strip():
        raise SchemaValidationError("empty response", raw)

    cleaned = raw.strip()

    # Strip markdown code fences
    if "```" in cleaned:
        for fence in ("```json", "```"):
            if fence in cleaned:
                after = cleaned.split(fence, 1)[1]
                if "```" in after:
                    cleaned = after.split("```", 1)[0].strip()
                    break

    # Find first JSON object or array
    obj_start = -1
    depth = 0
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if obj_start < 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start >= 0:
                cleaned = cleaned[obj_start : i + 1]
                break
    else:
        if obj_start >= 0:
            cleaned = cleaned[obj_start:] + "}" * depth

    # Parse JSON
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        # Last resort: repair common issues
        import re as _re

        repaired = cleaned
        repaired = _re.sub(r",\s*([}\]])", r"\1", repaired)  # trailing commas
        repaired = _re.sub(r"(?<![\"'\w])(\w[\w\d_]*)(\s*:)", r'"\1"\2', repaired)  # unquoted keys
        repaired = repaired.replace("'", '"').replace("`", '"')  # quotes
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError:
            raise SchemaValidationError(f"JSON parse failed: {exc}", raw) from exc

    if not is_dataclass(schema_type):
        return data  # pass-through for non-dataclass targets

    return _dict_to_dataclass(data, schema_type, raw)


def _dict_to_dataclass(data: Any, schema_type: type, raw: str) -> Any:
    """Recursively convert a dict (or list of dicts) to a dataclass instance."""
    if not isinstance(data, dict):
        raise SchemaValidationError(
            f"expected dict for {schema_type.__name__}, got {type(data).__name__}",
            raw,
        )

    hints = get_type_hints(schema_type)
    field_map = {f.name: f for f in fields(schema_type)}
    kwargs: dict[str, Any] = {}

    for name, f in field_map.items():
        target_type = hints.get(name, str)
        if name in data:
            val = data[name]
        else:
            try:
                val = _default_for(f)
            except KeyError:
                raise SchemaValidationError(
                    f"missing required field '{name}' for {schema_type.__name__}",
                    raw,
                )

        # Handle nested dataclass lists (e.g. list[TaskStepSchema])
        origin = getattr(target_type, "__origin__", None)
        if origin is list and isinstance(val, list):
            elem_type = getattr(target_type, "__args__", (Any,))[0]
            if is_dataclass(elem_type):
                kwargs[name] = [
                    _dict_to_dataclass(item, cast(type, elem_type), raw) for item in val
                ]
                continue
            kwargs[name] = [_coerce_type(v, elem_type) for v in val]
            continue

        if is_dataclass(target_type) and isinstance(val, dict):
            kwargs[name] = _dict_to_dataclass(val, cast(type, target_type), raw)
        else:
            kwargs[name] = _coerce_type(val, target_type)

    # Validate agent names at step level
    if schema_type is TaskStepSchema and data.get("agent", "") not in VALID_AGENTS:
        kwargs["agent"] = "rain"

    return schema_type(**kwargs)


def _default_for(f: Any) -> Any:
    """Return the default value for a dataclass field."""
    if f.default is not MISSING:
        return f.default
    if f.default_factory is not MISSING:
        return f.default_factory()
    raise KeyError(f"required field '{f.name}' missing from input")


# ── Convenience parsers ──────────────────────────────────────────────────


def parse_task_plan(raw: str) -> TaskPlanSchema | None:
    """Parse Snow's orchestration output into a typed task plan."""
    try:
        result: TaskPlanSchema = parse_schema(raw, TaskPlanSchema)
        return result
    except SchemaValidationError:
        return None


def parse_facts(raw: str) -> ExtractionResultSchema | None:
    """Parse fact-extraction output into structured facts."""
    try:
        result: ExtractionResultSchema = parse_schema(raw, ExtractionResultSchema)
        return result
    except SchemaValidationError:
        return None

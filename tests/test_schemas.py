"""Tests for structured output schema validation."""
from __future__ import annotations

import pytest

from weather_agents.core.schemas import (
    TaskPlanSchema,
    TaskStepSchema,
    parse_schema,
    parse_task_plan,
    SchemaValidationError,
)


class TestParseTaskPlan:
    def test_valid_json(self):
        raw = '{"goal": "build app", "steps": [{"id": "1", "description": "design", "agent": "fog"}]}'
        plan = parse_task_plan(raw)
        assert plan is not None
        assert plan.goal == "build app"
        assert len(plan.steps) == 1
        assert plan.steps[0].agent == "fog"

    def test_markdown_fenced(self):
        raw = '```json\n{"goal": "x", "steps": [{"id": 1, "description": "a", "agent": "rain"}]}\n```'
        plan = parse_task_plan(raw)
        assert plan is not None
        assert len(plan.steps) == 1

    def test_invalid_agent_falls_back_to_rain(self):
        raw = '{"goal": "x", "steps": [{"id": "1", "description": "a", "agent": "unknown"}]}'
        plan = parse_task_plan(raw)
        assert plan is not None
        assert plan.steps[0].agent == "rain"

    def test_empty_steps(self):
        raw = '{"goal": "x", "steps": []}'
        plan = parse_task_plan(raw)
        assert plan is not None
        assert len(plan.steps) == 0

    def test_invalid_json_returns_none(self):
        raw = "this is not json at all"
        plan = parse_task_plan(raw)
        assert plan is None

    def test_depends_on_parsed(self):
        raw = '{"goal": "x", "steps": [{"id": "1", "description": "a", "depends_on": ["0"]}]}'
        plan = parse_task_plan(raw)
        assert plan is not None
        assert plan.steps[0].depends_on == ["0"]

    def test_default_fields(self):
        raw = '{"goal": "x", "steps": [{"id": "1", "description": "a"}]}'
        plan = parse_task_plan(raw)
        assert plan is not None
        assert plan.steps[0].agent == "rain"
        assert plan.steps[0].depends_on == []


class TestSchemaValidationError:
    def test_raises_on_empty(self):
        with pytest.raises(SchemaValidationError):
            parse_schema("", TaskPlanSchema)

    def test_raises_on_garbage(self):
        with pytest.raises(SchemaValidationError):
            parse_schema("<html>garbage</html>", TaskPlanSchema)


class TestExtractionSchema:
    def test_parse_facts(self):
        from weather_agents.core.schemas import parse_facts

        raw = '{"facts": [{"key": "language", "value": "Python"}]}'
        result = parse_facts(raw)
        assert result is not None
        assert len(result.facts) == 1
        assert result.facts[0].key == "language"

    def test_parse_facts_invalid(self):
        from weather_agents.core.schemas import parse_facts

        result = parse_facts("not json")
        assert result is None

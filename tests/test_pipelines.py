"""Tests for pipeline templates and factory short-circuits."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from weather_agents.core.pipelines import (
    Pipeline,
    PipelineStep,
    build_tasks_from_pipeline,
    list_pipelines,
    match_pipeline,
)


class TestMatch:
    def test_matches_code_review_zh(self):
        p = match_pipeline("帮我做一次代码审查")
        assert p is not None
        assert p.name == "code_review"

    def test_matches_code_review_en(self):
        p = match_pipeline("please code review this PR")
        assert p is not None
        assert p.name == "code_review"

    def test_matches_research_then_write(self):
        p = match_pipeline("先调研再写一篇博客")
        assert p is not None
        assert p.name == "research_then_write"

    def test_matches_implement_and_review(self):
        p = match_pipeline("实现并审查这个功能")
        assert p is not None
        assert p.name == "implement_and_review"

    def test_no_match_for_plain_question(self):
        assert match_pipeline("你好") is None
        assert match_pipeline("解释一下 RAG") is None
        assert match_pipeline("") is None

    def test_case_insensitive(self):
        assert match_pipeline("CODE REVIEW THIS") is not None


class TestBuild:
    def test_substitutes_goal(self):
        p = Pipeline(
            name="t",
            triggers=("x",),
            steps=(PipelineStep(id="1", agent="rain", description_template="do {goal}"),),
        )
        tasks = build_tasks_from_pipeline(p, "thing")
        assert tasks[0].description == "do thing"
        assert tasks[0].assigned_to == "rain"
        assert tasks[0].metadata["pipeline"] == "t"

    def test_dependency_becomes_parent_id(self):
        p = Pipeline(
            name="t",
            triggers=("x",),
            steps=(
                PipelineStep(id="1", agent="fog", description_template="a"),
                PipelineStep(
                    id="2", agent="rain", description_template="b", depends_on=("1",)
                ),
            ),
        )
        tasks = build_tasks_from_pipeline(p, "g")
        assert tasks[0].parent_id is None
        assert tasks[1].parent_id == "1"


class TestList:
    def test_list_pipelines_non_empty(self):
        items = list_pipelines()
        assert len(items) >= 3
        for item in items:
            assert "name" in item and "triggers" in item and "steps" in item


# ── Factory integration: pipeline skips snow.orchestrate ───────────────────


def _fake_agent(name: str, *, chat_result: str = "answer", success: bool = True):
    """Build a stub agent with the methods orchestrate_task touches."""
    a = MagicMock()
    a.name = name
    a.chat = AsyncMock(return_value=chat_result)
    result = MagicMock(success=success, content=chat_result)
    a.execute_task = AsyncMock(return_value=result)
    a.orchestrate = AsyncMock()  # will not be called when pipeline matches
    return a


class TestFactoryShortCircuits:
    @pytest.mark.asyncio
    async def test_pipeline_match_skips_snow_orchestrate(self):
        from weather_agents.core.factory import orchestrate_task

        agent_map = {
            "snow": _fake_agent("snow"),
            "frost": _fake_agent("frost", chat_result="frost result"),
        }
        # "代码审查" triggers code_review pipeline → only frost runs, snow.orchestrate untouched.
        tasks, results, _summary = await orchestrate_task(
            "请做代码审查", agent_map, agent_map["snow"]
        )
        agent_map["snow"].orchestrate.assert_not_awaited()
        assert len(tasks) == 1
        assert tasks[0].assigned_to == "frost"
        assert results[0].content == "frost result"

    @pytest.mark.asyncio
    async def test_single_result_skips_summary_llm(self):
        from weather_agents.core.factory import orchestrate_task

        agent_map = {
            "snow": _fake_agent("snow"),
            "frost": _fake_agent("frost", chat_result="single answer body"),
        }
        # Code review pipeline → exactly one task → summary should NOT invoke snow.chat.
        _tasks, _results, summary = await orchestrate_task(
            "code review", agent_map, agent_map["snow"]
        )
        agent_map["snow"].chat.assert_not_awaited()
        assert summary == "single answer body"

    @pytest.mark.asyncio
    async def test_multi_result_still_summarizes(self):
        from weather_agents.core.factory import orchestrate_task

        snow = _fake_agent("snow", chat_result="aggregated summary")
        agent_map = {
            "snow": snow,
            "fog": _fake_agent("fog", chat_result="research result"),
            "rain": _fake_agent("rain", chat_result="writeup result"),
        }
        # research_then_write pipeline → 2 tasks → real summary call expected.
        _t, _r, summary = await orchestrate_task("先调研再写一篇", agent_map, snow)
        snow.chat.assert_awaited()
        assert summary == "aggregated summary"

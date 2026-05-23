"""Pipeline templates — predefined multi-agent DAGs.

Why: Snow's task-decomposition LLM call costs 2-3k tokens per orchestration.
For common, recognizable workflows (code review, research-then-write, etc.),
we already know the right shape — paying an LLM to re-derive it every time
is pure waste. A pipeline match short-circuits Snow's planner entirely.

The match function is rules-only (keyword + regex), runs in microseconds, and
falls back gracefully: ``match_pipeline`` returns None when nothing fits, and
``factory.orchestrate_task`` then takes the original LLM-planning path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from weather_agents.core.agent import Task


@dataclass(frozen=True)
class PipelineStep:
    """One step in a pipeline. Mirrors ``Task`` so we can map 1:1."""

    id: str
    agent: str
    description_template: str  # `{goal}` placeholder gets substituted at build time
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class Pipeline:
    """A predefined collaboration template."""

    name: str
    triggers: tuple[str, ...]  # case-insensitive substring tokens
    steps: tuple[PipelineStep, ...]
    # Optional regex requirement. If set, at least one must match in addition
    # to triggers. Use it to keep noisy triggers (like "代码") from over-firing.
    require_regex: tuple[str, ...] = ()


_PIPELINES: tuple[Pipeline, ...] = (
    Pipeline(
        name="code_review",
        triggers=("审查代码", "代码审查", "code review", "review my code", "审计安全", "安全审计"),
        steps=(PipelineStep(id="1", agent="frost", description_template="审查代码: {goal}"),),
    ),
    Pipeline(
        name="research_then_write",
        triggers=(
            "调研后写",
            "research and write",
            "先调研再写",
            "research then write",
            "调研后生成",
            "调研并撰写",
        ),
        steps=(
            PipelineStep(id="1", agent="fog", description_template="调研: {goal}"),
            PipelineStep(
                id="2",
                agent="rain",
                description_template="基于第 1 步的调研结果撰写: {goal}",
                depends_on=("1",),
            ),
        ),
    ),
    Pipeline(
        name="research_review_write",
        triggers=(
            "调研审查后写",
            "research review write",
            "先调研审查再写",
            "调研并审查后生成",
        ),
        steps=(
            PipelineStep(id="1", agent="fog", description_template="调研: {goal}"),
            PipelineStep(
                id="2",
                agent="frost",
                description_template="审查第 1 步的调研结果,确认可行性",
                depends_on=("1",),
            ),
            PipelineStep(
                id="3",
                agent="rain",
                description_template="基于第 1 步调研和第 2 步审查意见撰写: {goal}",
                depends_on=("2",),
            ),
        ),
    ),
    Pipeline(
        name="implement_and_review",
        triggers=(
            "实现并审查",
            "写完后审查",
            "implement and review",
            "写代码并 review",
            "实现并审计",
        ),
        steps=(
            PipelineStep(id="1", agent="rain", description_template="实现: {goal}"),
            PipelineStep(
                id="2",
                agent="frost",
                description_template="审查第 1 步的实现",
                depends_on=("1",),
            ),
        ),
    ),
    Pipeline(
        name="fix_and_verify",
        triggers=(
            "修复并验证",
            "fix and verify",
            "修复bug并验证",
            "修复后审查",
            "bugfix review",
        ),
        steps=(
            PipelineStep(id="1", agent="rain", description_template="修复: {goal}"),
            PipelineStep(
                id="2",
                agent="frost",
                description_template="验证第 1 步的修复是否正确",
                depends_on=("1",),
            ),
        ),
    ),
    Pipeline(
        name="implement_test_deploy",
        triggers=("实现测试部署", "实现并部署", "写完测试再部署", "implement test deploy"),
        steps=(
            PipelineStep(id="1", agent="rain", description_template="实现: {goal}"),
            PipelineStep(
                id="2",
                agent="frost",
                description_template="审查第 1 步的实现",
                depends_on=("1",),
            ),
            PipelineStep(
                id="3",
                agent="dew",
                description_template="部署第 1 步的实现",
                depends_on=("2",),
            ),
        ),
    ),
    Pipeline(
        name="design_implement_review_deploy",
        triggers=(
            "设计实现审查部署",
            "设计开发测试上线",
            "design implement review deploy",
            "完整开发流程",
        ),
        steps=(
            PipelineStep(id="1", agent="fog", description_template="设计方案: {goal}"),
            PipelineStep(
                id="2",
                agent="rain",
                description_template="根据第 1 步的设计实现: {goal}",
                depends_on=("1",),
            ),
            PipelineStep(
                id="3",
                agent="frost",
                description_template="审查第 2 步的实现代码",
                depends_on=("2",),
            ),
            PipelineStep(
                id="4",
                agent="dew",
                description_template="部署第 2 步的实现到生产环境",
                depends_on=("3",),
            ),
        ),
    ),
    # 原 research_then_fair pipeline 已移除: fair 是独立陪伴 agent,不参与
    # 任务编排。需要"调研后总结"请直接让 fog 自己完成总结 (fog 全能,够用)。
    Pipeline(
        name="investigate_report",
        triggers=(
            "安全审计",
            "security audit",
            "漏洞扫描",
            "vulnerability scan",
            "安全检测",
        ),
        steps=(
            PipelineStep(id="1", agent="fog", description_template="安全调研: {goal}"),
            PipelineStep(
                id="2",
                agent="frost",
                description_template="基于第 1 步的调研生成安全报告",
                depends_on=("1",),
            ),
        ),
    ),
    Pipeline(
        name="debug_and_deploy",
        triggers=(
            "调试部署",
            "debug and deploy",
            "修复并上线",
            "hotfix deploy",
        ),
        steps=(
            PipelineStep(id="1", agent="rain", description_template="调试修复: {goal}"),
            PipelineStep(
                id="2",
                agent="dew",
                description_template="部署第 1 步的修复",
                depends_on=("1",),
            ),
        ),
    ),
)


def match_pipeline(goal: str) -> Pipeline | None:
    """Return the first matching pipeline, or None.

    Matching is case-insensitive substring + optional regex. Fast (~10us).
    """
    if not goal:
        return None
    lower = goal.lower()
    for p in _PIPELINES:
        if not any(tok.lower() in lower for tok in p.triggers):
            continue
        if p.require_regex and not any(re.search(rx, lower) for rx in p.require_regex):
            continue
        return p
    return None


def build_tasks_from_pipeline(pipeline: Pipeline, goal: str) -> list[Task]:
    """Materialize a pipeline into runtime Task objects (full DAG)."""
    tasks: list[Task] = []
    for step in pipeline.steps:
        description = step.description_template.format(goal=goal)
        depends = list(step.depends_on)
        parent_id = depends[0] if depends else None
        tasks.append(
            Task(
                id=step.id,
                description=description,
                assigned_to=step.agent,
                parent_id=parent_id,
                depends_on=depends,
                metadata={"goal": goal, "pipeline": pipeline.name},
            )
        )
    return tasks


def list_pipelines() -> list[dict[str, Any]]:
    """For CLI / debug introspection."""
    return [
        {
            "name": p.name,
            "triggers": list(p.triggers),
            "steps": [
                {
                    "id": s.id,
                    "agent": s.agent,
                    "depends_on": list(s.depends_on),
                }
                for s in p.steps
            ],
        }
        for p in _PIPELINES
    ]

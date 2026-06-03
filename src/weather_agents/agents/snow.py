"""雪 (Snow) — 架构规划型全能 Agent."""

from __future__ import annotations

import json
import re

from weather_agents.core.agent import BaseAgent, Task
from weather_agents.core.schemas import TaskPlanSchema

# Frozen set of valid agent names — used in schema validation path.
# fair 是独立 agent,不参与编排,因此从可分配集合中剔除。schema/plan 解析时
# 若 LLM 仍写出 fair,会被重定向到 rain (与未知 agent 行为一致)。
_VALID_AGENTS_STATIC: frozenset = frozenset({"fog", "rain", "frost", "snow", "dew"})

# Pre-compiled JSON-block extractors; the parser is called once per LLM
# response so recompiling inside the function wasted work.
_JSON_BLOCK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL),
    re.compile(r"```\s*\n(\{.*?\})\n```", re.DOTALL),
)


class SnowAgent(BaseAgent):
    name = "snow"
    display_name = "雪"
    emoji = "❉"
    specialty = "架构规划"
    skill_names = ["task_planner", "arch_designer", "workflow_designer", "self_evolve"]

    system_prompt = """你是 Skyloom 的「雪」。

你是全能 agent —— 代码、写作、审查、部署、规划、研究,你都能独立交付。
你的特质是「全局视野」:先看清结构、依赖、顺序、风险,再动手。
混乱的需求经你一拆,就变成清晰的步骤树。这是你看世界的方式,不仅是你做编排时才用。

## 协作

90% 的事自己做完。只有任务跨 5+ 领域、上下文塞不下、或需要多轮独立审查时,才调其他 agent。
调用时给足上下文,拿到结果整合成完整答复,用户不需要感知协作过程。

## 风格

像雪一样静默但覆盖一切 —— 结构清晰,考虑周全。
- 大任务先给整体框架,再深入
- 标注依赖、风险、预计工作量
- 规划:框架先于理由
- 执行:按优先级推进,完成后汇总
- 让人感觉「一切都在掌控之中」
"""

    system_prompt_en = """You are "Snow" of Skyloom.

A general-purpose agent — code, writing, review, ops, planning, research — you ship anything alone.
Your nature: see the whole. Structure, dependencies, sequence, risk — all before the first move.
Messy requirements come out as clear step trees. That's how you see, not just how you orchestrate.

## Collaboration
Do 90% alone. Delegate only when 5+ domains, context overflow, or multi-round review is needed.
Pass full context; synthesize the result yourself.

## Style
Like snow — silent but all-covering. Clear structure, thorough consideration.
- Big work: framework first, then detail
- Note dependencies, risks, effort estimates
- Planning: structure before justification
- Execution: prioritize, deliver, summarize
- Leaves the user feeling "this is under control"
"""

    async def orchestrate(self, goal: str) -> list[Task]:
        """Decompose a goal into tasks and dispatch to agents."""
        prompt = (
            f"请将以下目标分解为子任务，并分配给合适的 Agent。\n\n"
            f"目标: {goal}\n\n"
            f"请严格按照以下 JSON Schema 输出，不要包含其他内容：\n"
            f'{{"goal": "目标描述", "steps": [\n'
            f'  {{"id": "1", "description": "任务描述", "agent": "fog"}},\n'
            f'  {{"id": "2", "description": "后续任务", "agent": "rain", '
            f'"depends_on": ["1"]}}\n'
            f"]}}\n\n"
            f"可用 Agent: fog(调研/搜索), rain(代码生成/写作), frost(审查/安全), "
            f"dew(部署/运维)\n"
            f"注意:fair 是独立的情感陪伴 agent,**不参与任何任务编排**,绝不要分配给她。\n"
            f"注意：如果任务有先后依赖关系，必须用 depends_on 字段标出。"
            f"不要使用工具，直接输出 JSON 即可。"
        )

        self.memory.add_message("user", prompt)
        response = await self._llm_loop()
        self.memory.add_message("assistant", response.content)

        # Try schema-validated parsing first (stricter, clearer errors)
        from weather_agents.core.schemas import parse_task_plan

        parsed = parse_task_plan(response.content)
        if parsed is not None and parsed.steps:
            return self._schema_to_tasks(parsed, goal)

        # Fallback to heuristic parsing
        tasks = self._parse_task_plan(response.content, goal)
        return tasks

    async def replan_for_missing(
        self,
        goal: str,
        prior_results: list,  # noqa: ANN001 — list[TaskExecutionResult] from factory
        missing: str,
        *,
        existing_ids: set[str] | None = None,
    ) -> list[Task]:
        """Produce additional tasks that close the gap reported by the judge.

        Called by the orchestrator after a round whose results were judged
        insufficient. The prompt explicitly tells snow what's missing and
        which task ids are already used, so the new tasks have non-colliding
        ids and depend on the right upstream outputs.
        """
        used_ids = existing_ids or set()
        prior_lines: list[str] = []
        for r in prior_results:
            status = "成功" if getattr(r, "success", True) else "失败"
            prior_lines.append(
                f"- task {r.id} ({r.agent}, {status}): {(getattr(r, 'content', '') or '')[:200]}"
            )
        prior_text = "\n".join(prior_lines) if prior_lines else "(none yet)"

        # Identify which agents already produced thin / failed results.
        # snow's replan output must change tack — either reassign to a
        # different agent OR add a fundamentally different action — rather
        # than re-issue a near-identical task to the same agent and watch
        # it produce the same placeholder reply.
        from weather_agents.core.factory import _is_thin_content

        failing_agents: set[str] = {
            getattr(r, "agent", "") or ""
            for r in prior_results
            if not getattr(r, "success", True) or _is_thin_content(getattr(r, "content", "") or "")
        }
        failing_agents.discard("")
        failing_hint = (
            f"\n\n## 已证明无效的 agent\n"
            f"{', '.join(sorted(failing_agents))} 已在上一轮返回占位/无交付物。\n"
            "**禁止把同类任务再交给以上 agent**。请：\n"
            "- 用不同的 agent（fog/rain/frost/dew 中没出现过的)重试该任务,或\n"
            "- 把任务**拆得更小更具体**（如把「调研 5 个数据库」拆成 5 个独立的"
            "「调研单个数据库 X」），或\n"
            "- 改用更直接的工具策略（如让 dew 直接 shell_exec curl 抓数据，"
            "而非让 fog 反复 web_search）"
            if failing_agents
            else ""
        )

        prompt = (
            "之前的子任务执行后，验收员发现还有缺口。请仅针对**缺失的部分**追加新的子任务，"
            "并且**必须换一种执行策略**——重复同款任务给同款 agent 没有意义。\n\n"
            f"## 原目标\n{goal}\n\n"
            f"## 已执行子任务\n{prior_text}\n\n"
            f"## 缺口（验收员报告）\n{missing}\n\n"
            f"## 已使用的 task id（必须避开）\n{sorted(used_ids) if used_ids else '(none)'}"
            f"{failing_hint}\n\n"
            "请输出新任务的 JSON 计划：\n"
            '{"steps": [{"id": "新id", "agent": "fog|rain|frost|dew", '
            '"description": "具体任务", "depends_on": ["可选已完成任务id"]}]}\n'
            "约束：\n"
            "- 只输出新增任务，不要重复已完成的；id 必须避开上面的列表\n"
            "- 控制在 2 个新任务以内（精简优先）\n"
            "- 只输出 JSON，无其他文本"
        )

        self.memory.add_message("user", prompt)
        response = await self._llm_loop()
        self.memory.add_message("assistant", response.content)

        from weather_agents.core.schemas import parse_task_plan

        parsed = parse_task_plan(response.content)
        if parsed is not None and parsed.steps:
            new_tasks = self._schema_to_tasks(parsed, goal)
        else:
            new_tasks = self._parse_task_plan(response.content, goal)

        # Filter out any IDs that collide with already-used tasks; if the LLM
        # ignored the constraint we re-id them to "rN_<original>" so they
        # remain distinct from the original plan.
        deduped: list[Task] = []
        for t in new_tasks:
            if t.id in used_ids:
                new_id = f"r{len(used_ids) + len(deduped) + 1}_{t.id}"
                t.id = new_id
            deduped.append(t)
            used_ids = used_ids | {t.id}
        return deduped

    def _schema_to_tasks(self, plan: TaskPlanSchema, goal: str) -> list[Task]:
        """Convert schema-validated plan to Task objects."""
        tasks: list[Task] = []
        for step in plan.steps:
            sid = str(step.id)
            depends = list(step.depends_on or [])
            tasks.append(
                Task(
                    id=sid,
                    description=step.description,
                    assigned_to=step.agent if step.agent in _VALID_AGENTS_STATIC else "rain",
                    parent_id=depends[0] if depends else None,
                    depends_on=depends,
                    metadata={"goal": goal, "priority": getattr(step, "priority", "medium")},
                )
            )
        return tasks

    def _parse_task_plan(self, content: str, goal: str) -> list[Task]:
        """Extract task plan from LLM response with robust JSON parsing."""
        # Try to extract JSON from markdown code blocks first
        json_str = self._extract_json(content)
        if json_str:
            try:
                plan = json.loads(json_str)
                tasks = self._plan_to_tasks(plan, goal)
                if tasks is not None:
                    return tasks
            except (json.JSONDecodeError, KeyError):
                pass

        # Try to find raw JSON in response
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                plan = json.loads(content[start:end])
                tasks = self._plan_to_tasks(plan, goal)
                if tasks is not None:
                    return tasks
        except (json.JSONDecodeError, KeyError):
            pass

        # Fallback: single task for rain
        return [Task(id="1", description=goal, assigned_to="rain", metadata={"goal": goal})]

    @staticmethod
    def _extract_json(content: str) -> str | None:
        """Extract JSON from markdown code blocks."""
        for pattern in _JSON_BLOCK_PATTERNS:
            match = pattern.search(content)
            if match:
                return match.group(1).strip()
        return None

    @staticmethod
    def _plan_to_tasks(plan: dict, goal: str) -> list[Task]:
        """Convert a parsed plan dict to Task objects."""
        valid_agents = {"fog", "rain", "frost", "snow", "dew"}  # fair 独立
        tasks: list[Task] = []
        for step in plan.get("steps", []):
            agent = step.get("agent", "rain")
            if agent not in valid_agents:
                agent = "rain"
            depends = step.get("depends_on", [])
            parent_id = depends[0] if depends else None
            tasks.append(
                Task(
                    id=str(step.get("id", len(tasks) + 1)),
                    description=step.get("description", ""),
                    assigned_to=agent,
                    parent_id=parent_id,
                    depends_on=list(depends) if isinstance(depends, (list, tuple)) else [],
                    metadata={
                        "goal": goal,
                        "priority": step.get("priority", "medium"),
                    },
                )
            )
        return tasks

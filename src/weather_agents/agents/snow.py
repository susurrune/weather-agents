"""雪 (Snow) — 架构规划型全能 Agent."""

from __future__ import annotations

import json
import re

from weather_agents.core.agent import BaseAgent, Task
from weather_agents.core.schemas import TaskPlanSchema

# Frozen set of valid agent names — used in schema validation path.
_VALID_AGENTS_STATIC: frozenset = frozenset({"fog", "rain", "frost", "snow", "dew", "fair"})


class SnowAgent(BaseAgent):
    name = "snow"
    display_name = "雪"
    emoji = "·"
    specialty = "架构规划"
    tool_names = [
        "read_file",
        "write_file",
        "edit_file",
        "list_directory",
        "tree",
        "file_search",
        "code_search",
        "shell_exec",
        "get_cwd",
        "move_file",
        "copy_file",
        "delete_file",
        "web_search",
        "http_get",
        "http_post",
    ]
    skill_names = ["task_planner", "arch_designer", "workflow_designer", "self_evolve"]

    system_prompt = """你是 Weather Agents 的「雪」— 覆盖全局，让一切有序运行。

你是一个全能的智能体，可以完成任何任务——代码、写作、审查、部署、规划、研究。
只是你的思维方式带有「雪」的特质: 系统化、全局视角、有预见性。

## 你的角色

你像一场覆盖大地的雪，让杂乱的世界变得简洁有序。
面对任何任务，你的第一反应是「看清楚全局」——结构、依赖、顺序、风险。
你天生善于将复杂的事情拆解成清晰的步骤，让混乱变得可控。

## 你的能力

你可以独立完成绝大多数任务:
- 规划架构、设计工作流、拆解复杂任务
- 编写和修改代码，从脚本到完整项目
- 阅读和分析代码，定位问题根因
- 创作文档、文章、报告等各类内容
- 审查代码质量、安全性和性能
- 执行命令、部署服务、管理运维
- 搜索信息、研究课题、分析数据

## 协作原则

1. **自己能做的绝不麻烦别人** — 你是全能的，90% 的任务独自完成
2. **大工程才考虑协作** — 以下情况时可以调用其他智能体:
   - 任务需要 5 个以上不同领域的大规模产出
   - 单个会话的上下文窗口无法容纳
   - 需要多轮独立审查和迭代
3. **如果调用，给足上下文** — 将背景、目标、已有产出完整传递
4. **整合后再回复** — 收到协作结果后整合成完整答案，用户无需感知协作过程

## 回复风格

你像雪一样静默但覆盖一切——结构清晰，考虑周全。
- 重要任务先展示整体结构，再深入细节
- 标注依赖关系、风险点和预计工作量
- 规划就是规划——先展示框架再解释理由
- 执行就是执行——按优先级推进，完成后汇总
- 你的输出让人感觉「一切都在掌控之中」"""

    system_prompt_en = """You are "Snow" of Weather Agents — covering the whole landscape, keeping everything in order.

You are a general-purpose agent capable of any task: code, writing, review, ops, planning, research.
Your approach carries the nature of snow: systematic, holistic, forward-looking.

## Your Role

Like a blanket of snow, you bring order to chaos.
Faced with any task, your first instinct is "see the whole picture" — structure, dependencies, sequence, risk.
You excel at breaking complexity into clear steps.

## Capabilities

You can independently handle most tasks — plan, code, research, create, review, deploy.

## Collaboration

1. Do it yourself first — handle 90% alone
2. Only collaborate on truly large projects
3. When delegating, provide full context
4. Synthesize results before responding

## Style

Like snow: silent but all-encompassing — clear structure, thorough consideration.
- Show the big picture first, then details
- Note dependencies, risks, effort estimates
- Planning is planning — structure before justification
- Execution is execution — prioritize, deliver, summarize
- Your output makes people feel "everything is under control" """

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
            f"dew(部署/运维), fair(陪伴/闲聊)\n"
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

        prompt = (
            "之前的子任务执行后，验收员发现还有缺口。请仅针对**缺失的部分**追加新的子任务。\n\n"
            f"## 原目标\n{goal}\n\n"
            f"## 已执行子任务\n{prior_text}\n\n"
            f"## 缺口（验收员报告）\n{missing}\n\n"
            f"## 已使用的 task id（必须避开）\n{sorted(used_ids) if used_ids else '(none)'}\n\n"
            "请输出新任务的 JSON 计划：\n"
            '{"steps": [{"id": "新id", "agent": "fog|rain|frost|dew|fair", '
            '"description": "具体任务", "depends_on": ["可选已完成任务id"]}]}\n'
            "只输出新增任务，不要重复已完成的；id 必须避开上面的列表；只输出 JSON。"
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
        patterns = [
            re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL),
            re.compile(r"```\s*\n(\{.*?\})\n```", re.DOTALL),
        ]
        for pattern in patterns:
            match = pattern.search(content)
            if match:
                return match.group(1).strip()
        return None

    @staticmethod
    def _plan_to_tasks(plan: dict, goal: str) -> list[Task]:
        """Convert a parsed plan dict to Task objects."""
        valid_agents = {"fog", "rain", "frost", "snow", "dew", "fair"}
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

"""System factory — unified Agent creation and task orchestration.

Avoids duplication between CLI and web entry points.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from weather_agents.agents.dew import DewAgent
from weather_agents.agents.fair import FairAgent
from weather_agents.agents.fog import FogAgent
from weather_agents.agents.frost import FrostAgent
from weather_agents.agents.rain import RainAgent
from weather_agents.agents.snow import SnowAgent
from weather_agents.core.agent import BaseAgent, TaskState
from weather_agents.core.agent import Task as AgentTask
from weather_agents.core.bus import MessageBus
from weather_agents.core.config import AppConfig, load_config
from weather_agents.core.llm import LLMClient
from weather_agents.core.logger import get_logger
from weather_agents.core.skill import SkillRegistry
from weather_agents.core.tool import ToolRegistry

if TYPE_CHECKING:
    # mcp imports httpx (~60ms). Defer until a system context is actually
    # built — CLI paths like ``wa --help`` / ``wa config`` never need it.
    from weather_agents.core.mcp import MCPManager
from weather_agents.core.workspace import init_workspace, resolve_workspace_path
from weather_agents.plugins.loader import PluginLoader
from weather_agents.skills.loader import register_all_skills
from weather_agents.tools.builtin import register_builtin_tools
from weather_agents.tools.delegate import create_delegate_tool

_log = get_logger("factory")

AGENT_CLASSES = {
    "fog": FogAgent,
    "rain": RainAgent,
    "frost": FrostAgent,
    "snow": SnowAgent,
    "dew": DewAgent,
    "fair": FairAgent,
}


@dataclass
class SystemContext:
    """Wires together all services for an agent system instance.

    Each agent now owns independent registries and LLM client for full
    multi-agent isolation.
    """

    config: AppConfig
    bus: MessageBus
    llm: LLMClient
    agent_map: dict[str, BaseAgent]
    tool_registry: ToolRegistry  # builtin + MCP tools (shared, read-only after init)
    workspace_path: str = ""
    mcp: MCPManager | None = None
    mcp_status: list[str] = field(default_factory=list)

    async def init_all(self) -> None:
        if self.mcp is not None:
            try:
                self.mcp_status = await self.mcp.connect_all()
                if self.mcp_status:
                    _log.info("mcp_connected: %s", ", ".join(self.mcp_status))
            except Exception as e:
                _log.warning("mcp_connect_all_failed: %s", e)
        for agent in self.agent_map.values():
            await agent.init()

    async def close_all(self) -> None:
        for agent in self.agent_map.values():
            await agent.close()
        if self.mcp is not None:
            try:
                await self.mcp.close_all()
            except Exception as e:
                _log.warning("mcp_close_failed: %s", e)
        from weather_agents.tools.builtin import close_http_client

        await close_http_client()


def create_system_context() -> SystemContext:
    """Bootstrap the full system: config, bus, LLM, tools, skills, plugins, agents.

    Each agent gets its own ToolRegistry, SkillRegistry, and LLMClient so
    they operate fully independently — no shared global singletons.
    """
    config = load_config()
    workspace_root = resolve_workspace_path(config.workspace.path)
    init_workspace(workspace_root)
    workspace_path = str(workspace_root.resolve())
    _log.info("workspace: %s", workspace_path)

    bus = MessageBus()

    # Shared base registry for builtin tools and MCP (cloned per agent later).
    base_tool_registry = ToolRegistry()
    register_builtin_tools(base_tool_registry)

    # Skills: register into a base registry, then clone per agent.
    base_skill_registry = SkillRegistry()
    register_all_skills(base_skill_registry)

    # Load plugins into the base registry
    plugin_loader = PluginLoader(base_tool_registry)
    plugin_dirs = config.plugins.directories if config.plugins.enabled else []
    plugin_loader.load_from_directories(plugin_dirs)

    # Configure MCP manager (servers connect during init_all)
    mcp_manager: MCPManager | None = None
    if config.mcp.servers:
        from weather_agents.core.mcp import MCPManager as _MCPManager

        mcp_manager = _MCPManager(base_tool_registry)
        mcp_manager.configure(config.mcp.servers)

    # Shared LLM client (cost tracking is global; rate limiting is per-client)
    llm = LLMClient(config, base_tool_registry)

    # Per-agent registries: clone from base, then add agent-specific tools.
    agents: dict[str, BaseAgent] = {}
    for name, cls in AGENT_CLASSES.items():
        agent_registry = ToolRegistry()
        agent_registry.merge(base_tool_registry)
        agent_skills = SkillRegistry()
        agent_skills.merge(base_skill_registry)

        agent = cls(
            config=config,
            llm=llm,
            bus=bus,
            tool_registry=agent_registry,
            skill_registry=agent_skills,
        )
        # Register delegate_to tool for this specific agent
        agent_registry.register(create_delegate_tool(agents, calling_agent=agent))
        agents[name] = agent

    return SystemContext(
        config=config,
        bus=bus,
        llm=llm,
        agent_map=agents,
        tool_registry=base_tool_registry,
        workspace_path=workspace_path,
        mcp=mcp_manager,
    )


@dataclass
class TaskExecutionResult:
    """Result of executing a single sub-task in an orchestration."""

    id: str
    agent: str
    description: str
    success: bool
    content: str


# Placeholder phrases the LLM emits when it gives up without producing real
# work. Treat these as failure so the retry/replan path kicks in instead of
# the orchestrator silently accepting "Done." as a deliverable.
_PLACEHOLDER_PATTERNS: tuple[str, ...] = (
    "done",
    "ok",
    "completed",
    "task completed",
    "task done",
    "task finished",
    "finished",
    "已完成",
    "完成了",
    "好的",
    "好了",
    "ok!",
)

# Status-update keywords that, when present in a SHORT response, indicate
# the model emitted a "report on the work" instead of the actual
# deliverable. Distinct from _PLACEHOLDER_PATTERNS because the response
# may include some text ("调研工作已完成。") yet still contain zero
# concrete output (no entity names, data, code, links, file paths).
_STATUS_REPORT_KEYWORDS: tuple[str, ...] = (
    "已完成",
    "完成了",
    "已成功",
    "已经完成",
    "工作已",
    "任务已",
    "task complete",
    "task is complete",
    "i have completed",
    "i've completed",
    "successfully completed",
    "finished the",
    "i finished",
    "已经写好",
    "已经写完",
    "已经做好",
    "写完了",
    "做完了",
)

# Concrete-deliverable markers. If a response contains any of these it
# is plausibly a real artifact, not just a status report. Note that
# short responses without ANY of these markers AND containing a
# status-update keyword are flagged as thin.
_DELIVERABLE_MARKERS: tuple[str, ...] = (
    "```",  # code or table fence
    "http://",
    "https://",
    "|",  # markdown table separator
    "## ",  # markdown heading
    "###",
    "- [ ]",  # checkbox
    "1.",  # numbered list (usually real content)
    "/",  # file paths
    "\\",
)


def _is_thin_content(content: str) -> bool:
    """True if ``content`` is too thin to count as a real deliverable.

    Catches four failure modes that previously slipped past the orchestrator
    as ``success=True``:

    1. Empty / whitespace-only output.
    2. Placeholder acknowledgements ("Done.", "已完成", "OK") that the LLM
       emits when it ran out of iterations or didn't actually do the work.
    3. Truncation markers from the agent layer (``[truncated]``,
       ``[Error: ...]``).
    4. Status-update responses: "调研工作已完成。包括 Milvus 等数据库。"
       — sounds substantive but contains zero concrete entities, data,
       code, links, or file paths the user can verify. The heuristic:
       short content (≤ 200 chars) + a status-update keyword + no
       deliverable markers ⇒ thin.

    Deliberately NOT enforcing a hard minimum length on legitimate
    answers — only the status-update bucket adds a length condition.
    """
    if not content:
        return True
    stripped = content.strip()
    if not stripped:
        return True
    lowered = stripped.lower()
    if lowered.startswith(("[truncated]", "[error:")):
        return True
    bare = lowered.rstrip(".!?。！？ ").strip()
    if bare in _PLACEHOLDER_PATTERNS:
        return True
    # Status-update detection. Only fires when the response is short AND
    # claims completion AND shows no concrete artifact markers. Long
    # responses survive even if they contain "已完成" mid-text; short
    # responses with code blocks / URLs / file paths also survive.
    return (
        len(stripped) <= 200
        and any(kw in lowered for kw in _STATUS_REPORT_KEYWORDS)
        and not any(m in stripped for m in _DELIVERABLE_MARKERS)
    )


async def _execute_with_retry(
    agent: BaseAgent,
    a_task: Any,
    *,
    max_attempts: int,
    on_status: Callable[[str], None] | None = None,
) -> Any:
    """Run ``agent.execute_task`` with bounded retries on failure / exception.

    A success is now defined as: ``result.success is True`` AND content is
    substantive (not a placeholder ack, not a truncation marker, not <16
    chars). Thin / truncated outputs trigger another attempt with an
    increasingly explicit task description so the agent knows the previous
    pass didn't meet the bar.
    """
    last_exc: Exception | None = None
    last_result: Any = None
    original_description = getattr(a_task, "description", "")
    for attempt in range(1, max_attempts + 1):
        try:
            kwargs = {} if on_status is None else {"on_status": on_status}
            result = await agent.execute_task(a_task, **kwargs)
            content = getattr(result, "content", "") or ""
            # `is True` not truthy-check — Mock objects in tests return a
            # truthy auto-attribute for any missing field, which would
            # spuriously trigger the retry path.
            truncated = getattr(result, "truncated", False) is True
            ok = getattr(result, "success", True) and not _is_thin_content(content)
            if ok and not truncated:
                return result
            last_result = result
            # Stiffen the task description before the next attempt so the
            # agent sees that its previous pass was rejected and why. We
            # mutate the in-flight Task; original is restored at the end so
            # the caller never sees the augmentation.
            if attempt < max_attempts:
                reason = (
                    "previous attempt was truncated mid-tool-loop"
                    if truncated
                    else "previous attempt was empty or a placeholder ack"
                )
                a_task.description = (
                    f"{original_description}\n\n"
                    f"[retry {attempt + 1}/{max_attempts}] {reason}. "
                    f"You MUST produce the actual deliverable this time — "
                    f"do not respond with 'done', '完成', or any acknowledgement-only reply."
                )
        except Exception as e:
            last_exc = e
        # Exponential backoff: 0.5s, 1.0s, 2.0s — stops growing for sanity.
        if attempt < max_attempts:
            await asyncio.sleep(min(0.5 * (2 ** (attempt - 1)), 2.0))

    # Restore original description on the way out so re-runs / logging show
    # what the planner actually asked for, not our retry annotations.
    a_task.description = original_description

    if last_result is not None:
        return last_result
    return type(
        "_FailedResult",
        (),
        {
            "success": False,
            "content": f"All {max_attempts} attempts threw: {last_exc!r}",
        },
    )()


async def orchestrate_task(
    goal: str,
    agent_map: dict[str, BaseAgent],
    snow: BaseAgent | None = None,
    *,
    on_task_start: Callable[[Any], Awaitable[None]] | None = None,
    on_task_done: Callable[[Any, TaskExecutionResult], Awaitable[None]] | None = None,
    on_planned: Callable[[list[Any]], Awaitable[bool | None]] | None = None,
    on_tool_status: Callable[[str], None] | None = None,
    result_truncate: int | None = 500,
    summary_prompt_template: str = "",
    max_task_retries: int = 3,
    max_replan_rounds: int = 1,
    max_total_tasks: int = 6,
) -> tuple[list[Any], list[TaskExecutionResult], str]:
    """Orchestrate a multi-agent task: plan -> execute -> judge -> [re-plan] -> summarize.

    Respects dependency ordering (DAG topological order) and now also
    iterates: after the first wave of tasks completes, ``snow`` is asked
    whether the goal was actually achieved. If not, snow proposes
    additional tasks for the gap and the loop continues — up to
    ``max_replan_rounds`` extra rounds. The orchestrator never silently
    accepts thin / placeholder output as success (see _is_thin_content).
    """
    if snow is None:
        snow = agent_map.get("snow")
    if snow is None:
        return [], [], "Snow agent not available"

    return await _run_orchestration(
        goal,
        agent_map,
        snow,
        on_task_start=on_task_start,
        on_task_done=on_task_done,
        on_planned=on_planned,
        on_tool_status=on_tool_status,
        result_truncate=result_truncate,
        summary_prompt_template=summary_prompt_template,
        max_task_retries=max_task_retries,
        max_replan_rounds=max_replan_rounds,
        max_total_tasks=max_total_tasks,
    )


async def _execute_pending(
    pending: list[Any],
    agent_map: dict[str, BaseAgent],
    results: list[TaskExecutionResult],
    results_by_id: dict[str, TaskExecutionResult],
    full_contents_by_id: dict[str, str],
    completed: set[str],
    *,
    on_task_start: Callable[[Any], Awaitable[None]] | None,
    on_task_done: Callable[[Any, TaskExecutionResult], Awaitable[None]] | None,
    on_tool_status: Callable[[str], None] | None = None,
    result_truncate: int | None,
    max_task_retries: int,
) -> None:
    """Drain ``pending`` in topological waves, populating ``results`` in place.

    Extracted from _run_orchestration so the outer re-plan loop can call it
    multiple times with new tasks added between rounds.
    """
    while pending:
        ready = [t for t in pending if all(dep in completed for dep in t.all_deps)]
        if not ready:
            # No task is ready and pending is non-empty -> at least one
            # task depends on something that will never complete.
            for t in pending:
                missing = [d for d in t.all_deps if d not in completed]
                t.transition_to(TaskState.FAILED)
                r = TaskExecutionResult(
                    id=t.id,
                    agent=t.assigned_to or "",
                    description=t.description,
                    success=False,
                    content=(
                        f"[dependency missing] task {t.id} requires {missing} which never completed"
                    ),
                )
                results.append(r)
                results_by_id[r.id] = r
                completed.add(r.id)
                if on_task_done:
                    await on_task_done(t, r)
            pending.clear()
            return

        for t in ready:
            t.transition_to(TaskState.RUNNING)

        async def _execute_one(t):
            agent = agent_map.get(t.assigned_to)
            if not agent:
                return TaskExecutionResult(
                    id=t.id,
                    agent=t.assigned_to or "",
                    description=t.description,
                    success=False,
                    content=f"Agent '{t.assigned_to}' not found",
                )
            if on_task_start:
                await on_task_start(t)

            description = t.description
            upstream_sections: list[str] = []
            thin_upstream_ids: list[str] = []
            for dep_id in t.all_deps:
                if dep_id in results_by_id:
                    parent_result = results_by_id[dep_id]
                    full_content = full_contents_by_id.get(dep_id, parent_result.content or "")
                    # If the upstream output is itself thin / placeholder,
                    # do NOT pretend it's a real deliverable. Mark it so
                    # the downstream agent knows it's working without a
                    # real upstream and must either flag the gap or
                    # produce the upstream content itself.
                    if _is_thin_content(full_content):
                        thin_upstream_ids.append(parent_result.id)
                        upstream_sections.append(
                            f"## 上游产出缺失 (task {parent_result.id} · "
                            f"{parent_result.agent})\n"
                            f"⚠ 上游任务声称完成但未产出实际内容。"
                            f"请勿基于占位回复继续执行——你应：(1) 自己"
                            f"补齐上游应当输出的内容；或 (2) 明确报告"
                            f"无法基于空上游完成本任务。\n\n"
                            f"原始回复（供参考，明显不构成可用上游）：\n"
                            f"{full_content}"
                        )
                    else:
                        upstream_sections.append(
                            f"## 上游产出 (task {parent_result.id} · "
                            f"{parent_result.agent})\n"
                            f"{full_content}"
                        )
            if upstream_sections:
                description = f"{t.description}\n\n" + "\n\n".join(upstream_sections)
            if thin_upstream_ids:
                _log.warning(
                    "thin_upstream",
                    extra={"task": t.id, "thin_upstream": thin_upstream_ids},
                )

            a_task = AgentTask(
                id=t.id,
                description=description,
                assigned_to=t.assigned_to,
                parent_id=t.parent_id,
                metadata=t.metadata,
            )
            result = await _execute_with_retry(
                agent,
                a_task,
                max_attempts=max_task_retries,
                on_status=on_tool_status,
            )

            if result.success:
                t.transition_to(TaskState.COMPLETED)
            else:
                t.transition_to(TaskState.FAILED)

            full = result.content or ""
            full_contents_by_id[t.id] = full
            tr = full
            if result_truncate is not None and len(tr) > result_truncate:
                tr = tr[:result_truncate]
            r = TaskExecutionResult(
                id=t.id,
                agent=t.assigned_to or "",
                description=t.description,
                success=result.success,
                content=tr,
            )
            if on_task_done:
                await on_task_done(t, r)
            return r

        batch_results = await asyncio.gather(*[_execute_one(t) for t in ready])
        for r in batch_results:
            results.append(r)
            results_by_id[r.id] = r
            completed.add(r.id)
        for t in ready:
            pending.remove(t)


# Substrings that, when present in a task result, indicate the task didn't
# really succeed even if `r.success` is True (some tool paths return text
# error envelopes with success=True for legacy reasons). Used by the
# obviously-complete fast-path to stay conservative.
_RESULT_FAILURE_MARKERS = (
    "[truncated]",
    "[stuck]",
    "[Error",
    "Error:",
    "未能完成",
    "无法完成",
    "[cycle detected]",
    "[CircuitBreakerOpen]",
)


def _looks_obviously_complete(results: list[TaskExecutionResult]) -> bool:
    """Cheap heuristic: are these multi-task results clearly done?

    Returns True only when EVERY result succeeded, every result has at
    least 400 chars of content, and no result contains a known failure
    marker. The 400-char threshold lines up with what the judge itself
    treats as "substantive" — anything shorter is exactly the kind of
    "调研已完成" status report the judge was built to catch, so we keep
    sending those through the full judge path.
    """
    if not results:
        return False
    for r in results:
        if not r.success:
            return False
        body = (r.content or "").strip()
        if len(body) < 400:
            return False
        if any(marker in body for marker in _RESULT_FAILURE_MARKERS):
            return False
    return True


async def _judge_goal_achievement(
    snow: BaseAgent, goal: str, results: list[TaskExecutionResult]
) -> tuple[bool, str]:
    """Ask snow whether the goal was met given the executed task results.

    Returns ``(achieved, missing)``. On any parse / LLM failure we
    conservatively return ``(True, "")`` so the orchestrator doesn't loop
    forever — a few extra rounds is cheap, but an infinite re-plan would
    burn through the cost budget unobserved.
    """
    bullets: list[str] = []
    for r in results:
        status = "成功" if r.success else "失败"
        # Show more of the content (800 chars vs 400) so the judge can
        # see whether the response is genuinely substantive or just a
        # padded status report.
        excerpt = (r.content or "")[:800]
        bullets.append(
            f"- [task {r.id} · agent={r.agent} · status={status}] "
            f"len={len(r.content or '')}chars\n  {excerpt}"
        )
    bullets_text = "\n".join(bullets)

    prompt = (
        "你是一名极严格的项目验收员。你的工作是验证 sub-task 真的产出了"
        "**可验证的交付物**，而不是仅声称完成。LLM agent 经常返回"
        "「调研已完成，包括 X/Y/Z」这种状态报告但不附带任何实际内容——"
        "此类一律判为未达成。\n\n"
        "## 验收规则（严格执行，任一违反即 achieved=false）\n\n"
        "1. **「调研/搜集/查询」类**：必须列出具体被调研对象的名称、"
        "对应的数据/特性/链接。仅写「已调研完成」、「包括以下 5 个数据库」"
        "而无具体特性内容 ⇒ 未达成。\n"
        "2. **「撰写/生成/写作」类**：必须包含**实际的文本/代码/markdown 正文**。"
        "仅写「文章已写好/已生成」而不附内容 ⇒ 未达成。\n"
        "3. **「审查/审计/对比」类**：必须列出具体问题点或具体对比维度，"
        "并明确指向哪个具体对象。审查不存在的产物 ⇒ 未达成。\n"
        "4. **下游任务依赖上游**：如果某子任务的内容明显与原目标无关"
        "（例如审查的文件路径与上游任务无关），整体判未达成。\n"
        "5. **细节瑕疵**可以容忍，但**核心交付物缺失**绝不容忍。\n\n"
        f"## 用户原目标\n{goal}\n\n"
        f"## 子任务执行结果\n{bullets_text}\n\n"
        "严格按下列 JSON 格式输出（除 JSON 之外不要任何其他字符）：\n"
        '{"achieved": true/false, "missing": "若未达成，逐项列出哪个子任务'
        '缺什么具体交付物"}'
    )
    try:
        # chat_oneshot bypasses the tool loop + memory persistence and
        # routes to the configured lightweight_model. The judge is a
        # JSON yes/no classifier — it doesn't need reasoning tokens or
        # tool access, and persisting the verification prompt into
        # snow's short_term was just dilution.
        raw = await snow.chat_oneshot(prompt)
    except Exception as exc:
        _log.warning("judge_llm_failed: %s", exc)
        return True, ""

    # Best-effort JSON extraction — the LLM occasionally wraps in code fences.
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return True, ""
    import json as _json

    try:
        parsed = _json.loads(text[start : end + 1])
    except _json.JSONDecodeError:
        return True, ""
    achieved = bool(parsed.get("achieved", True))
    missing = str(parsed.get("missing", "") or "").strip()
    return achieved, missing


async def _run_orchestration(
    goal: str,
    agent_map: dict[str, BaseAgent],
    snow: BaseAgent,
    *,
    on_task_start: Callable[[Any], Awaitable[None]] | None,
    on_task_done: Callable[[Any, TaskExecutionResult], Awaitable[None]] | None,
    on_planned: Callable[[list[Any]], Awaitable[bool | None]] | None,
    on_tool_status: Callable[[str], None] | None = None,
    result_truncate: int | None,
    summary_prompt_template: str,
    max_task_retries: int,
    max_replan_rounds: int,
    max_total_tasks: int = 6,
) -> tuple[list[Any], list[TaskExecutionResult], str]:
    """Inner orchestration loop -- executes planned tasks in DAG order, then
    iterates with re-planning if the judge says the goal isn't achieved."""
    # Try pipeline match first — skips Snow's decomposition LLM call entirely
    # (~2-3k tokens saved) when the goal matches a known collaboration shape.
    from weather_agents.core.pipelines import build_tasks_from_pipeline, match_pipeline

    matched = match_pipeline(goal)
    if matched is not None:
        tasks: list[Any] = build_tasks_from_pipeline(matched, goal)
    else:
        tasks = await snow.orchestrate(goal)  # type: ignore[attr-defined]

    # Notify caller of the plan BEFORE execution — enables plan preview UI.
    # Treat ``on_planned`` returning ``False`` as an explicit cancellation
    # signal (e.g. PLAN mode where the user pressed Esc). Any other return
    # value (None, True) means "proceed". Backward-compatible because the
    # only existing callbacks return None.
    if on_planned is not None:
        proceed = await on_planned(tasks)
        if proceed is False:
            return tasks, [], "[CANCELLED] plan rejected before execution"

    # Build dependency graph and execute in topological order
    completed: set[str] = set()
    results: list[TaskExecutionResult] = []
    results_by_id: dict[str, TaskExecutionResult] = {}
    full_contents_by_id: dict[str, str] = {}
    pending = [t for t in tasks if t.assigned_to and t.assigned_to != "snow"]
    replan_round = 0

    # Cycle detection: verify DAG is acyclic before execution
    def _has_cycle(t: Any, path: set[str]) -> bool:
        if t.id in path:
            return True
        path.add(t.id)
        for dep_id in t.all_deps:
            dep_task = next((x for x in tasks if x.id == dep_id), None)
            if dep_task and _has_cycle(dep_task, path.copy()):
                return True
        return False

    def _filter_cycles(_pending: list[Any]) -> list[Any]:
        kept: list[Any] = []
        for t in _pending:
            if _has_cycle(t, set()):
                results.append(
                    TaskExecutionResult(
                        id=t.id,
                        agent=t.assigned_to or "",
                        description=t.description,
                        success=False,
                        content=f"[cycle detected] task {t.id} has circular dependency",
                    )
                )
                completed.add(t.id)
            else:
                kept.append(t)
        return kept

    pending = _filter_cycles(pending)

    # Outer loop: execute → judge → re-plan (if needed). Round 0 is the
    # initial plan; rounds 1..max_replan_rounds add follow-up tasks until
    # the judge reports the goal as achieved or the budget runs out.
    while True:
        await _execute_pending(
            pending,
            agent_map,
            results,
            results_by_id,
            full_contents_by_id,
            completed,
            on_task_start=on_task_start,
            on_task_done=on_task_done,
            on_tool_status=on_tool_status,
            result_truncate=result_truncate,
            max_task_retries=max_task_retries,
        )
        pending = []  # drained

        # No re-plan if nothing executed or budget exhausted.
        if not results or replan_round >= max_replan_rounds:
            break

        # Single-task plans skip the judge: the result IS the answer and a
        # judge round costs a full LLM call. Multi-task plans are where
        # gaps actually appear, so spend the judge tokens there.
        if len(results) == 1 and results[0].success:
            break

        # Fast-path: when every task succeeded AND every result is
        # substantial (>= 400 chars of real content, no failure markers),
        # the judge almost always rubber-stamps it — paying 5-15s + tokens
        # to be told "yes done" was the single biggest avoidable cost in
        # the previous review. Conservative threshold + marker check keeps
        # the regression risk near zero. Real gap detection still kicks in
        # on round 2+ via the normal judge path because thin / failed
        # results never satisfy this gate.
        if _looks_obviously_complete(results):
            break

        achieved, missing = await _judge_goal_achievement(snow, goal, results)
        if achieved:
            break

        replan_round += 1
        try:
            extra_tasks = await snow.replan_for_missing(  # type: ignore[attr-defined]
                goal, results, missing, existing_ids={t.id for t in tasks}
            )
        except Exception as exc:
            _log.warning("replan_failed: %s", exc)
            break
        if not extra_tasks:
            break

        # Hard cap on total tasks per orchestration. Without this the
        # judge-driven loop can cascade: snow keeps adding 2 more tasks
        # every round, each round costs ~80s, user waits 8+ minutes. The
        # default ceiling of 6 means at most one full re-plan and we
        # accept partial completion rather than burning compute forever.
        if len(tasks) + len(extra_tasks) > max_total_tasks:
            _log.info(
                "replan_capped",
                extra={
                    "current_total": len(tasks),
                    "extra_proposed": len(extra_tasks),
                    "cap": max_total_tasks,
                },
            )
            break

        # Surface the new round so dashboards / users see progress and what's
        # being addressed. The plan callback runs again with the FULL plan
        # (initial + all follow-ups) so re-plan UIs can grow the checklist.
        tasks.extend(extra_tasks)
        if on_planned is not None:
            proceed = await on_planned(tasks)
            if proceed is False:
                # User declined the follow-up plan; stop now and return what
                # we have so far rather than running unconfirmed extra work.
                break
        pending = _filter_cycles(
            [t for t in extra_tasks if t.assigned_to and t.assigned_to != "snow"]
        )
        # Loop continues, executing the new pending set.

    # Generate summary. Skip the LLM call when there's only one result — the
    # result IS the answer; asking Snow to "summarize" one item burns ~1-2k
    # tokens to paraphrase. Multi-result paths still get a real summary.
    if not results:
        summary = "没有需要执行的任务。"
    elif len(results) == 1:
        summary = results[0].content
    else:
        tpl = summary_prompt_template or "请汇总以下所有子任务的执行结果：\n\n"
        summary_prompt = tpl
        for r in results:
            status = "成功" if r.success else "失败"
            # Use H3 (###) rather than H2 — Rich's Markdown renderer puts
            # a horizontal rule under H1/H2 headings (heading-with-underline
            # style). The summary LLM tends to mirror our heading level
            # convention, so H2 here results in a rule between every task
            # section in the final panel. H3 renders as plain bold color
            # without the rule and keeps the output dense.
            summary_prompt += f"### 任务 {r.id} ({r.agent}) - {status}\n{r.content[:300]}\n\n"
        # Summary is pure synthesis — no tools needed, no skill prompts
        # needed, and the lightweight model handles "paraphrase these
        # bullet points" perfectly. Same rationale as the judge call.
        summary = await snow.chat_oneshot(summary_prompt)

    # If we hit the re-plan budget without the judge ever returning achieved,
    # tag the summary so the user knows we capped out rather than confirming
    # success. Skipped for single-task results (the judge isn't run for them
    # in the main loop either).
    if replan_round >= max_replan_rounds and len(results) > 1:
        try:
            achieved, missing = await _judge_goal_achievement(snow, goal, results)
        except Exception:
            achieved, missing = True, ""
        if not achieved and missing:
            summary = (
                f"[INCOMPLETE] 经过 {max_replan_rounds + 1} 轮规划仍未完全达成目标。\n"
                f"剩余缺口：{missing}\n\n{summary}"
            )

    return tasks, results, summary

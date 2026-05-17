"""System factory — unified Agent creation and task orchestration.

Avoids duplication between CLI and web entry points.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from weather_agents.agents.dew import DewAgent
from weather_agents.agents.fog import FogAgent
from weather_agents.agents.frost import FrostAgent
from weather_agents.agents.rain import RainAgent
from weather_agents.agents.snow import SnowAgent
from weather_agents.agents.sunshine import SunshineAgent
from weather_agents.core.agent import BaseAgent
from weather_agents.core.agent import Task as AgentTask
from weather_agents.core.bus import MessageBus
from weather_agents.core.config import AppConfig, load_config
from weather_agents.core.llm import LLMClient
from weather_agents.core.logger import get_logger
from weather_agents.core.mcp import MCPManager
from weather_agents.core.skill import global_skill_registry
from weather_agents.core.tool import global_registry
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
    "sunshine": SunshineAgent,
}

AGENT_EMOJI = {
    "fog": "~",
    "rain": "/",
    "frost": "+",
    "snow": "·",
    "dew": ",",
    "sunshine": "*",
}

AGENT_COLORS: dict[str, str] = {
    "fog": "bright_white",
    "rain": "blue",
    "frost": "cyan",
    "snow": "bright_white",
    "dew": "green",
    "sunshine": "#FFD700",
}


@dataclass
class SystemContext:
    """Wires together all shared services for an agent system instance."""

    config: AppConfig
    bus: MessageBus
    llm: LLMClient
    agent_map: dict[str, BaseAgent]
    workspace_path: str = ""
    mcp: MCPManager | None = None
    mcp_status: list[str] = field(default_factory=list)

    async def init_all(self) -> None:
        # Connect MCP servers first so their tools are registered before agents
        # snapshot the tool registry.
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
    """Bootstrap the full system: config, bus, LLM, tools, skills, plugins, agents."""
    config = load_config()
    workspace_root = resolve_workspace_path(config.workspace.path)
    init_workspace(workspace_root)
    workspace_path = str(workspace_root.resolve())
    _log.info("workspace: %s", workspace_path)

    bus = MessageBus()
    register_builtin_tools()
    register_all_skills()

    # Load plugins
    plugin_loader = PluginLoader(global_registry)
    plugin_dirs = config.plugins.directories if config.plugins.enabled else []
    plugin_loader.load_from_directories(plugin_dirs)

    # Configure MCP manager (servers connect during init_all so async I/O
    # happens inside the event loop, not at construction time).
    mcp_manager: MCPManager | None = None
    if config.mcp.servers:
        mcp_manager = MCPManager(global_registry)
        mcp_manager.configure(config.mcp.servers)

    llm = LLMClient(config, global_registry)
    agents = {
        name: cls(
            config=config,
            llm=llm,
            bus=bus,
            tool_registry=global_registry,
            skill_registry=global_skill_registry,
        )
        for name, cls in AGENT_CLASSES.items()
    }

    global_registry.register(create_delegate_tool(agents))

    return SystemContext(
        config=config,
        bus=bus,
        llm=llm,
        agent_map=agents,
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


async def _execute_with_retry(agent: BaseAgent, a_task: Any, *, max_attempts: int) -> Any:
    """Run ``agent.execute_task`` with bounded retries on failure / exception.

    Returns whatever ``execute_task`` returned on success. On final failure,
    returns the last result object (which already has ``.success=False``) so
    the caller's result shape stays uniform.
    """
    last_exc: Exception | None = None
    last_result: Any = None
    for attempt in range(1, max_attempts + 1):
        try:
            result = await agent.execute_task(a_task)
            if getattr(result, "success", True):
                return result
            last_result = result
        except Exception as e:
            last_exc = e
        # Exponential backoff: 0.5s, 1.0s, 2.0s — stops growing for sanity.
        if attempt < max_attempts:
            await asyncio.sleep(min(0.5 * (2 ** (attempt - 1)), 2.0))

    if last_result is not None:
        return last_result
    # Synthesize a result object if every attempt threw.
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
    result_truncate: int | None = 500,
    summary_prompt_template: str = "",
    max_task_retries: int = 3,
) -> tuple[list[Any], list[TaskExecutionResult], str]:
    """Orchestrate a multi-agent task: plan -> execute -> summarize.

    Respects dependency ordering: tasks with depends_on wait for their
    parent to complete before starting.
    """
    if snow is None:
        snow = agent_map.get("snow")
    if snow is None:
        return [], [], "Snow agent not available"

    # Try pipeline match first — skips Snow's decomposition LLM call entirely
    # (~2-3k tokens saved) when the goal matches a known collaboration shape.
    from weather_agents.core.pipelines import build_tasks_from_pipeline, match_pipeline

    matched = match_pipeline(goal)
    if matched is not None:
        tasks: list[Any] = build_tasks_from_pipeline(matched, goal)
    else:
        tasks = await snow.orchestrate(goal)  # type: ignore[attr-defined]

    # Build dependency graph and execute in topological order
    completed: set[str] = set()
    results: list[TaskExecutionResult] = []
    results_by_id: dict[str, TaskExecutionResult] = {}
    # Original, un-truncated content keyed by task id — used so downstream
    # description-injection can know when the excerpt is incomplete and
    # so shared_working stores the full value (not the 500-char excerpt).
    full_contents_by_id: dict[str, str] = {}
    pending = [t for t in tasks if t.assigned_to and t.assigned_to != "snow"]

    while pending:
        # Find tasks whose dependencies are satisfied
        ready = [t for t in pending if not t.parent_id or t.parent_id in completed]
        if not ready:
            ready = pending[:1]  # break deadlock

        # Execute ready tasks concurrently
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
            # Inject upstream results so downstream agents actually see what
            # their dependencies produced. Two complementary channels:
            #   (1) Inline excerpt in the description so single-pass agents
            #       without tool support still work. Capped at 500 chars to
            #       keep the prompt small.
            #   (2) The FULL value written to shared_working keyed by
            #       `task_<parent_id>_output`. Downstream agents can pull it
            #       in full via the read_shared_memory tool when the truncated
            #       excerpt is insufficient — the "按需检索" pattern.
            description = t.description
            if t.parent_id and t.parent_id in results_by_id:
                parent_result = results_by_id[t.parent_id]
                # Prefer the un-truncated original when available — otherwise
                # the description excerpt would be capped at the per-result
                # truncation limit (default 500), making the "incomplete"
                # detection useless.
                full_content = full_contents_by_id.get(t.parent_id, parent_result.content or "")
                excerpt = full_content
                truncated = False
                if len(excerpt) > 500:
                    excerpt = excerpt[:500]
                    truncated = True
                tail = (
                    "\n\n[完整内容存于 shared memory · key=task_{pid}_output · "
                    "调用 read_shared_memory 拉取]"
                    if truncated
                    else ""
                ).format(pid=parent_result.id)
                description = (
                    f"{t.description}\n\n"
                    f"## 上游产出 (task {parent_result.id} · {parent_result.agent})\n"
                    f"{excerpt}{tail}"
                )
            a_task = AgentTask(
                id=t.id,
                description=description,
                assigned_to=t.assigned_to,
                parent_id=t.parent_id,
                metadata=t.metadata,
            )
            # Retry on failure — README has long advertised "失败任务自动重试
            # (最多 3 轮)" but the code only awaited once. Exponential backoff
            # is bounded to keep failure cases from dragging on too long.
            result = await _execute_with_retry(agent, a_task, max_attempts=max_task_retries)
            full = result.content or ""
            # Publish FULL output to shared memory BEFORE truncation so
            # downstream agents can fetch the original via read_shared_memory.
            # Best-effort — DB / session errors must never break orchestration.
            if full:
                with contextlib.suppress(Exception):
                    await agent.memory.write_shared(f"task_{t.id}_output", full)
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
            summary_prompt += f"## 任务 {r.id} ({r.agent}) - {status}\n{r.content[:300]}\n\n"
        summary = await snow.chat(summary_prompt)

    return tasks, results, summary

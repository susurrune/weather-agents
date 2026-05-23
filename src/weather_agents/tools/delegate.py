"""delegate_to — allows an agent to hand off work to a specialist agent."""

from __future__ import annotations

import contextlib
import contextvars
from typing import TYPE_CHECKING

from weather_agents.core.bus import Event, EventType
from weather_agents.core.icons import icon_text
from weather_agents.core.logger import get_logger
from weather_agents.core.tool import Tool, ToolParameter

# Depth of the current delegation *chain* (snow → fog → frost = 2). A
# ContextVar — not a closure-local int — so concurrent delegations from the
# same caller (snow doing asyncio.gather(delegate_to(fog), delegate_to(rain))
# don't share a single counter and falsely trip the depth limit on their
# siblings. ContextVars are inherited by spawned tasks, so true nesting
# (snow → fog → frost) still counts correctly down the chain.
_delegation_depth_var: contextvars.ContextVar[int] = contextvars.ContextVar(
    "_delegation_depth", default=0
)

if TYPE_CHECKING:
    from weather_agents.core.agent import BaseAgent


def _build_shared_context(calling_agent: BaseAgent | None, context: str) -> str:
    """Assemble shared context for the delegated agent.

    Includes the most recent conversation exchange from the calling agent
    so the delegate arrives with enough information to work independently,
    plus a list of the parent's currently-active skills so the delegate
    knows which capabilities the parent already brought to the problem
    (and doesn't waste a list_skills round-trip discovering them).
    """
    parts: list[str] = []
    if context:
        parts.append(f"Additional context: {context}")

    if calling_agent:
        # Pass the last 2 messages from the calling agent as shared context
        recent = calling_agent.memory.short_term
        non_system = [m for m in recent if m.role != "system"]
        if non_system:
            ctx_msgs = non_system[-4:]  # last 2 exchanges
            msg_text = "\n".join(f"[{m.role}] {(m.content or '')[:500]}" for m in ctx_msgs)
            if msg_text:
                parts.append(f"Calling agent context:\n{msg_text}")

        raw_active = getattr(calling_agent, "_active_skills", None)
        # Be defensive: in tests calling_agent may be a bare Mock whose
        # auto-attribute returns another Mock (not iterable). Only treat
        # the value as the active skill set when it's actually a
        # set/list/tuple of strings.
        if isinstance(raw_active, (set, frozenset, list, tuple)):
            active = sorted(str(s) for s in raw_active if isinstance(s, str))
        else:
            active = []
        if active:
            # Just inform — actual activation on the target happens via
            # trigger matching against the task description, which is
            # more precise than blindly copying the parent's skill set
            # (frost reviewing a deck doesn't need the pptx skill that
            # fog used to generate it).
            parts.append(
                "Parent agent had these skills active when delegating: "
                + ", ".join(active)
                + ". If you need the same context, call use_skill(name); "
                "otherwise proceed with your own specialty."
            )

    return "\n\n".join(parts)


_log = get_logger("delegate")

# fair 故意从此表移除:她是独立的情感陪伴 agent,不参与 agent-to-agent 编排
# (既不被 delegate 调用,也不被 snow.orchestrate 分配任务)。
AGENT_SPECIALTIES: dict[str, str] = {
    "fog": "research / code analysis / knowledge retrieval / information synthesis",
    "rain": "code generation / content creation / data transformation / multi-file projects",
    "frost": "code review / security audit / performance analysis / debugging",
    "snow": "task planning / architecture design / workflow management / codebase refactoring",
    "dew": "command execution / deployment / API integration / system operations",
}

# Cap delegate result size injected into the caller's context. Lowered from
# 8000 so that long delegated outputs don't dominate the caller's short-term
# memory and cause the caller's next reply to mimic the delegate's voice.
_MAX_RESULT_CHARS = 4000


def create_delegate_tool(
    agent_map: dict[str, BaseAgent], *, calling_agent: BaseAgent | None = None
) -> Tool:
    """Build a ``delegate_to`` tool whose handler closes over *agent_map*.

    Call this **after** all agents have been constructed so the handler
    can look up target agents at execution time.

    *calling_agent* is the agent that owns this tool instance (set when
    per-agent registries are used). When None, shared context building
    is skipped (safe fallback for backward compatibility).
    """
    from weather_agents.core.agent import AgentState, Task

    _MAX_DEPTH = 2  # allow 1 level of nesting (0→1→2, blocked at 3)

    async def _handle(agent: str, task: str, context: str = "") -> str:
        # fair 是独立 agent: 既不被调用,也不调用别人。守护这条边界,使得 fair
        # 始终保留她的对话上下文,不被 orchestration prompt 污染。
        if calling_agent and calling_agent.name == "fair":
            return (
                "Fair (晴) is an independent companion agent and does not "
                "delegate work. Continue the conversation directly with the user."
            )
        if agent == "fair":
            return (
                "Fair (晴) cannot be delegated to — she is the user's personal "
                "companion, not a work agent. Complete this task yourself, or "
                "delegate to fog / rain / frost / snow / dew."
            )

        if agent not in agent_map:
            names = ", ".join(sorted(k for k in agent_map.keys() if k != "fair"))
            return f"Unknown agent '{agent}'. Available agents: {names}"

        if calling_agent and agent == calling_agent.name:
            return (
                f"You are already {calling_agent.display_name}. "
                f"Complete the task directly using your own tools and knowledge "
                f"— do not delegate to yourself."
            )

        target = agent_map[agent]

        current_depth = _delegation_depth_var.get()
        if current_depth >= _MAX_DEPTH:
            return (
                f"Nested delegation depth limit ({_MAX_DEPTH}) reached. "
                f"Agent '{agent}' must complete the task directly using its own tools."
            )

        _token = _delegation_depth_var.set(current_depth + 1)
        try:
            await target.init()

            # Run trigger-based skill activation on the target against
            # the task description before execute_task. Without this the
            # target would still need to discover skills via list_skills /
            # use_skill, paying the same round-trip cost the parent's
            # chat_stream auto-activation was designed to avoid.
            with contextlib.suppress(Exception):
                if hasattr(target, "_auto_activate_skills"):
                    target._auto_activate_skills(task)

            shared_ctx = _build_shared_context(calling_agent, context)

            task_obj = Task(
                id=f"dlg-{id(task) & 0xFFFF:04x}",
                description=task,
                assigned_to=agent,
                metadata={"context": shared_ctx} if shared_ctx else {},
            )

            _log.info(
                "delegation_start",
                extra={
                    "target": agent,
                    "task": task[:120],
                    "depth": current_depth + 1,
                },
            )

            target.bus.add_event(
                Event(
                    type=EventType.TOOL_CALL,
                    source=agent,
                    data={"tool": "delegate_to", "phase": "start", "task": task[:200]},
                )
            )

            result = await target.execute_task(task_obj)

            if target.state == AgentState.ERROR:
                await target._set_state(AgentState.IDLE)

            target.bus.add_event(
                Event(
                    type=EventType.TOOL_CALL,
                    source=agent,
                    data={
                        "tool": "delegate_to",
                        "phase": "done",
                        "success": result.success,
                    },
                )
            )

            content = result.content
            if len(content) > _MAX_RESULT_CHARS:
                content = content[:_MAX_RESULT_CHARS] + "\n\n… (truncated)"

            status = "completed" if result.success else "failed"
            header = f"[{icon_text(target.name)} {target.display_name}] {status}"

            _log.info(
                "delegation_done",
                extra={
                    "target": agent,
                    "success": result.success,
                    "chars": len(result.content),
                },
            )

            # Frame the result so the calling LLM treats it as third-party
            # data, not its own voice. Without explicit delimiters the
            # caller tends to mimic the delegate's tone/phrasing on its
            # next turn (observed: rain echoing fog/fair after a
            # delegation). The "do not redo" half is the round-4 add —
            # the PPT case study showed fog re-running its own review
            # immediately after frost delegated-review came back, paying
            # ~5 minutes of compute for the same conclusion.
            trust_clause = (
                f"[Hint: {target.display_name}'s work above is COMPLETE and "
                "authoritative for the sub-task you delegated. Do NOT "
                "re-verify, re-audit, re-implement, or repeat the same "
                "operation in your own voice — that doubles the cost for "
                "the same answer. Synthesize a brief reply in YOUR OWN "
                "voice citing their conclusion, and only do MORE work if "
                "it's distinctly different from what they completed.]"
                if result.success
                else (
                    f"[Hint: {target.display_name}'s attempt failed. "
                    "Decide whether to retry with a different approach, "
                    "ask the user for guidance, or skip this sub-task. "
                    "Don't just call delegate_to again with the same task.]"
                )
            )
            return (
                f"<delegated_response from='{target.display_name}'>\n"
                f"{header}\n\n{content}\n"
                f"</delegated_response>\n"
                f"{trust_clause}"
            )

        except Exception as exc:
            _log.exception("delegation_error: %s", exc)
            return f"Delegation to '{agent}' failed: {exc}"
        finally:
            _delegation_depth_var.reset(_token)

    return Tool(
        name="delegate_to",
        description=(
            "Delegate a task to a specialist agent and receive the result. "
            "Use this when a task would benefit from another agent's expertise. "
            "Available agents and their specialties:\n"
            + "\n".join(f"  - {k}: {v}" for k, v in AGENT_SPECIALTIES.items())
        ),
        parameters=[
            ToolParameter(
                name="agent",
                type="string",
                description=("Target agent name. One of: fog, rain, frost, snow, dew."),
                required=True,
            ),
            ToolParameter(
                name="task",
                type="string",
                description="Clear, specific description of what the agent should do.",
                required=True,
            ),
            ToolParameter(
                name="context",
                type="string",
                description="Additional context or data the target agent needs.",
                required=False,
                default="",
            ),
        ],
        handler=_handle,
        # Each delegation triggers a full LLM run on the target agent —
        # never serve a stale result for the same (agent, task) pair.
        cacheable=False,
    )

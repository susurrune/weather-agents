"""delegate_to — allows an agent to hand off work to a specialist agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

from weather_agents.core.bus import Event, EventType
from weather_agents.core.icons import icon_text
from weather_agents.core.logger import get_logger
from weather_agents.core.tool import Tool, ToolParameter

if TYPE_CHECKING:
    from weather_agents.core.agent import BaseAgent


def _build_shared_context(calling_agent: BaseAgent | None, context: str) -> str:
    """Assemble shared context for the delegated agent.

    Includes the most recent conversation exchange from the calling agent
    so the delegate arrives with enough information to work independently.
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

    return "\n\n".join(parts)


_log = get_logger("delegate")

AGENT_SPECIALTIES: dict[str, str] = {
    "fog": "research / code analysis / knowledge retrieval / information synthesis",
    "rain": "code generation / content creation / data transformation / multi-file projects",
    "frost": "code review / security audit / performance analysis / debugging",
    "snow": "task planning / architecture design / workflow management / codebase refactoring",
    "dew": "command execution / deployment / API integration / system operations",
    "sunshine": "emotional support / thoughtful conversation / bilingual companionship / creative insight",
}

# Cap delegate result size injected into the caller's context. Lowered from
# 8000 so that long delegated outputs don't dominate the caller's short-term
# memory and cause the caller's next reply to mimic the delegate's voice.
_MAX_RESULT_CHARS = 4000


def create_delegate_tool(agent_map: dict[str, BaseAgent]) -> Tool:
    """Build a ``delegate_to`` tool whose handler closes over *agent_map*.

    Call this **after** all agents have been constructed so the handler
    can look up target agents at execution time.
    """
    from weather_agents.core.agent import AgentState, Task

    _delegation_depth = 0
    _MAX_DEPTH = 2  # allow 1 level of nesting (0→1→2, blocked at 3)

    async def _handle(agent: str, task: str, context: str = "") -> str:
        nonlocal _delegation_depth

        if agent not in agent_map:
            names = ", ".join(sorted(agent_map.keys()))
            return f"Unknown agent '{agent}'. Available agents: {names}"

        target = agent_map[agent]

        if _delegation_depth >= _MAX_DEPTH:
            return (
                f"Nested delegation depth limit ({_MAX_DEPTH}) reached. "
                f"Agent '{agent}' must complete the task directly using its own tools."
            )

        _delegation_depth += 1
        try:
            await target.init()

            # Build shared context from the calling agent. Read via ContextVar
            # (not a module global) so concurrent delegations from different
            # callers don't clobber each other.
            from weather_agents.core.agent import get_call_agent

            shared_ctx = _build_shared_context(get_call_agent(), context)

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
                    "depth": _delegation_depth,
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
            # next turn (observed: rain echoing fog/sunshine after a
            # delegation).
            return (
                f"<delegated_response from='{target.display_name}'>\n"
                f"{header}\n\n{content}\n"
                f"</delegated_response>\n"
                f"[Hint: above is {target.display_name}'s reply. "
                f"Synthesize a brief reply in YOUR OWN voice; do not quote or "
                f"continue their text verbatim.]"
            )

        except Exception as exc:
            _log.exception("delegation_error: %s", exc)
            return f"Delegation to '{agent}' failed: {exc}"
        finally:
            _delegation_depth -= 1

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
                description=("Target agent name. One of: fog, rain, frost, snow, dew, sunshine."),
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
    )

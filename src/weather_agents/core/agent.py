"""Base agent class for all Weather Agents."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import cast

from weather_agents.core.bus import Event, EventType, MessageBus
from weather_agents.core.config import AppConfig
from weather_agents.core.llm import LLMClient, LLMResponse
from weather_agents.core.logger import get_logger
from weather_agents.core.memory import Memory
from weather_agents.core.skill import Skill, SkillRegistry
from weather_agents.core.tool import Tool, ToolRegistry

_log = get_logger("agent")

# Set by chat_stream() each call — gives tool handlers (use_skill, list_skills)
# a way to reach the agent that is currently processing.
_call_agent: BaseAgent | None = None


class AgentState(StrEnum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    WAITING = "waiting"
    ERROR = "error"


@dataclass
class Task:
    id: str
    description: str
    assigned_to: str | None = None
    parent_id: str | None = None
    status: str = "pending"
    result: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class TaskResult:
    success: bool
    content: str
    data: dict = field(default_factory=dict)


class BaseAgent:
    """Base class for all Weather Agents."""

    name: str = ""
    display_name: str = ""
    emoji: str = ""
    specialty: str = ""
    system_prompt: str = ""
    tool_names: list[str] = []
    skill_names: list[str] = []

    # Time-tag cache (shared across all instances, 30s TTL)
    _time_tag: str | None = None
    _time_tag_ts: float = 0.0

    def __init__(
        self,
        config: AppConfig,
        llm: LLMClient,
        bus: MessageBus,
        tool_registry: ToolRegistry,
        skill_registry: SkillRegistry | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.bus = bus
        self.tool_registry = tool_registry
        self.skill_registry = skill_registry or SkillRegistry()
        self.state = AgentState.IDLE
        self.memory = Memory(config.memory, self.name)
        self._tools: list[Tool] = []
        self._skills: list[Skill] = []
        self._active_skills: set[str] = set()
        self._skill_tools: dict[str, list[str]] = {}  # skill_name -> tool_names
        self._skill_config_overrides: dict[str, dict] = {}  # active overrides merged from skills
        self._base_system_prompt: str = ""
        agent_cfg = getattr(config.agents, self.name, None)
        self._max_tool_rounds: int = agent_cfg.max_tool_rounds if agent_cfg else 10

    def _resolve_system_prompt(self) -> str:
        """Pick the language-appropriate system prompt based on config."""
        lang = getattr(self.config.llm, "language", "zh")
        if lang == "en" and hasattr(self.__class__, "system_prompt_en"):
            return self.__class__.system_prompt_en  # type: ignore[no-any-return]
        return self.system_prompt

    def _inject_workspace_info(self, prompt: str) -> str:
        """Append workspace path to the system prompt."""
        from weather_agents.core.workspace import init_workspace, resolve_workspace_path

        ws_root = resolve_workspace_path(self.config.workspace.path)
        init_workspace(ws_root)  # idempotent — ensures tree exists
        ws_str = str(ws_root.resolve())

        lang = getattr(self.config.llm, "language", "zh")
        if lang == "en":
            ws_block = (
                f"\n\n## Workspace\n"
                f"Your workspace directory is `{ws_str}`.\n"
                f"- Use `{ws_str}/files/` for generated files.\n"
                f"- Use `{ws_str}/output/` for task results and exports.\n"
                f"- Use `{ws_str}/temp/` for temporary/scratch files.\n"
                f"- Always prefer paths under the workspace for file operations."
            )
        else:
            ws_block = (
                f"\n\n## 工作空间\n"
                f"你的工作空间目录是 `{ws_str}`。\n"
                f"- 生成的文件放在 `{ws_str}/files/`\n"
                f"- 任务结果和导出放在 `{ws_str}/output/`\n"
                f"- 临时文件放在 `{ws_str}/temp/`\n"
                f"- 所有文件操作优先使用工作空间内的路径。"
            )
        return prompt + ws_block

    def _current_time_tag(self) -> str:
        """One-line current-date tag for the system prompt.

        30-second class-level cache saves the clock call on rapid rebuilds
        (skill activate/deactivate, language switch).
        """
        now = time.time()
        if BaseAgent._time_tag is not None and now - BaseAgent._time_tag_ts < 30:
            return BaseAgent._time_tag
        import datetime

        tag = (
            f"Today is {datetime.date.today().isoformat()}. "
            f"Current time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}."
        )
        BaseAgent._time_tag = tag
        BaseAgent._time_tag_ts = now
        return tag

    def _inject_behavior_rules(self, prompt: str) -> str:
        """Append concise behavior rules to the system prompt."""
        lang = getattr(self.config.llm, "language", "zh")
        if lang == "en":
            rules = (
                "\n\n## Behavior\n"
                '1. Execute tools immediately — no "I will..." or "Let me..." narration before tool calls. Just call them.\n'
                "2. After completing work: give a 1-2 sentence concise report, no log-style verbosity.\n"
                "3. Do not use decorative separator lines (---, ***, ___, etc) — the interface handles visual separation.\n"
                "4. No emoji in generated web pages — use SVG icons, CSS shapes, or Unicode symbols instead.\n"
                "5. Proportional planning: for simple tasks (stop server, show status, answer a question) — just do it directly. "
                "Only break work into numbered steps when the task genuinely needs 3+ distinct actions. "
                "Don't survey the workspace or read files unless the task requires it.\n"
                "6. Self-verify: after tool execution, check results. If a tool returned an error, "
                "diagnose the cause and try a corrected approach immediately. "
                "If a file write/edit succeeded, read it back to confirm the content is correct. "
                "Report the final verified state, not just the attempt."
            )
        else:
            rules = (
                "\n\n## 行为守则\n"
                "1. 执行工具前不赘述 — 不说「我将要...」「让我先...」，直接调用\n"
                "2. 完成后用 1-2 句简洁汇报结果，不要日志式的冗长叙述\n"
                "3. 不要使用 ---、***、___ 等装饰性分隔线 — 界面自有视觉分隔\n"
                "4. 制作网页时不要使用 emoji 表情符 — 需要图标时使用 SVG 或 CSS 实现\n"
                "5. 按需规划：简单任务（停止服务器、查看状态、回答问题）直接做，不要列出步骤。"
                "只有任务确实需要 3 步以上不同操作时才拆分计划。"
                "不要为了了解情况而遍历工作空间或读取文件，除非任务本身需要。\n"
                "6. 执行后验证：工具执行后检查结果，如有错误立即诊断并重试修正。"
                "写入/编辑文件后应读取确认内容正确。汇报最终验证后的状态，而非仅说已尝试。"
            )
        return prompt + rules

    def _inject_programming_wisdom(self, prompt: str) -> str:
        """Append programming capability and self-iteration awareness."""
        lang = getattr(self.config.llm, "language", "zh")
        if lang == "en":
            wisdom = (
                "\n\n## Engineering\n"
                "You are a top-tier programming agent:\n"
                "1. Code quality: type safety, error handling, readability, testability\n"
                "2. Debugging: isolate root cause → understand → fix → verify with tests\n"
                "3. Code review: security, performance, maintainability, design patterns\n"
                "4. Self-discipline: study existing code before writing, match project style, run tests before reporting done\n"
                "5. Self-evolution: you can read, analyze, and modify Weather Agents' own source to improve the system"
            )
        else:
            wisdom = (
                "\n\n## 工程能力\n"
                "你是顶级编程智能体：\n"
                "1. 代码质量：类型安全、错误处理、可读性、可测试性\n"
                "2. 调试：定位根因 → 理解 → 修复 → 用测试验证\n"
                "3. 代码审查：安全性、性能、可维护性、设计模式\n"
                "4. 自律：先阅读既有代码再动手，风格与项目保持一致，完成前跑测试\n"
                "5. 自我进化：你可以阅读、分析和修改 Weather Agents 自身代码来改进系统"
            )
        return prompt + wisdom

    def reinit_language(self) -> None:
        """Rebuild system prompt with current language setting.

        Called after a language switch (``/language`` command) to regenerate
        the system prompt in the new language in-place, without losing
        conversation history.
        """
        self._base_system_prompt = ""
        self._base_system_prompt = self._resolve_system_prompt()
        self._base_system_prompt = self._inject_workspace_info(self._base_system_prompt)
        self._base_system_prompt = self._inject_behavior_rules(self._base_system_prompt)
        self._base_system_prompt = self._inject_programming_wisdom(self._base_system_prompt)
        self._base_system_prompt += "\n\n" + self._current_time_tag()
        for msg in self.memory.short_term:
            if msg.role == "system":
                msg.content = self._base_system_prompt
                return
        self.memory.add_message("system", self._base_system_prompt)

    async def init(self) -> None:
        """Initialize agent (memory, subscriptions, skills, etc). Idempotent."""
        if self._base_system_prompt:
            return
        await self.memory.init_db()
        self._base_system_prompt = self._resolve_system_prompt()
        self._base_system_prompt = self._inject_workspace_info(self._base_system_prompt)
        self._base_system_prompt = self._inject_behavior_rules(self._base_system_prompt)
        self._base_system_prompt = self._inject_programming_wisdom(self._base_system_prompt)
        self._base_system_prompt += "\n\n" + self._current_time_tag()
        if not any(m.role == "system" for m in self.memory.short_term):
            self.memory.add_message("system", self._base_system_prompt)
        self._tools = self.tool_registry.get_tools()
        self._load_skills()
        self.bus.subscribe(self.name, self._handle_event)

    def _load_skills(self) -> None:
        """Store skill references for on-demand activation.

        Skills are NOT pre-loaded (no system prompts, no required_tools merged).
        The agent calls list_skills / use_skill tools to discover and activate
        skills on demand — saves tokens by keeping inactive skill text out of
        the context.
        """
        self._skills = self.skill_registry.get_skills()
        self._register_skill_tools()

    def _register_skill_tools(self) -> None:
        """Register use_skill / list_skills for LLM-driven skill activation.

        The LLM can call list_skills() to see available skills, then
        use_skill(name) to activate one.  The system prompt is rebuilt only
        on activation — no token cost for inactive skills.

        Registered once globally; _call_agent (set by chat_stream) ensures
        the handler reaches the agent that made the call.
        """
        if self.tool_registry.get("use_skill"):
            return  # already registered (shared global registry)

        from weather_agents.core.tool import ToolParameter

        async def _use(name: str) -> str:
            agent = _call_agent
            if agent is None:
                return "Error: no active agent"
            if agent.activate_skill(name):
                skill = next((s for s in agent._skills if s.name == name), None)
                desc = skill.description if skill else ""
                return f"✓ Skill '{name}' activated: {desc}"
            return f"✗ Skill '{name}' not found. Call list_skills to see available options."

        async def _list() -> str:
            agent = _call_agent
            if agent is None:
                return "Error: no active agent"
            skills = agent.get_available_skills()
            if not skills:
                return "No skills available."
            lines = [f"• {s['name']}: {s['description']}" for s in skills]
            return "Available skills:\n" + "\n".join(lines)

        self.tool_registry.register(
            Tool(
                name="list_skills",
                description=(
                    "List all available skills with their names and descriptions. "
                    "Use this first to discover what skills you can activate."
                ),
                parameters=[],
                handler=_list,
            )
        )
        self.tool_registry.register(
            Tool(
                name="use_skill",
                description=(
                    "Activate a named skill to gain specialized capabilities "
                    "(e.g. code_reviewer for code review, web_research for research). "
                    "Call list_skills first to see available options."
                ),
                parameters=[
                    ToolParameter(
                        name="name",
                        type="string",
                        description="The name of the skill to activate",
                        required=True,
                    ),
                ],
                handler=_use,
            )
        )

    def activate_skill(self, name: str) -> bool:
        """Activate a skill by name. Invokes handler for custom tool injection.

        Applies skill-level config overrides (model, temperature, max_tokens)
        when the skill is activated. Later activations take precedence for
        conflicting overrides.

        Searches both pre-loaded skills and the global registry, allowing
        runtime activation of any registered skill.
        """
        skill = next((s for s in self._skills if s.name == name), None)
        if not skill:
            skill = self.skill_registry.get(name)
            if skill:
                self._skills.append(skill)
        if not skill:
            return False
        self._active_skills.add(name)
        if skill.handler:
            handler_tools = skill.handler(self, self.tool_registry)
            if handler_tools:
                self._skill_tools[name] = [t.name for t in handler_tools]
        # Merge config overrides (later activations take precedence)
        overrides: dict = {}
        if skill.model:
            overrides["model"] = skill.model
        if skill.temperature is not None:
            overrides["temperature"] = skill.temperature
        if skill.max_tokens is not None:
            overrides["max_tokens"] = skill.max_tokens
        if overrides:
            self._skill_config_overrides[name] = overrides
        self._rebuild_system_prompt()
        return True

    def deactivate_skill(self, name: str) -> bool:
        """Deactivate a skill. Removes handler-injected tools and config overrides."""
        if name not in self._active_skills:
            return False
        self._active_skills.discard(name)
        # Remove handler-injected tools
        for tool_name in self._skill_tools.pop(name, []):
            self.tool_registry.unregister(tool_name)
        # Remove config overrides
        self._skill_config_overrides.pop(name, None)
        self._rebuild_system_prompt()
        return True

    def deactivate_all_skills(self) -> None:
        """Deactivate all skills, remove handler tools, restore base prompt."""
        for name in list(self._active_skills):
            self.deactivate_skill(name)

    def _rebuild_system_prompt(self) -> None:
        """Rebuild the system prompt with active skill prompts appended."""
        if not self._active_skills:
            prompt = self._base_system_prompt
        else:
            skill_prompts = [
                skill.system_prompt
                for skill in self._skills
                if skill.name in self._active_skills and skill.system_prompt
            ]
            prompt = self._base_system_prompt
            if skill_prompts:
                prompt += "\n\n" + "\n\n".join(skill_prompts)

        for _i, msg in enumerate(self.memory.short_term):
            if msg.role == "system":
                msg.content = prompt
                break
        else:
            self.memory.add_message("system", prompt)

    def get_active_skills(self) -> list[str]:
        return list(self._active_skills)

    def get_skill_config_overrides(self) -> dict:
        """Merge config overrides from all active skills.

        Later activations take precedence for conflicting keys.
        Returns a dict with optional keys: model, temperature, max_tokens.
        """
        merged: dict = {}
        for overrides in self._skill_config_overrides.values():
            merged.update(overrides)
        return merged

    def get_available_skills(self) -> list[dict]:
        return [
            {
                "name": s.name,
                "description": s.description,
                "active": s.name in self._active_skills,
            }
            for s in self._skills
        ]

    async def close(self) -> None:
        await self.memory.close()
        self.bus.unsubscribe(self.name)

    async def _set_state(self, new_state: AgentState) -> None:
        if self.state != new_state:
            old_state = self.state
            self.state = new_state
            event = Event(
                type=EventType.STATE_CHANGE,
                source=self.name,
                data={"old_state": old_state.value, "new_state": new_state.value},
            )
            self.bus.add_event(event)
            await self.bus.notify_state_change(event)

    async def _handle_event(self, event: Event) -> None:
        if event.type == EventType.TASK_ASSIGNED and event.target == self.name:
            task = Task(**event.data)
            result = await self.execute_task(task)
            await self.bus.publish(
                Event(
                    type=EventType.TASK_COMPLETED,
                    source=self.name,
                    target=event.source,
                    data={
                        "task_id": task.id,
                        "success": result.success,
                        "content": result.content,
                    },
                )
            )

    async def chat(
        self,
        message: str,
        on_status: Callable[[str], None] | None = None,
    ) -> str:
        """General-purpose chat mode with optional status callback.

        Args:
            message: User message.
            on_status: Called with a status string when state changes
                       (e.g. "thinking...", "calling read_file...").
        """
        await self._set_state(AgentState.THINKING)
        self.memory.add_message("user", message)

        if self._should_auto_compact():
            await self.compact()

        try:
            if on_status:
                on_status("thinking...")
            response = await self._llm_loop(on_status=on_status)
            self.memory.add_message(
                "assistant",
                response.content,
                tool_calls=response.tool_calls,
                reasoning_content=response.reasoning_content,
            )
            await self._set_state(AgentState.IDLE)
            return response.content
        except Exception as e:
            await self._set_state(AgentState.ERROR)
            self._pop_last_user_message()
            self.memory._prune_dangling_tool_calls()
            error_msg = f"[{self.display_name}] Error: {e}"
            self.memory.add_message("assistant", error_msg)
            return error_msg

    async def chat_stream(self, message: str) -> AsyncIterator[dict]:
        """Streaming chat with tool-call support.

        Yields: {"type": "content", "text": "..."} | {"type": "tool_status", "label": "..."} | {"type": "done"}
        """
        global _call_agent
        _call_agent = self
        await self._set_state(AgentState.THINKING)
        self.memory.add_message("user", message)
        assistant_stored = False

        # Auto-compress when context gets too large
        if self._should_auto_compact():
            await self.compact()

        try:
            full_content = ""
            for _iteration in range(self._max_tool_rounds):
                messages = self.memory.get_messages()
                tool_names = self._active_tool_names()

                tool_calls_received: list[dict] = []
                streaming_reasoning: str | None = None
                stream_usage: dict | None = None
                round_content = ""
                async for event in self.llm.stream_with_tools(
                    messages=messages,
                    agent_name=self.name,
                    tools=tool_names or None,
                    tool_registry=self.tool_registry if tool_names else None,
                    overrides=self.get_skill_config_overrides() or None,
                ):
                    if event.type == "content":
                        full_content += event.text
                        round_content += event.text
                        yield {"type": "content", "text": event.text}
                    elif event.type == "tool_call" and event.tool_call:
                        tool_calls_received.append(event.tool_call)
                    elif event.type == "error":
                        yield {"type": "content", "text": f"\n[Error: {event.text}]"}
                        await self._set_state(AgentState.ERROR)
                        if not assistant_stored:
                            self._pop_last_user_message()
                        return
                    elif event.type == "reasoning" and event.text:
                        yield {"type": "reasoning", "text": event.text}
                    elif event.type == "done":
                        stream_usage = event.usage
                        streaming_reasoning = event.reasoning_content

                if not tool_calls_received:
                    self.memory.add_message(
                        "assistant",
                        round_content,
                        reasoning_content=streaming_reasoning,
                    )
                    assistant_stored = True
                    await self._set_state(AgentState.IDLE)
                    yield {"type": "done"}
                    return

                # Record assistant message with tool calls
                self.memory.add_message(
                    "assistant",
                    round_content,
                    tool_calls=tool_calls_received,
                    reasoning_content=streaming_reasoning,
                )
                assistant_stored = True

                if stream_usage:
                    self.bus.add_event(
                        Event(
                            type=EventType.LLM_CALL,
                            source=self.name,
                            data={"model": "", "usage": stream_usage},
                        )
                    )

                # ── Phase 1: Parse args, look up tools, emit status (serial, fast) ──
                tool_prep: list[dict] = []
                for tc in tool_calls_received:
                    tool_name = tc["function"]["name"]
                    raw_args = tc["function"]["arguments"]
                    if isinstance(raw_args, str):
                        tool_args = _parse_tool_args(raw_args)
                        parse_error = (
                            f"Invalid JSON in tool call arguments for '{tool_name}': {raw_args[:200]}"
                            if tool_args is None
                            else None
                        )
                    else:
                        tool_args = raw_args
                        parse_error = None

                    self.bus.add_event(
                        Event(
                            type=EventType.TOOL_CALL,
                            source=self.name,
                            data={"tool": tool_name, "args": tool_args or {}},
                        )
                    )

                    tool_label = (
                        _tool_status_label(tool_name, tool_args)
                        if tool_args
                        else f"{tool_name} (bad args)"
                    )
                    yield {"type": "tool_status", "label": tool_label}
                    tool_prep.append(
                        dict(
                            tc=tc,
                            tool_name=tool_name,
                            tool_args=tool_args,
                            parse_error=parse_error,
                            tool_label=tool_label,
                        )
                    )

                # ── Phase 2: Execute all tool handlers in parallel ──
                async def _exec_one(p: dict) -> tuple:
                    tc = p["tc"]
                    tool_name = p["tool_name"]
                    tool_args = p["tool_args"]
                    if tool_args is None:
                        return (tc, p["parse_error"], False, tool_name)

                    tool = self.tool_registry.get(tool_name)
                    if not tool:
                        return (tc, f"Tool '{tool_name}' not found", False, tool_name)

                    if tool.dangerous:
                        _log.warning(
                            "dangerous_tool_call",
                            extra={
                                "tool": tool_name,
                                "agent": self.name,
                                "tool_args": dict(tool_args) if tool_args else {},
                            },
                        )
                    await self._set_state(AgentState.ACTING)
                    try:
                        result = await tool.execute(**tool_args)
                        return (tc, result, True, tool_name)
                    except Exception as exc:
                        _log.exception("Tool '%s' execution failed: %s", tool_name, exc)
                        return (tc, f"Tool '{tool_name}' execution failed: {exc}", False, tool_name)

                exec_results = await asyncio.gather(*[_exec_one(p) for p in tool_prep])

                # ── Phase 3: Record results in original order ──
                for tc, result, success, _tool_name in exec_results:
                    self.memory.add_message("tool", result, name=_tool_name, tool_call_id=tc["id"])
                    label = next(p["tool_label"] for p in tool_prep if p["tc"] is tc)
                    yield {"type": "tool_done", "label": label, "success": success}
            # Max iterations reached
            if not assistant_stored:
                self._pop_last_user_message()
            await self._set_state(AgentState.IDLE)
            yield {"type": "done"}

        except Exception as e:
            if not assistant_stored:
                self._pop_last_user_message()
            await self._set_state(AgentState.ERROR)
            err_text = str(e) or type(e).__name__
            yield {"type": "content", "text": f"\n[Error: {err_text}]"}
        finally:
            # Clean up any orphaned tool_calls that lack corresponding tool results.
            # Critical when the stream is interrupted (Esc) mid-tool-execution:
            # the assistant message with tool_calls was already persisted but tool
            # results were never written. Without this the next LLM call will fail
            # with "insufficient tool message" from providers like DeepSeek.
            self.memory._prune_dangling_tool_calls()

    def _pop_last_user_message(self) -> None:
        """Remove the most recent user message from short-term memory.

        Used to clean up after an error so the conversation history doesn't
        contain a dangling user message with no assistant response.
        """
        for i in range(len(self.memory.short_term) - 1, -1, -1):
            if self.memory.short_term[i].role == "user":
                self.memory.short_term.pop(i)
                break

    async def compact(self, keep_recent: int = 8) -> str:
        """Compress conversation context by summarising older messages.

        Keeps system prompt intact, replaces old messages with a summary,
        and retains the most recent *keep_recent* messages.
        """
        system_msgs = [m for m in self.memory.short_term if m.role == "system"]
        non_system = [m for m in self.memory.short_term if m.role != "system"]

        if len(non_system) <= keep_recent + 4:
            return "context is already compact"

        to_summarize = non_system[:-keep_recent]
        recent = non_system[-keep_recent:]

        text = ""
        for m in to_summarize:
            role = m.role
            content = (m.content or "")[:300]
            if m.tool_calls:
                names = ",".join(tc["function"]["name"] for tc in m.tool_calls)
                content += f" [tools: {names}]"
            text += f"[{role}] {content}\n"

        resp = await self.llm.complete(
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Summarise this conversation into a single paragraph (<500 chars). "
                        "Keep all key facts, decisions, code snippets, file paths, and context:\n\n"
                        + text
                    ),
                }
            ],
            agent_name=self.name,
            overrides=self.get_skill_config_overrides() or None,
        )
        summary = resp.content.strip()[:600]

        self.memory.short_term = system_msgs.copy()
        self.memory.add_message(
            "user",
            f"[Context compressed: {len(to_summarize)} earlier messages summarised]\n\n{summary}",
        )
        self.memory.add_message("assistant", "Got it, continuing with full context.")
        for m in recent:
            self.memory.short_term.append(m)
        self.memory.prune_tool_messages()

        return f"compressed {len(to_summarize)} messages ({len(summary)} char summary)"

    def context_usage(self) -> dict:
        """Return current context usage stats for display."""
        from weather_agents.core.config import get_model_context_window

        usage = self.memory.get_context_window_usage()
        model = self.llm._get_model(self.name)
        max_ctx = get_model_context_window(model)
        est_tokens = usage["estimated_tokens"]
        return {
            "estimated_tokens": est_tokens,
            "max_tokens": max_ctx,
            "pct": int(est_tokens / max_ctx * 100) if max_ctx else 0,
            "message_count": usage["message_count"],
            "model": model,
        }

    def _should_auto_compact(self) -> bool:
        """Check whether context should be auto-compressed."""
        usage = self.memory.get_context_window_usage()
        from weather_agents.core.config import get_model_context_window

        model = self.llm._get_model(self.name)
        max_ctx = get_model_context_window(model)
        # Trigger auto-compact at 75% of context window
        return int(usage["estimated_tokens"]) > max_ctx * 0.75

    def _active_tool_names(self) -> list[str]:
        """All tool names available to this agent (registry tools + active skill tools)."""
        names = self.tool_registry.list_names()
        seen = set(names)
        for skill in self._skills:
            if skill.name not in self._active_skills:
                continue
            for tool_name in skill.required_tools:
                if tool_name not in seen:
                    names.append(tool_name)
                    seen.add(tool_name)
        return names

    async def _llm_loop(
        self,
        max_iterations: int = 10,
        on_status: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """LLM reasoning loop with tool calling support."""
        response = LLMResponse(content="")
        tool_names = self._active_tool_names()

        try:
            for _ in range(max_iterations):
                messages = self.memory.get_messages()
                if on_status:
                    on_status("thinking...")
                response = await self.llm.complete(
                    messages=messages,
                    agent_name=self.name,
                    tools=tool_names or None,
                    overrides=self.get_skill_config_overrides() or None,
                )

                if not response.tool_calls:
                    return response

                self.bus.add_event(
                    Event(
                        type=EventType.LLM_CALL,
                        source=self.name,
                        data={"model": response.model, "usage": response.usage},
                    )
                )

                # Record assistant message with tool_calls
                self.memory.add_message(
                    "assistant",
                    response.content or "",
                    tool_calls=response.tool_calls,
                    reasoning_content=response.reasoning_content,
                )

                for tc in response.tool_calls:
                    tool_name = tc["function"]["name"]
                    raw_args = tc["function"]["arguments"]
                    if isinstance(raw_args, str):
                        tool_args = _parse_tool_args(raw_args)
                        if tool_args is None:
                            parse_error = f"Invalid JSON in tool call arguments for '{tool_name}': {raw_args[:200]}"
                    else:
                        tool_args = raw_args

                    tool = self.tool_registry.get(tool_name)
                    tool_label = (
                        _tool_status_label(tool_name, tool_args)
                        if tool_args
                        else f"{tool_name} (bad args)"
                    )

                    self.bus.add_event(
                        Event(
                            type=EventType.TOOL_CALL,
                            source=self.name,
                            data={"tool": tool_name, "args": tool_args or {}},
                        )
                    )

                    if on_status:
                        on_status(tool_label)

                    if tool_args is None:
                        self.memory.add_message(
                            "tool",
                            parse_error,
                            name=tool_name,
                            tool_call_id=tc["id"],
                        )
                    elif tool:
                        if tool.dangerous:
                            _log.warning(
                                "dangerous_tool_call",
                                extra={
                                    "tool": tool_name,
                                    "agent": self.name,
                                    "tool_args": dict(tool_args) if tool_args else {},
                                },
                            )
                        await self._set_state(AgentState.ACTING)
                        result = await tool.execute(**tool_args)
                        self.memory.add_message(
                            "tool",
                            result,
                            name=tool_name,
                            tool_call_id=tc["id"],
                        )
                    else:
                        self.memory.add_message(
                            "tool",
                            f"Tool '{tool_name}' not found",
                            name=tool_name,
                            tool_call_id=tc["id"],
                        )

                await self._set_state(AgentState.THINKING)

            return response
        except Exception:
            # On tool execution failure: remove any orphaned tool_calls that
            # lack corresponding tool results so the next call doesn't fail.
            self.memory._prune_dangling_tool_calls()
            raise

    async def execute_task(
        self,
        task: Task,
        on_status: Callable[[str], None] | None = None,
    ) -> TaskResult:
        """Execute a specific task using agent specialty."""
        await self._set_state(AgentState.THINKING)
        task.status = "in_progress"
        self.memory.set_working("current_task", task)

        prompt = f"Please complete this task: {task.description}"
        if task.metadata:
            ctx_data = {k: v for k, v in task.metadata.items() if k != "goal"}
            if ctx_data:
                prompt += f"\nContext: {json.dumps(ctx_data, ensure_ascii=False)}"

        self.memory.add_message("user", prompt)

        try:
            response = await self._llm_loop(on_status=on_status)
            self.memory.add_message(
                "assistant",
                response.content,
                tool_calls=response.tool_calls,
                reasoning_content=response.reasoning_content,
            )
            task.status = "completed"
            task.result = response.content
            await self._set_state(AgentState.IDLE)
            return TaskResult(success=True, content=response.content)
        except Exception as e:
            task.status = "failed"
            task.result = str(e)
            self.memory._prune_dangling_tool_calls()
            await self._set_state(AgentState.ERROR)
            return TaskResult(success=False, content=str(e))

    async def request_help(self, target_agent: str, description: str) -> None:
        """Request another agent's assistance."""
        await self.bus.publish(
            Event(
                type=EventType.AGENT_REQUEST,
                source=self.name,
                target=target_agent,
                data={"description": description},
            )
        )

    def get_status(self) -> dict:
        usage = self.llm.get_usage_stats().get(self.name, {})
        return {
            "name": self.name,
            "display_name": self.display_name,
            "emoji": self.emoji,
            "specialty": self.specialty,
            "state": self.state.value,
            "skills": self.get_available_skills(),
            "usage": {
                "calls": usage.get("calls", 0),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "cost": round(usage.get("cost", 0.0), 6),
            },
        }


# -- Helpers ---------------------------------------------------------------

_TOOL_LABELS: dict[str, str] = {
    "read_file": "Reading {path}",
    "write_file": "Writing {path}",
    "edit_file": "Editing {path}",
    "list_directory": "Listing {path}",
    "file_search": "Searching {directory}/{pattern}",
    "code_search": "Searching for '{query}'",
    "grep": "Grepping '{pattern}'",
    "shell_exec": "Running: {command}",
    "http_get": "GET {url}",
    "http_post": "POST {url}",
    "web_search": "Searching: {query}",
    "move_file": "Moving {src}",
    "copy_file": "Copying {src}",
    "delete_file": "Deleting {path}",
    "get_cwd": "Getting working directory",
    "tree": "Tree {directory}",
    "lint_file": "Linting {path}",
    "scan_deps": "Scanning {directory}",
    "fetch_page": "Fetching {url}",
    "delegate_to": "Delegating to {agent}: {task}",
    "use_skill": "Activating {name}",
    "list_skills": "Listing available skills",
    "git_status": "Git status",
    "git_diff": "Git diff",
    "git_log": "Git log",
    "git_add": "Git add {files}",
    "git_commit": "Git commit",
    "git_checkout": "Git checkout {branch}",
}


def _parse_tool_args(raw: str) -> dict | None:
    """Parse tool call JSON with multi-stage repair for LLM output quirks.

    Handles: markdown fences, Python literals, backtick quotes, single quotes,
    unquoted keys, trailing commas, key=value syntax, unquoted string values,
    trailing text, and unbalanced braces.
    """
    import re as _re

    if not raw or not raw.strip():
        return None

    cleaned = raw.strip()

    # ── 1. Direct parse ────────────────────────────────────────────────────
    try:
        return cast(dict, json.loads(cleaned))
    except json.JSONDecodeError:
        pass

    # ── 2. Strip markdown code fences ──────────────────────────────────────
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned[3:]
        cleaned = cleaned.rsplit("```", 1)[0] if "```" in cleaned else cleaned
        cleaned = cleaned.strip()
        try:
            return cast(dict, json.loads(cleaned))
        except json.JSONDecodeError:
            pass

    # ── 3. Extract first JSON object/array from surrounding text ───────────
    obj_match = _re.search(r"(\{.*\}|\[.*\])", cleaned, _re.DOTALL)
    if obj_match:
        cleaned = obj_match.group(1)
        try:
            return cast(dict, json.loads(cleaned))
        except json.JSONDecodeError:
            pass

    # ── 4. Key=value format: query="weather", count=5 → {"query": "weather", "count": 5}
    #    Typically from models that emit function-call-style rather than JSON.
    if not cleaned.startswith("{") and _re.search(r"\b\w[\w\d_]*\s*=", cleaned):
        kv_pairs: list[str] = []
        for m in _re.finditer(r'(\w[\w\d_]*)\s*=\s*("[^"]*"|\'[^\']*\'|[\w\d_.+-]+)', cleaned):
            key = m.group(1)
            val = m.group(2)
            if val.startswith("'") and val.endswith("'"):
                val = '"' + val[1:-1] + '"'
            kv_pairs.append(f'"{key}": {val}')
        if kv_pairs:
            json_str = "{" + ", ".join(kv_pairs) + "}"
            json_str = _re.sub(r":\s*None\s*([,}])", r": null\1", json_str)
            json_str = _re.sub(r":\s*True\s*([,}])", r": true\1", json_str)
            json_str = _re.sub(r":\s*False\s*([,}])", r": false\1", json_str)
            return cast(dict, json.loads(json_str))

    # ── 5. Python → JSON literals ──────────────────────────────────────────
    #    Must happen before quote transformations to avoid corrupting strings.
    cleaned = _re.sub(r"\bNone\b", "null", cleaned)
    cleaned = _re.sub(r"\bTrue\b", "true", cleaned)
    cleaned = _re.sub(r"\bFalse\b", "false", cleaned)

    # ── 6. Backtick → double quote ────────────────────────────────────────
    cleaned = cleaned.replace("`", '"')

    # ── 7. Fix single-quote strings ────────────────────────────────────────
    if "'" in cleaned:
        cleaned = cleaned.replace("'", '"')

    # ── 8. Fix unquoted keys: {key: "value"} → {"key": "value"} ────────────
    cleaned = _re.sub(r"([{,]\s*)(\w[\w\d_]*)(\s*:)", r'\1"\2"\3', cleaned)

    # ── 9. Fix trailing commas before ] or } ───────────────────────────────
    cleaned = _re.sub(r",\s*([}\]])", r"\1", cleaned)
    cleaned = cleaned.rstrip(",").strip()

    # ── 10. Fix unquoted string values: {"key": bare word} → {"key": "bare word"} ──
    cleaned = _re.sub(
        r"(:\s*)([a-zA-Z_.][a-zA-Z0-9_ ./\\@.\-+#~$]*?)(\s*[,}\]])",
        lambda m: (
            m.group(0)
            if m.group(2) in ("null", "true", "false")
            or m.group(2).lstrip("-").replace(".", "").isdigit()
            or m.group(2).startswith(('"', "{", "["))
            else f'{m.group(1)}"{m.group(2)}"{m.group(3)}'
        ),
        cleaned,
    )

    # ── 11. Attempt parse ──────────────────────────────────────────────────
    try:
        return cast(dict, json.loads(cleaned))
    except json.JSONDecodeError:
        pass

    # ── 12. Balanced-brace extraction ──────────────────────────────────────
    depth = 0
    start = -1
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return cast(dict, json.loads(cleaned[start : i + 1]))
                except json.JSONDecodeError:
                    pass

    # If a JSON object was started but never closed, try auto-closing
    if start >= 0 and depth > 0:
        candidate = cleaned[start:] + "}" * depth
        try:
            return cast(dict, json.loads(candidate))
        except json.JSONDecodeError:
            pass

    return None


def _tool_status_label(name: str, args: dict) -> str:
    """Build a human-readable one-liner for a tool call."""
    template = _TOOL_LABELS.get(name)
    if template:
        try:
            label = template.format_map(args)
        except (KeyError, IndexError):
            label = f"{name}..."
    else:
        label = f"{name}..."
    # Truncate long labels
    if len(label) > 60:
        label = label[:57] + "..."
    return label

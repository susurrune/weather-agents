"""Base agent class for all Weather Agents."""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import cast

from weather_agents.core.bus import Event, EventType, MessageBus
from weather_agents.core.config import AppConfig
from weather_agents.core.constants import TASK_DONE_SENTINEL
from weather_agents.core.llm import LLMClient, LLMResponse
from weather_agents.core.logger import get_logger
from weather_agents.core.memory import Memory, Message
from weather_agents.core.skill import Skill, SkillRegistry
from weather_agents.core.tool import Tool, ToolRegistry

_log = get_logger("agent")


class AgentState(StrEnum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    WAITING = "waiting"
    ERROR = "error"


class TaskState(StrEnum):
    """Task lifecycle states actually used in execution.

    Prior versions defined QUEUED/ASSIGNED/VALIDATING/RETRYING here but no
    code path ever transitioned tasks into those states, so the validation
    gate they implied was a fiction. The current set reflects what the
    orchestrator and retry logic really do.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


_VALID_TRANSITIONS: dict[TaskState, set[TaskState]] = {
    TaskState.PENDING: {TaskState.RUNNING, TaskState.SKIPPED, TaskState.FAILED},
    # RUNNING -> RUNNING is allowed: orchestrator marks the outer plan task
    # RUNNING, and the inner execute_task_impl also wants to mark its own
    # task RUNNING. Without the self-loop one of them would raise and the
    # call site had to suppress(ValueError).
    TaskState.RUNNING: {TaskState.RUNNING, TaskState.COMPLETED, TaskState.FAILED},
    # FAILED -> RUNNING enables retries to actually move the state forward
    # instead of silently swallowing transition errors.
    TaskState.FAILED: {TaskState.RUNNING, TaskState.SKIPPED},
    TaskState.COMPLETED: set(),
    TaskState.SKIPPED: set(),
}


@dataclass
class Task:
    id: str
    description: str
    assigned_to: str | None = None
    parent_id: str | None = None
    depends_on: list[str] = field(default_factory=list)
    status: TaskState = TaskState.PENDING
    priority: int = 0
    result: str | None = None
    metadata: dict = field(default_factory=dict)

    def transition_to(self, new_state: TaskState) -> None:
        """Validate and apply state transition."""
        allowed = _VALID_TRANSITIONS.get(self.status, set())
        if new_state not in allowed:
            raise ValueError(
                f"Invalid task state transition: {self.status.value} → {new_state.value}"
            )
        self.status = new_state

    @property
    def all_deps(self) -> list[str]:
        """Return all dependency IDs (depends_on + legacy parent_id)."""
        deps = list(self.depends_on)
        if self.parent_id and self.parent_id not in deps:
            deps.append(self.parent_id)
        return deps


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
        self._max_tool_rounds: int = agent_cfg.max_tool_rounds if agent_cfg else 20
        # Hard ceiling for auto-extension. Lowered from 100 → 40 after the
        # PPT case study: a single user turn legitimately needs <30 rounds
        # for almost any task; hitting 100 was always a stuck loop the
        # detector failed to catch, not real progress. 40 keeps a buffer
        # over the soft default (20) for genuinely complex turns but caps
        # the worst-case wall clock at ~5 minutes instead of ~10+.
        self._max_tool_rounds_hard_cap: int = 40
        # Fire-and-forget fact-extraction state. We count completed chat turns
        # so the extractor only runs every N (default 10), keeping cost low.
        self._user_turns_since_extract: int = 0
        self._pending_extracts: set = set()
        # Pending inter-agent request futures, keyed by correlation_id.
        self._pending_requests: dict[str, asyncio.Future] = {}
        # Tracked background tasks so unhandled exceptions don't go silently.
        self._bg_tasks: set[asyncio.Task] = set()
        # Human-in-loop approval gate: the CLI (or test) sets this to a
        # callable that prompts the user.  ``None`` = auto-approve.
        self.approval_callback: Callable[[str, dict], Awaitable[bool]] | None = None
        # Serialize entry into chat/chat_stream/execute_task on this agent.
        # Concurrent turns on the same agent (e.g. snow.gather(
        # delegate_to(fog, A), delegate_to(fog, B))) used to interleave
        # short_term messages mid-turn — A appends "user A", B appends
        # "user B", A reads short_term and sees BOTH user messages as its
        # own context, LLM returns a confused assistant_A based on both
        # tasks. The lock makes concurrent callers queue instead of mix.
        self._turn_lock: asyncio.Lock = asyncio.Lock()

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
                f"`{ws_str}` — write to `files/`, `output/`, `temp/`. "
                f"Prefer workspace paths for all file ops."
            )
        else:
            ws_block = (
                f"\n\n## 工作空间\n"
                f"`{ws_str}` — 产物写到 `files/` / `output/` / `temp/`。"
                f"文件操作优先用此路径。"
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
                '- Act, don\'t narrate. No "I will..." before tool calls.\n'
                "- Stay in scope. Do what's asked, then stop.\n"
                "- Batch independent tool calls in one response.\n"
                "- Verify writes: read back, report verified state.\n"
                "- No decorative `---` lines; no emoji in generated web pages."
            )
        else:
            rules = (
                "\n\n## 行为守则\n"
                "- 直接行动,不预告。不说「我将要...」,直接调用工具\n"
                "- 不擅自扩大范围。用户要什么做什么,核心完成即止\n"
                "- 独立的工具调用一次发出,并行执行\n"
                "- 写入后回读验证,汇报已验证状态而非仅尝试\n"
                "- 不用 `---` 装饰线,网页里不用 emoji"
            )
        return prompt + rules

    def _inject_programming_wisdom(self, prompt: str) -> str:
        """Append programming capability and self-iteration awareness."""
        lang = getattr(self.config.llm, "language", "zh")
        if lang == "en":
            wisdom = (
                "\n\n## Engineering\n"
                "Top-tier engineer: type-safe code, real error handling, debugging by root cause, "
                "reviewing for security & perf. You can read and modify Weather Agents' own source."
            )
        else:
            wisdom = (
                "\n\n## 工程能力\n"
                "顶级工程师:类型安全、真实的错误处理、按根因调试、按安全与性能审查。"
                "你可以阅读和修改 Weather Agents 自身源码。"
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
        # Route through _rebuild_system_prompt so the runtime identity
        # block (current model id, language-aware) is regenerated in
        # the new language too.
        self._rebuild_system_prompt()

    async def init(self) -> None:
        """Initialize agent (memory, subscriptions, skills, etc). Idempotent."""
        if self._base_system_prompt:
            return
        await self.memory.init_db()
        # Always own a session. Prefer resuming this agent's most recent
        # session so `wa chat` feels continuous across invocations; fall back
        # to creating a new one for first run or when WA_NO_RESUME=1 is set
        # (e.g. tests asserting cross-process isolation).
        import os as _os

        if self.memory.get_active_session() is None:
            resumed = None
            if _os.environ.get("WA_NO_RESUME") != "1":
                resumed = await self.memory.resume_latest_session()
            if resumed is None:
                await self.memory.create_session()
        self._base_system_prompt = self._resolve_system_prompt()
        self._base_system_prompt = self._inject_workspace_info(self._base_system_prompt)
        self._base_system_prompt = self._inject_behavior_rules(self._base_system_prompt)
        self._base_system_prompt = self._inject_programming_wisdom(self._base_system_prompt)
        self._base_system_prompt += "\n\n" + self._current_time_tag()
        # Route through _rebuild_system_prompt so the runtime identity
        # block (current model id) is always present on first turn —
        # otherwise the LLM has to guess what model it's running on and
        # tends to hallucinate "I'm Claude" regardless of the actual
        # provider.
        self._rebuild_system_prompt()
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

        Each agent has its OWN ToolRegistry (created in factory.create_system_context
        per agent, not the global singleton), so the closures defined here
        capture this agent's ``self``. Cross-agent leakage is impossible by
        construction — there is no ContextVar involved.
        """
        if self.tool_registry.get("use_skill"):
            return  # already registered on this agent's registry

        from weather_agents.core.tool import ToolParameter

        async def _use(name: str) -> str:
            if self.activate_skill(name):
                skill = next((s for s in self._skills if s.name == name), None)
                desc = skill.description if skill else ""
                return f"✓ Skill '{name}' activated: {desc}"
            return f"✗ Skill '{name}' not found. Call list_skills to see available options."

        async def _list() -> str:
            skills = self.get_available_skills()
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

        async def _extend(n: int = 10) -> str:
            old = self._max_tool_rounds
            self._max_tool_rounds += n
            return (
                f"✓ Tool-round limit extended by {n} "
                f"(was {old}, now {self._max_tool_rounds}). "
                "You have more iterations to complete the task."
            )

        self.tool_registry.register(
            Tool(
                name="extend_rounds",
                description=(
                    "Extend the tool-call budget for the current turn. Call this "
                    "when you know you need more iterations to finish — e.g. before "
                    "the limit is hit. Accepts a positive integer (default 10). "
                    "The limit can be extended multiple times per turn."
                ),
                parameters=[
                    ToolParameter(
                        name="n",
                        type="integer",
                        description="Number of additional rounds to add (default 10)",
                        required=False,
                    ),
                ],
                handler=_extend,
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

    def _auto_activate_skills(self, message: str) -> list[str]:
        """Activate skills whose triggers substring-match the user message.

        Saves the LLM a full ``list_skills + use_skill`` round-trip (~2s)
        when the user types something whose intent matches a declared
        trigger phrase. Returns the names of newly-activated skills (mainly
        useful for tests). Skills already active are skipped — no churn.
        """
        if not message:
            return []
        lowered = message.lower()
        candidates = list(self._skills)
        # Also consider globally-registered skills not yet imported into
        # _skills — auto-trigger is most useful for skills loaded via the
        # skill_registry that the agent hasn't bound yet.
        for s in self.skill_registry.get_skills():
            if s.name not in {x.name for x in candidates}:
                candidates.append(s)
        activated: list[str] = []
        for skill in candidates:
            if skill.name in self._active_skills:
                continue
            if not skill.triggers:
                continue
            for trig in skill.triggers:
                if trig and trig.lower() in lowered:
                    if self.activate_skill(skill.name):
                        activated.append(skill.name)
                    break
        return activated

    def _runtime_identity_block(self) -> str:
        """One-line "you are running on model X" block injected into the
        system prompt so the LLM can answer "what model are you?"
        truthfully instead of guessing (DeepSeek pretending to be Claude
        was the bug that triggered this). Computed at rebuild time so
        ``/model`` switches take effect immediately after the next
        ``_rebuild_system_prompt()`` call.
        """
        # Mirror LLMClient._get_model's resolution but read from the
        # agent's own config so the helper stays testable without a
        # real LLMClient (mock LLMs would otherwise return a Mock auto-
        # attribute for _get_model and we'd serialize that into the
        # prompt). Per-agent override wins; falls back to default.
        model = self.config.llm.default_model
        try:
            agent_cfg = getattr(self.config.agents, self.name, None)
            if agent_cfg and getattr(agent_cfg, "model", None):
                model = str(agent_cfg.model)
        except Exception:
            pass
        lang = getattr(self.config.llm, "language", "zh")
        if lang == "en":
            return (
                f"\n\n## Runtime\n"
                f"You are the {self.display_name} agent in Weather Agents, "
                f"powered by the **{model}** language model. If the user asks "
                "what model you are, give them this exact model id verbatim — "
                "do NOT guess or claim to be a different model."
            )
        return (
            f"\n\n## 运行环境\n"
            f"你是 Weather Agents 中的「{self.display_name}」智能体，"
            f"底层语言模型为 **{model}**。如果用户问你是什么模型，"
            "直接告诉他们这个准确的 model id —— 不要猜测、不要冒充其他模型。"
        )

    def _rebuild_system_prompt(self) -> None:
        """Rebuild the system prompt with active skill prompts appended.

        Skill prompts are concatenated in **sorted skill name order**, not
        activation order. A stable prefix means upstream prompt-caching
        layers (Anthropic/DeepSeek prefix cache, our own llm cache) see the
        same byte sequence across turns where the active-skill set is
        identical, saving 100-500ms on first-token latency.

        The runtime identity block (current model name) is appended LAST
        so its byte position is stable across turns where only the model
        changed — same cache-friendly principle as the skill ordering.
        """
        identity = self._runtime_identity_block()
        if not self._active_skills:
            prompt = self._base_system_prompt + identity
        else:
            # Index skills by name so we don't depend on _skills list order.
            by_name = {s.name: s for s in self._skills}
            skill_prompts: list[str] = []
            for name in sorted(self._active_skills):
                s = by_name.get(name)
                if not s:
                    continue
                parts: list[str] = []
                if s.system_prompt:
                    parts.append(s.system_prompt)
                lang = getattr(self.config.llm, "language", "zh")
                # Lazy-load hint. Bodies over a few KB are truncated at
                # parse time to keep the active prompt small (see
                # Skill.from_markdown). Tell the LLM where to find the
                # full content so it can read_file on demand instead of
                # operating from a stub.
                if s.body_truncated and s.source_path:
                    if lang == "en":
                        parts.append(
                            f"[Lazy-loaded skill: only the summary is in "
                            f"your prompt. Full guide at "
                            f"`{s.source_path}` — call read_file on that "
                            f"path when you need detailed instructions "
                            f"or examples.]"
                        )
                    else:
                        parts.append(
                            f"[此技能为懒加载：当前提示中只放了摘要。"
                            f"完整指南位于 `{s.source_path}` —— "
                            f"需要详细步骤或示例时，用 read_file 读取该文件。]"
                        )
                # Append resource-dir hint so the LLM can find skill assets
                # without searching the filesystem first.
                if s.resource_dir:
                    if lang == "en":
                        parts.append(f"Resource directory: {s.resource_dir}")
                    else:
                        parts.append(f"资源目录: {s.resource_dir}")
                skill_prompts.append("\n\n".join(parts))
            prompt = self._base_system_prompt
            if skill_prompts:
                prompt += "\n\n" + "\n\n".join(skill_prompts)
            prompt += identity

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
        # Drain in-flight background work BEFORE closing memory. Fact
        # extraction and any other _bg_tasks may still be writing to the
        # SQLite DB; closing the DB first turns those writes into swallowed
        # exceptions and silently loses the data.
        pending = list(self._pending_extracts) + list(self._bg_tasks)
        if pending:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True),
                    timeout=10.0,
                )
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
        elif event.type == EventType.AGENT_REQUEST and event.target == self.name:
            t = asyncio.create_task(self._handle_request(event))
            self._bg_tasks.add(t)
            t.add_done_callback(self._bg_tasks.discard)
        elif event.type == EventType.AGENT_RESPONSE and event.target == self.name:
            self._handle_response(event)

    async def chat_oneshot(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """One-off LLM call with no tools, no memory persistence, no skills.

        For auxiliary calls the agent makes to itself (judge / summary /
        classification / rewrite) where the response should NOT pollute
        chat history and tool-loop machinery is pure overhead. By default
        routes to ``llm.lightweight_model`` if configured — a deepseek
        flash variant or haiku run at ~10× lower cost than the reasoning
        models we use for the main loop. Pass ``model=`` to override.

        Skips:
          - the tool_router scan (~5-50ms),
          - _messages_with_recall (~30-150ms on a populated fact store),
          - skill-prompt assembly,
          - add_message persistence on both prompt and reply.

        Returns the assistant content string (empty on LLM error — the
        caller decides whether that's recoverable).
        """
        overrides: dict = {}
        # Resolve model: explicit arg > configured lightweight > None
        # (None means LLMClient picks the agent's default).
        if model is not None:
            overrides["model"] = model
        else:
            lw = getattr(self.config.llm, "lightweight_model", None)
            if isinstance(lw, str) and lw.strip():
                overrides["model"] = lw
        if temperature is not None:
            overrides["temperature"] = temperature
        if max_tokens is not None:
            overrides["max_tokens"] = max_tokens

        messages = [{"role": "user", "content": prompt}]
        response = await self.llm.complete(
            messages=messages,
            agent_name=self.name,
            tools=None,
            overrides=overrides or None,
        )
        return response.content

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
        async with self._turn_lock:
            return await self._chat_impl(message, on_status)

    async def _chat_impl(
        self,
        message: str,
        on_status: Callable[[str], None] | None,
    ) -> str:
        await self._set_state(AgentState.THINKING)
        self.memory.add_message("user", message)

        # Auto-compact may fail (LLM transient error, etc.). Don't let that
        # nuke the turn — log and proceed without compaction. The user's
        # message is already persisted; we'd otherwise leave a dangling
        # user turn with no response.
        if self._should_auto_compact():
            try:
                await self.compact()
            except Exception as exc:
                _log.warning("auto_compact_failed: %s", exc)

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
            self._maybe_extract_facts()
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
        # Auto-activate skills whose triggers appear in the user message.
        # Saves a full LLM round-trip (list_skills + use_skill ~2s+tokens)
        # for common patterns where the trigger is unambiguous. Only fires
        # when the skill isn't already active, so explicit /skill activation
        # still takes precedence.
        activated_now = self._auto_activate_skills(message)
        # Serialize concurrent turns on this agent so short_term doesn't
        # get interleaved by two parallel callers (see _turn_lock comment).
        async with self._turn_lock:
            try:
                async for ev in self._chat_stream_impl(message, auto_activated=activated_now):
                    yield ev
            except BaseException:
                # If _chat_stream_impl crashed before persisting the assistant
                # response, remove the dangling user message so memory stays
                # consistent — every user message should have a matching
                # assistant response or be cleaned up.
                if self.memory.short_term and self.memory.short_term[-1].role == "user":
                    self._pop_last_user_message()
                raise

    async def _chat_stream_impl(
        self,
        message: str,
        *,
        auto_activated: list[str] | None = None,
    ) -> AsyncIterator[dict]:
        await self._set_state(AgentState.THINKING)
        self.memory.add_message("user", message)
        assistant_stored = False

        # Auto-compress when context gets too large. Wrap in try/except so a
        # flaky summariser LLM call doesn't strand the user's already-
        # persisted message with no response — without this guard a 5xx on
        # the summariser appeared to users as a "dropped" turn.
        if self._should_auto_compact():
            try:
                await self.compact()
            except Exception as exc:
                _log.warning("auto_compact_failed: %s", exc)

        # Track delegations that occurred this turn so we can synthesize a
        # short content line if the model ends the turn after only emitting
        # delegate_to tool calls (no plain text). Without this the REPL's
        # "empty response, retrying..." path fires even though work
        # actually happened.
        delegations: list[tuple[str, bool]] = []  # (target_agent, success)

        # Tools that returned a [CircuitBreakerOpen] error this turn — we
        # drop them from the active tool set for the rest of the turn so the
        # LLM doesn't waste iterations re-calling a known-broken tool.
        suppressed_tools: set[str] = set()
        # When an auto-trigger already activated the matching skill(s),
        # the LLM has zero reason to call list_skills — that's a full
        # extra round-trip (~1-2s + the skill catalog ~1k tokens) to
        # re-derive information we already used. Suppressing the tool
        # for the rest of the turn directly removes the temptation.
        # Inject a one-line system note so the model knows it shouldn't
        # try (and so it can decide to call use_skill for something
        # else if needed). use_skill stays available — the user might
        # mention several skills the trigger heuristic doesn't catch.
        if auto_activated:
            suppressed_tools.add("list_skills")
            self.memory.add_message(
                "system",
                (
                    "[Auto-activated skills: "
                    + ", ".join(auto_activated)
                    + "] These were chosen from your message's keywords; "
                    "their prompts and tools are already loaded. Do NOT "
                    "call list_skills — proceed directly with the task. "
                    "Only call use_skill(name) if you need a DIFFERENT "
                    "skill that wasn't auto-activated."
                ),
            )
        # Rolling window of recent tool successes — used to detect stuck
        # loops where the LLM keeps trying variations of a failing search.
        recent_tool_outcomes: list[bool] = []
        stuck_hint_injected = False
        # Rolling window of the agent's recent text responses — used to
        # detect "narration loops" where the model keeps writing the
        # same "现在要做 X" sentence in different phrasings without
        # actually moving forward. Distinct from the tool-failure loop
        # signal: this fires on success-with-repetition.
        recent_response_texts: list[str] = []
        repetition_hint_injected = False
        # Rolling fingerprints of recent tool calls — catches "tool loop"
        # patterns the text-similarity check misses. The PPT case study:
        # edit_file on the same path 20+ times with micro-coordinate
        # changes (y=5.35→5.5→5.25→5.35) succeeded every round and never
        # tripped the failure-marker or narration-similarity detectors,
        # but was clearly stuck. A signature counter over the last
        # _SIG_WINDOW calls catches this directly.
        recent_tool_sigs: list[str] = []
        tool_loop_hint_injected = False

        # Per-turn tool-selection cache. `select_relevant_tools` scores every
        # candidate tool against the user message — that's O(n_tools)
        # tokenization per round. The user message is fixed for the whole
        # turn, so re-running it every iteration of a 20-round loop wastes
        # ~50-200ms total + churns garbage. Cache the result and invalidate
        # only when the inputs (suppressed_tools, active skills) actually
        # change.
        from weather_agents.core.tool_router import select_relevant_tools as _select

        tool_names_cache: list[str] | None = None
        cache_key: tuple[frozenset[str], frozenset[str]] | None = None

        def _resolve_tool_names() -> list[str]:
            nonlocal tool_names_cache, cache_key
            key = (frozenset(suppressed_tools), frozenset(self._active_skills))
            if tool_names_cache is not None and cache_key == key:
                return tool_names_cache
            candidates = [t for t in self._active_tool_names() if t not in suppressed_tools]
            must = {
                t for s in self._skills if s.name in self._active_skills for t in s.required_tools
            }
            tool_names_cache = _select(
                self.tool_registry,
                candidates,
                message,
                must_include=must,
            )
            cache_key = key
            return tool_names_cache

        try:
            full_content = ""
            _round_limit = self._max_tool_rounds
            _round_count = 0
            while True:
                if _round_count >= _round_limit:
                    if _round_limit >= self._max_tool_rounds_hard_cap:
                        break
                    extend_by = min(15, self._max_tool_rounds_hard_cap - _round_limit)
                    _round_limit += extend_by
                    self._max_tool_rounds = _round_limit
                    self.memory.add_message(
                        "system",
                        f"[Auto-extended tool-round limit by {extend_by} to {_round_limit}. Continue working.]",
                    )
                    continue
                _round_count += 1
                # extend_rounds tool may raise the limit mid-turn.
                _round_limit = max(_round_limit, self._max_tool_rounds)
                messages = await self._messages_with_recall()
                tool_names = _resolve_tool_names()

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
                        if not assistant_stored:
                            self._pop_last_user_message()
                        await self._set_state(AgentState.IDLE)
                        return
                    elif event.type == "reasoning" and event.text:
                        yield {"type": "reasoning", "text": event.text}
                    elif event.type == "done":
                        stream_usage = event.usage
                        streaming_reasoning = event.reasoning_content

                if not tool_calls_received:
                    final_content = round_content
                    if not full_content.strip() and delegations:
                        # Model returned no text but did delegate. Use the
                        # synthesized summary as BOTH the displayed content
                        # AND the persisted assistant content so memory and
                        # the user's screen agree on what this turn produced.
                        final_content = _synthesize_delegation_summary(delegations)
                    self.memory.add_message(
                        "assistant",
                        final_content,
                        reasoning_content=streaming_reasoning,
                    )
                    assistant_stored = True
                    await self._set_state(AgentState.IDLE)
                    self._maybe_extract_facts()
                    if final_content and final_content != round_content:
                        yield {"type": "content", "text": final_content}
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
                            _format_args_parse_error(tool_name, raw_args)
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
                        else f"{tool_name} (unparseable args)"
                    )
                    yield {
                        "type": "tool_status",
                        "label": tool_label,
                        "tool_name": tool_name,
                        "args": tool_args or {},
                    }
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
                        # Suggest similar real tool names so the LLM can
                        # self-correct on the next iteration instead of
                        # repeatedly hallucinating the same wrong name.
                        suggestions = _suggest_tool_names(tool_name, self.tool_registry)
                        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
                        # Drop this name for the rest of the turn so the
                        # LLM doesn't burn iterations calling it again.
                        suppressed_tools.add(tool_name)
                        return (
                            tc,
                            f"Error: Tool '{tool_name}' does not exist.{hint}",
                            False,
                            tool_name,
                        )

                    if tool.dangerous:
                        _log.warning(
                            "dangerous_tool_call",
                            extra={
                                "tool": tool_name,
                                "agent": self.name,
                                "tool_args": dict(tool_args) if tool_args else {},
                            },
                        )
                        if not await self._check_tool_approval(tool_name, tool_args):
                            return (
                                tc,
                                f"[denied] dangerous tool '{tool_name}' blocked",
                                False,
                                tool_name,
                            )
                    await self._set_state(AgentState.ACTING)
                    try:
                        result = await tool.execute(agent_name=self.name, **tool_args)
                        return (tc, result, True, tool_name)
                    except Exception as exc:
                        _log.exception("Tool '%s' execution failed: %s", tool_name, exc)
                        return (tc, f"Tool '{tool_name}' execution failed: {exc}", False, tool_name)

                exec_results = await asyncio.gather(*[_exec_one(p) for p in tool_prep])

                # ── Phase 3: Record results in original order ──
                task_completed = False
                for tc, result, success, _tool_name in exec_results:
                    if isinstance(result, str) and "[CircuitBreakerOpen]" in result:
                        suppressed_tools.add(_tool_name)
                    # task_done() signals the agent believes the task is complete.
                    # Store the user-facing summary instead of the sentinel.
                    if _tool_name == "task_done" and result == TASK_DONE_SENTINEL:
                        task_completed = True
                        prep = next(p for p in tool_prep if p["tc"] is tc)
                        summary = (prep.get("tool_args") or {}).get("summary", "")
                        display_result = (
                            f"[Task completed: {summary}]" if summary else "[Task completed]"
                        )
                        self.memory.add_message(
                            "tool", display_result, name=_tool_name, tool_call_id=tc["id"]
                        )
                        yield {
                            "type": "tool_done",
                            "label": f"task_done: {summary}" if summary else "task_done",
                            "success": True,
                            "tool_name": "task_done",
                            "result": display_result,
                        }
                        continue
                    self.memory.add_message("tool", result, name=_tool_name, tool_call_id=tc["id"])
                    label = next(p["tool_label"] for p in tool_prep if p["tc"] is tc)
                    yield {
                        "type": "tool_done",
                        "label": label,
                        "success": success,
                        "tool_name": _tool_name,
                        "result": (result[:800] if result else "")
                        if isinstance(result, str)
                        else str(result)[:800]
                        if result
                        else "",
                    }
                    if _tool_name == "delegate_to":
                        # Recover the target agent name from the parsed args
                        # so we can report which sub-agent ran on this turn.
                        prep = next(p for p in tool_prep if p["tc"] is tc)
                        target = (prep.get("tool_args") or {}).get("agent", "?")
                        delegations.append((target, success))
                # Agent called task_done — exit cleanly without truncation warning.
                if task_completed:
                    if not assistant_stored:
                        self._pop_last_user_message()
                    await self._set_state(AgentState.IDLE)
                    yield {"type": "done"}
                    return

                # Narration-loop detection: catch the agent rephrasing the
                # same "现在要做 X" sentence round after round without
                # progressing. Distinct from tool-failure detection because
                # tools may be succeeding — the model is just generating
                # similar prose between actions. Compare this round's
                # content against the previous 1-2 rounds.
                normalized_round = round_content.strip()
                if normalized_round and recent_response_texts:
                    high_sim = any(
                        _text_similarity(normalized_round, prev) >= 0.7
                        for prev in recent_response_texts[-2:]
                    )
                    if high_sim and not repetition_hint_injected:
                        self.memory.add_message(
                            "system",
                            (
                                "[Stop narrating] Your last response is highly "
                                "similar to your previous one — you are "
                                "re-describing what you're about to do instead "
                                "of doing it. Stop writing prose. Either:\n"
                                "1. Emit ONLY the next tool call(s) with no "
                                "explanatory text, or\n"
                                "2. If the task is genuinely done, output the "
                                "final deliverable and stop.\n"
                                "Do NOT write another 'I am now going to ...' "
                                "or '接下来要 ...' paragraph."
                            ),
                        )
                        repetition_hint_injected = True
                recent_response_texts.append(normalized_round)
                if len(recent_response_texts) > 3:
                    recent_response_texts.pop(0)

                # Tool-signature loop: same (tool, primary-arg) fingerprint
                # called repeatedly = the model is micro-tuning the same
                # output and unable to converge. Distinct from failure /
                # narration loops because the calls SUCCEED every round.
                # The 4/_SIG_WINDOW threshold hints; 6/_SIG_WINDOW hard-
                # escapes — past that point further iterations have
                # historically NEVER produced a better result and only
                # burn the round budget.
                for p in tool_prep:
                    if p["tool_name"] in ("task_done", "list_skills", "use_skill"):
                        continue
                    sig = _tool_call_signature(p["tool_name"], p["tool_args"])
                    if sig:
                        recent_tool_sigs.append(sig)
                if len(recent_tool_sigs) > _SIG_WINDOW:
                    del recent_tool_sigs[: len(recent_tool_sigs) - _SIG_WINDOW]
                if recent_tool_sigs:
                    from collections import Counter as _Counter

                    top_sig, top_count = _Counter(recent_tool_sigs).most_common(1)[0]
                    if top_count >= _SIG_LOOP_HINT and not tool_loop_hint_injected:
                        self.memory.add_message(
                            "system",
                            (
                                f"[Tool loop] You have called `{top_sig}` "
                                f"{top_count} times in the last "
                                f"{len(recent_tool_sigs)} tool calls — you are "
                                "iterating on the same operation without "
                                "converging. STOP repeating it. Either: "
                                "(1) accept the current state and write the "
                                "final answer, (2) try a fundamentally "
                                "different approach (different tool / "
                                "different target), or (3) call task_done "
                                "if the work is good enough."
                            ),
                        )
                        tool_loop_hint_injected = True
                    if top_count >= _SIG_LOOP_HARDSTOP:
                        self.memory.add_message(
                            "assistant",
                            (
                                f"I have repeated `{top_sig}` {top_count} "
                                "times without converging on a better "
                                "outcome. Stopping here to avoid burning "
                                "the rest of the round budget on the same "
                                "micro-adjustment. Current artifacts (if "
                                "any) are the final state."
                            ),
                        )
                        yield {
                            "type": "content",
                            "text": (
                                f"\n\n[stuck] tool `{top_sig}` repeated "
                                f"{top_count}× in the last "
                                f"{len(recent_tool_sigs)} calls — "
                                "stopping rather than micro-tuning further.\n"
                            ),
                        }
                        await self._set_state(AgentState.IDLE)
                        yield {"type": "done"}
                        return

                # Stuck-loop detection: track outcomes of this round's tool
                # calls and, if the recent window is mostly failures, inject a
                # system nudge so the LLM stops grinding through dead-ends. The
                # hint fires once per turn — after that the model has been told,
                # and repeating the same advice every round is just noise.
                for _tc, result, success, _tool_name in exec_results:
                    if _tool_name == "task_done":
                        continue
                    failed = (not success) or (
                        isinstance(result, str) and _looks_like_failed_tool_result(result)
                    )
                    recent_tool_outcomes.append(not failed)
                    if len(recent_tool_outcomes) > 6:
                        recent_tool_outcomes.pop(0)
                if (
                    not stuck_hint_injected
                    and len(recent_tool_outcomes) >= 5
                    and sum(recent_tool_outcomes) <= 1
                ):
                    self.memory.add_message(
                        "system",
                        (
                            "[Recovery hint] Your last several tool calls have "
                            "mostly failed (blocked sources, no results, "
                            "timeouts). Stop trying more variations of the "
                            "same approach. Either: (1) synthesize a partial "
                            "answer from any tool outputs that DID succeed, "
                            "(2) draw on your general knowledge and clearly "
                            "label it as such, or (3) ask the user for "
                            "guidance. Do NOT keep calling tools that are "
                            "failing — finalize your answer now."
                        ),
                    )
                    stuck_hint_injected = True

                # Hard escape: if the LLM ignored the soft hint and 8+ of
                # the most-recent tool calls are still failing, break out
                # ourselves. Better to return an honest "I'm stuck" than
                # burn the entire 20-round budget rejecting the same dead
                # ends. The user can ask again with more context.
                if len(recent_tool_outcomes) >= 8 and sum(recent_tool_outcomes) == 0:
                    self.memory.add_message(
                        "assistant",
                        (
                            "I tried multiple approaches but every tool call is "
                            "failing (the file/path may not exist, search "
                            "backends are blocked, or the resource is "
                            "unavailable). Please give me more context — a "
                            "correct file path, an alternative source, or "
                            "clarification on what you're looking for."
                        ),
                    )
                    yield {
                        "type": "content",
                        "text": (
                            "\n\n[stuck] every recent tool call failed — "
                            "stopping rather than burning more iterations.\n"
                        ),
                    }
                    await self._set_state(AgentState.IDLE)
                    yield {"type": "done"}
                    return
            # Max iterations reached
            if not assistant_stored:
                self._pop_last_user_message()
            await self._set_state(AgentState.IDLE)
            if not full_content.strip() and delegations:
                synth = _synthesize_delegation_summary(delegations)
                # Persist the synthesized summary so memory matches what
                # the user saw; otherwise the next turn's LLM sees only an
                # orphaned tool result with no assistant follow-up.
                self.memory.add_message("assistant", synth)
                yield {"type": "content", "text": synth}
            # Signal that the answer is incomplete because we exhausted the
            # tool-call budget. The CLI renders this as a dim warning so the
            # user understands why the agent stopped mid-task.
            yield {
                "type": "truncated",
                "reason": f"max tool rounds ({self._max_tool_rounds}) reached",
            }
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
        with self.memory._short_term_lock:
            for i in range(len(self.memory.short_term) - 1, -1, -1):
                if self.memory.short_term[i].role == "user":
                    self.memory.short_term.pop(i)
                    break

    async def compact(self, keep_recent: int = 12) -> str:
        """Compress conversation context by summarising older messages.

        Keeps system prompt intact, replaces old messages with a terse
        bulleted digest inserted as a *system* note, and retains the most
        recent *keep_recent* messages verbatim.

        The digest is a system message (not a fake user/assistant exchange)
        so the model treats it as background context and does NOT continue
        or re-narrate it on the next turn — the root cause of the "之前你说
        了几次…" rambling observed earlier.
        """
        # Keep only true system prompts — exclude previous compaction digests
        # so they don't accumulate into an ever-growing stack of summaries.
        system_msgs = [
            m
            for m in self.memory.short_term
            if m.role == "system" and not (m.content or "").startswith("[Earlier-context digest")
        ]
        non_system = [m for m in self.memory.short_term if m.role != "system"]

        if len(non_system) <= keep_recent + 4:
            return "context is already compact"

        to_summarize = non_system[:-keep_recent]
        recent = non_system[-keep_recent:]

        # Extract user directives verbatim — these MUST survive compaction.
        # Heuristic: imperative/negative rules and short user messages with
        # constraint keywords. Lossy but biased toward keeping rules.
        directive_keywords = (
            "don't",
            "do not",
            "never",
            "always",
            "must",
            "no ",
            "不要",
            "不准",
            "禁止",
            "必须",
            "一定",
            "记住",
        )
        directives: list[str] = []
        for m in to_summarize:
            if m.role != "user":
                continue
            content = (m.content or "").strip()
            if not content or len(content) > 300:
                continue
            lower = content.lower()
            if any(k in lower for k in directive_keywords):
                directives.append(content)

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
                        "Produce a TERSE factual digest of the conversation below. "
                        "Strict format rules:\n"
                        "- Output bullet points only (one fact per line, prefix '- ').\n"
                        "- No narrative, no 'previously you', no commentary, no apology.\n"
                        "- Each bullet ≤ 80 chars: a single fact, decision, file path, or constraint.\n"
                        "- Preserve every user directive (don't / never / 必须 / 禁止 / 记住) verbatim in quotes.\n"
                        "- Maximum 12 bullets total. Drop trivial chit-chat.\n\n" + text
                    ),
                }
            ],
            agent_name=self.name,
            overrides=self.get_skill_config_overrides() or None,
        )
        summary = resp.content.strip()[:800]

        digest_parts = [
            f"[Earlier-context digest — {len(to_summarize)} messages compressed. "
            "Reference only. Do NOT acknowledge, continue, or re-narrate this digest.]",
            summary,
        ]
        if directives:
            digest_parts.append("Verbatim user directives to obey:")
            digest_parts.extend(f'  - "{d}"' for d in directives[-8:])

        # compact() runs across a long await (the digest LLM call), during
        # which other turns may have added messages. We snapshot system+recent
        # at the END and write atomically under the short-term lock so we don't
        # clobber messages appended in the interim.
        with self.memory._short_term_lock:
            self.memory.short_term = system_msgs.copy()
            # System-role insertion: persistence layer skips system rows, so this
            # stays purely in-process and never bleeds into other sessions.
            self.memory.add_message("system", "\n".join(digest_parts))
            for m in recent:
                self.memory.short_term.append(m)
        self.memory.prune_tool_messages()

        return f"compressed {len(to_summarize)} messages ({len(summary)} char digest)"

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
        # Trigger auto-compact at 92% of the context window. Lower
        # thresholds (was 75%) fired in normal-sized sessions and caused
        # users to see digest fragments surfacing mid-conversation. Now
        # auto-compact is a hard-near-limit safety net, not a routine
        # background operation; users can still trigger it explicitly via
        # /compact when they want it.
        return int(usage["estimated_tokens"]) > max_ctx * 0.92

    def _active_tool_names(self) -> list[str]:
        """All tool names available to this agent (registry tools + active
        skill tools), filtered by any active skill's ``allowed_tools``.

        Anthropic skill-spec semantics: a skill that declares
        ``allowed-tools`` in its frontmatter wants the LLM restricted to
        ONLY those tools while it's active. When multiple skills are
        active their allowed_tools sets UNION (combined capabilities
        rather than the empty intersection). A single active skill with
        no allowed_tools acts as a wildcard — restriction is lifted.
        """
        names = self.tool_registry.list_names()
        seen = set(names)
        # Allowed-tools accumulation
        restriction: set[str] | None = None
        any_unrestricted = False
        for skill in self._skills:
            if skill.name not in self._active_skills:
                continue
            for tool_name in skill.required_tools:
                if tool_name not in seen:
                    names.append(tool_name)
                    seen.add(tool_name)
            if skill.allowed_tools is None:
                any_unrestricted = True
            else:
                if restriction is None:
                    restriction = set()
                restriction.update(skill.allowed_tools)
        # Wildcard wins: any active skill without restriction lifts the
        # cap entirely. Empty active set also = no restriction.
        if restriction is not None and not any_unrestricted:
            names = [n for n in names if n in restriction]
        return names

    # -- Automatic fact extraction (durable long-term memory) -----------------

    EXTRACT_PROMPT_TEMPLATE = """你是一个事实抽取助手。从下面的对话中抽取**用户透露的稳定、可复用的事实**。

**应该抽取**：
- 工具/技术偏好（pkg_mgr=pnpm, editor=neovim, framework=FastAPI）
- 项目信息（project_lang=Python, project_name=weather-agents）
- 长期目标（goal=build_url_shortener）
- 关键约束（os=Windows, python_version=3.13）

**不要抽取**：
- 用户的情绪、心情
- 当前任务的临时细节（"帮我写这个函数"中的"这个"）
- 对话过程中的中间产物
- 任何不确定的信息

**规则**：
- 只在用户**明确陈述**时抽取（"我用 X"、"项目是 X"）
- key 用 snake_case 英文，value 简短
- 如无可抽取的稳定事实，输出 `[]`

**输出格式**：纯 JSON 数组，不要任何额外文字、不要 markdown 围栏：
[{{"key": "pkg_mgr", "value": "pnpm", "category": "user_pref"}}]

对话：
{conversation}

输出："""

    def _maybe_extract_facts(self) -> None:
        """Schedule a fact-extraction pass every N user turns (default 10).

        Fire-and-forget — schedules an async task and returns immediately so
        it doesn't slow down the user's next prompt. WA_NO_EXTRACT=1 or
        WA_EXTRACT_EVERY_N=0 disables; both are read each call so users can
        flip them mid-session.
        """
        import os as _os

        if _os.environ.get("WA_NO_EXTRACT") == "1":
            return
        try:
            # 默认从 10 提到 20: 抽取是次要的 LLM 调用,fact 命中率不高的项目
            # 每 10 轮跑一次纯属烧 token。WA_EXTRACT_EVERY_N 仍可手动调低。
            every_n = int(_os.environ.get("WA_EXTRACT_EVERY_N", "20"))
        except ValueError:
            every_n = 20
        if every_n <= 0:
            return

        self._user_turns_since_extract += 1
        if self._user_turns_since_extract < every_n:
            return
        self._user_turns_since_extract = 0

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._extract_facts_async())
        self._pending_extracts.add(task)
        task.add_done_callback(self._pending_extracts.discard)

    async def _extract_facts_async(self) -> int:
        """Run one fact-extraction pass against recent conversation.

        Returns the number of facts written. Exceptions are logged but never
        raised — extraction is best-effort, never blocks the main flow.
        Callable directly from tests to assert behavior without juggling
        the create_task indirection.
        """
        try:
            recent = self.memory.short_term[-20:]
            convo_msgs = [m for m in recent if m.role in ("user", "assistant") and m.content]
            if len(convo_msgs) < 4:
                return 0
            convo_text = "\n".join(f"{m.role}: {m.content[:500]}" for m in convo_msgs)
            prompt = self.EXTRACT_PROMPT_TEMPLATE.format(conversation=convo_text)
            response = await self.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                agent_name=f"{self.name}_extract",
                tools=None,
            )
            facts = self._parse_extracted_facts(response.content)
            written = 0
            for f in facts:
                key = f.get("key")
                value = f.get("value")
                category = f.get("category") or "auto_extracted"
                if not isinstance(key, str) or not key.strip() or value in (None, ""):
                    continue
                await self.memory.remember(key.strip(), value, category=str(category))
                written += 1
            if written:
                _log.info(
                    "auto_extracted_facts",
                    extra={"agent": self.name, "count": written},
                )
            return written
        except Exception as exc:
            _log.warning("fact_extract_failed: %s", exc)
            return 0

    @staticmethod
    def _parse_extracted_facts(content: str) -> list[dict]:
        """Best-effort JSON array extraction from LLM response.

        Tries: raw JSON parse → markdown-fenced ``` json ``` → first ``[...]``
        substring in the message. Returns ``[]`` on every failure path so the
        caller can iterate safely.
        """
        import re

        text = (content or "").strip()
        if not text:
            return []
        # 1. Direct parse
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [f for f in data if isinstance(f, dict)]
        except (json.JSONDecodeError, TypeError):
            pass
        # 2. Markdown-fenced JSON
        m = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
        if m:
            try:
                data = json.loads(m.group(1))
                if isinstance(data, list):
                    return [f for f in data if isinstance(f, dict)]
            except (json.JSONDecodeError, TypeError):
                pass
        # 3. First raw JSON array substring
        m = re.search(r"\[[\s\S]*?\]", text)
        if m:
            try:
                data = json.loads(m.group(0))
                if isinstance(data, list):
                    return [f for f in data if isinstance(f, dict)]
            except (json.JSONDecodeError, TypeError):
                pass
        return []

    async def _messages_with_recall(self) -> list[dict]:
        """Return short-term messages with a 'relevant facts' system block
        injected right before the latest user message.

        Implements the 'retrieval-injection' principle: instead of dumping
        all of long-term memory into the prompt, look up only what's
        relevant to the current turn (by token match against the user's
        latest message). Set WA_NO_RECALL=1 to disable, e.g. for debugging.

        Recall results are cached per (agent, query) — within a single tool
        loop the user's last message doesn't change between iterations, so
        re-running the DB scan + n-gram scoring every iteration is pure
        waste (30-150ms each on a large fact store). Cache is cleared when
        a new user message arrives via _on_user_message_changed.
        """
        import os as _os

        messages = self.memory.get_messages()
        if not messages or _os.environ.get("WA_NO_RECALL") == "1":
            return messages
        last_user_idx = next(
            (i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "user"),
            None,
        )
        if last_user_idx is None:
            return messages
        query = str(messages[last_user_idx].get("content") or "")[:200]
        # Skip recall for trivial inputs — short acknowledgements ("ok", "yes",
        # "好的") have no signal to match against and burn 30-150ms anyway.
        stripped = query.strip()
        if len(stripped) < 4:
            return messages
        # Per-turn cache: same query in the same loop -> same facts.
        cache = getattr(self, "_recall_cache", None)
        if cache is None:
            cache = {}
            self._recall_cache = cache
        cached_facts = cache.get(query)
        if cached_facts is not None:
            facts = cached_facts
        else:
            try:
                facts = await self.memory.recall_for_injection(query, limit=3)
            except Exception:
                return messages
            cache[query] = facts
            # Bound the cache so a very long session doesn't leak memory.
            if len(cache) > 32:
                cache.pop(next(iter(cache)))
        if not facts:
            return messages
        block = self.memory.format_facts_block(facts)
        if not block:
            return messages
        messages.insert(last_user_idx, {"role": "system", "content": block})
        return messages

    async def _llm_loop(
        self,
        max_iterations: int | None = None,
        on_status: Callable[[str], None] | None = None,
        *,
        ephemeral: bool = False,
    ) -> LLMResponse:
        """LLM reasoning loop with tool calling support.

        ephemeral=True 时所有 assistant/tool 中间消息不写 SQLite —— 供
        orchestration sub-task 使用,避免污染 chat 会话历史。
        """
        mi = max_iterations if max_iterations is not None else self._max_tool_rounds
        response = LLMResponse(content="")
        full_tool_names = self._active_tool_names()
        # Use the most recent user message as the routing query. This loop is
        # invoked from execute_task where the task description IS the last
        # user message, which matches user intent closely.
        from weather_agents.core.tool_router import select_relevant_tools

        last_user = next(
            (m.content for m in reversed(self.memory.short_term) if m.role == "user"),
            "",
        )
        must = {t for s in self._skills if s.name in self._active_skills for t in s.required_tools}
        tool_names = select_relevant_tools(
            self.tool_registry,
            full_tool_names,
            last_user or "",
            must_include=must,
        )

        try:
            _limit = mi
            _rounds = 0
            while True:
                if _rounds >= _limit:
                    if _limit >= self._max_tool_rounds_hard_cap:
                        break
                    extend_by = min(15, self._max_tool_rounds_hard_cap - _limit)
                    _limit += extend_by
                    self._max_tool_rounds = _limit
                    self.memory.add_message(
                        "system",
                        f"[Auto-extended tool-round limit by {extend_by} to {_limit}. Continue working.]",
                    )
                    continue
                _rounds += 1
                _limit = max(_limit, self._max_tool_rounds)
                messages = await self._messages_with_recall()
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
                    ephemeral=ephemeral,
                )

                for tc in response.tool_calls:
                    tool_name = tc["function"]["name"]
                    raw_args = tc["function"]["arguments"]
                    if isinstance(raw_args, str):
                        tool_args = _parse_tool_args(raw_args)
                        if tool_args is None:
                            parse_error = _format_args_parse_error(tool_name, raw_args)
                    else:
                        tool_args = raw_args

                    tool = self.tool_registry.get(tool_name)
                    tool_label = (
                        _tool_status_label(tool_name, tool_args)
                        if tool_args
                        else f"{tool_name} (unparseable args)"
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
                            ephemeral=ephemeral,
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
                            if not await self._check_tool_approval(tool_name, tool_args):
                                self.memory.add_message(
                                    "tool",
                                    f"[denied] dangerous tool '{tool_name}' blocked",
                                    name=tool_name,
                                    tool_call_id=tc["id"],
                                    ephemeral=ephemeral,
                                )
                                continue
                        await self._set_state(AgentState.ACTING)
                        result = await tool.execute(agent_name=self.name, **tool_args)
                        self.memory.add_message(
                            "tool",
                            result,
                            name=tool_name,
                            tool_call_id=tc["id"],
                            ephemeral=ephemeral,
                        )
                    else:
                        suggestions = _suggest_tool_names(tool_name, self.tool_registry)
                        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
                        self.memory.add_message(
                            "tool",
                            f"Error: Tool '{tool_name}' does not exist.{hint}",
                            name=tool_name,
                            tool_call_id=tc["id"],
                            ephemeral=ephemeral,
                        )

                await self._set_state(AgentState.THINKING)

            # Loop exhausted without the LLM producing a tool-call-free answer.
            # Surface truncation in the response content so the caller knows
            # the result is incomplete (was previously silent).
            response.truncated = True
            if not response.content:
                response.content = (
                    f"[truncated] max tool rounds ({mi}) reached without a "
                    "final answer; latest tool calls were not followed up."
                )
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
        async with self._turn_lock:
            return await self._execute_task_impl(task, on_status)

    async def _execute_task_impl(
        self,
        task: Task,
        on_status: Callable[[str], None] | None,
    ) -> TaskResult:
        await self._set_state(AgentState.THINKING)
        task.transition_to(TaskState.RUNNING)
        self.memory.set_working("current_task", task)

        prompt = (
            "Complete this task NOW using your available tools. Then write "
            "the actual deliverable content in your final reply.\n\n"
            f"Task: {task.description}"
        )
        if task.metadata:
            ctx_data = {k: v for k, v in task.metadata.items() if k != "goal"}
            if ctx_data:
                prompt += f"\nContext: {json.dumps(ctx_data, ensure_ascii=False)}"

        # Orchestration-task isolation. Sub-task 提示词不能污染 chat 会话:
        # 1) 内存层: 替换 short_term, finally 中恢复;
        # 2) 持久化层: add_message(ephemeral=True) 跳过 SQLite 写入。
        # 否则下次 wa chat 时 resume_latest_session 会把 "Complete this task NOW…"
        # 当成历史对话回放,导致 agent 答非所问。
        with self.memory._short_term_lock:
            saved_short_term = list(self.memory.short_term)
            self.memory.short_term = [m for m in saved_short_term if m.role == "system"]

        self.memory.add_message("user", prompt, ephemeral=True)

        # Snapshot pre-task message count so we can scan ONLY this task's
        # tool calls afterwards (not unrelated history).
        pre_len = len(self.memory.short_term)

        try:
            response = await self._llm_loop(on_status=on_status, ephemeral=True)
            # Inspect tool calls made during THIS task to find files the
            # agent wrote. Even if the model returned a placeholder reply
            # ("已完成"), the actual file paths come from the tool args we
            # can read off short_term. The orchestrator's judge / the user
            # need these paths to verify the deliverable.
            file_paths = _extract_file_paths_from_messages(self.memory.short_term[pre_len:])
            enriched = _enrich_response_with_artifacts(response.content, file_paths)
            self.memory.add_message(
                "assistant",
                enriched,
                tool_calls=response.tool_calls,
                reasoning_content=response.reasoning_content,
                ephemeral=True,
            )
            task.transition_to(TaskState.COMPLETED)
            task.result = enriched
            await self._set_state(AgentState.IDLE)
            return TaskResult(success=True, content=enriched)
        except Exception as e:
            task.transition_to(TaskState.FAILED)
            task.result = str(e)
            self.memory._prune_dangling_tool_calls()
            await self._set_state(AgentState.ERROR)
            return TaskResult(success=False, content=str(e))
        finally:
            # Restore chat history so interactive `chat()` continuity isn't
            # broken by the task isolation above.
            with self.memory._short_term_lock:
                self.memory.short_term = saved_short_term

    async def _check_tool_approval(self, tool_name: str, tool_args: dict) -> bool:
        """Check whether a dangerous tool call is approved.

        Delegates to ``approval_callback`` when the config mode is
        ``"interactive"``; auto-approves in ``"auto"`` mode and
        auto-denies in ``"strict"`` mode.
        """
        mode = getattr(self.config.cli, "approval_mode", "auto")
        if mode == "strict":
            _log.info("tool_denied_strict", extra={"tool": tool_name})
            return False
        if mode == "interactive":
            if self.approval_callback is not None:
                return await self.approval_callback(tool_name, tool_args)
            _log.info("tool_denied_no_callback", extra={"tool": tool_name})
            return False
        return True

    async def request_help(self, target_agent: str, description: str, timeout: float = 60.0) -> str:
        """Request another agent's assistance and await the response.

        Uses a correlation ID to match the response to this request via
        the event bus. The target agent's ``_handle_event`` picks up the
        ``AGENT_REQUEST``, processes it as a task, and publishes an
        ``AGENT_RESPONSE`` that resolves the pending future here.

        Returns the response content, or an error message on timeout /
        failure.
        """
        correlation_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending_requests[correlation_id] = future

        await self.bus.publish(
            Event(
                type=EventType.AGENT_REQUEST,
                source=self.name,
                target=target_agent,
                data={
                    "correlation_id": correlation_id,
                    "description": description,
                    "source": self.name,
                },
            )
        )

        try:
            result: str = await asyncio.wait_for(future, timeout=timeout)
            return result
        except TimeoutError:
            self._pending_requests.pop(correlation_id, None)
            if not future.done():
                future.cancel()
            return f"[{target_agent} did not respond within {timeout}s]"

    async def _handle_request(self, event: Event) -> None:
        """Process an incoming AGENT_REQUEST and publish AGENT_RESPONSE.

        Spawned as a background task from ``_handle_event`` so the bus
        handler returns immediately while the potentially-long task
        execution runs in the background.
        """
        description = event.data.get("description", "")
        correlation_id = event.data.get("correlation_id", "")
        source = event.data.get("source", "")
        if not correlation_id:
            return

        task = Task(
            id=f"req-{correlation_id[:8]}",
            description=description,
            assigned_to=self.name,
        )

        try:
            result = await self.execute_task(task)
            await self.bus.publish(
                Event(
                    type=EventType.AGENT_RESPONSE,
                    source=self.name,
                    target=source,
                    data={
                        "correlation_id": correlation_id,
                        "content": result.content,
                        "success": result.success,
                    },
                )
            )
        except Exception as exc:
            await self.bus.publish(
                Event(
                    type=EventType.AGENT_RESPONSE,
                    source=self.name,
                    target=source,
                    data={
                        "correlation_id": correlation_id,
                        "content": f"[error] {exc}",
                        "success": False,
                    },
                )
            )

    def _handle_response(self, event: Event) -> None:
        """Resolve a pending request future from an AGENT_RESPONSE."""
        correlation_id = event.data.get("correlation_id", "")
        if not correlation_id:
            return
        future = self._pending_requests.pop(correlation_id, None)
        if future is not None and not future.done():
            future.set_result(event.data.get("content", ""))

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


_RE_OBJ_OR_ARRAY = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)
_RE_KV_DETECT = re.compile(r"\b\w[\w\d_]*\s*=")
_RE_KV_PAIRS = re.compile(r'(\w[\w\d_]*)\s*=\s*("[^"]*"|\'[^\']*\'|[\w\d_.+-]+)')
_RE_NONE_LITERAL = re.compile(r":\s*None\s*([,}])")
_RE_TRUE_LITERAL = re.compile(r":\s*True\s*([,}])")
_RE_FALSE_LITERAL = re.compile(r":\s*False\s*([,}])")
_RE_PY_NONE = re.compile(r"\bNone\b")
_RE_PY_TRUE = re.compile(r"\bTrue\b")
_RE_PY_FALSE = re.compile(r"\bFalse\b")
_RE_UNQUOTED_KEY = re.compile(r"([{,]\s*)(\w[\w\d_]*)(\s*:)")
_RE_TRAILING_COMMA = re.compile(r",\s*([}\]])")
_RE_UNQUOTED_STRING = re.compile(r"(:\s*)([a-zA-Z_.][a-zA-Z0-9_ ./\\@.\-+#~$]*?)(\s*[,}\]])")


def _parse_tool_args(raw: str) -> dict | None:
    """Parse tool call JSON with multi-stage repair for LLM output quirks.

    Handles: markdown fences, Python literals, backtick quotes, single quotes,
    unquoted keys, trailing commas, key=value syntax, unquoted string values,
    trailing text, and unbalanced braces. All regexes are precompiled at
    module load so the repair stages don't pay re-compile cost per call —
    the fast path (stage 1, plain json.loads) is unaffected but failure-
    repair latency drops ~5-20ms.
    """
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
    obj_match = _RE_OBJ_OR_ARRAY.search(cleaned)
    if obj_match:
        cleaned = obj_match.group(1)
        try:
            return cast(dict, json.loads(cleaned))
        except json.JSONDecodeError:
            pass

    # ── 4. Key=value format: query="weather", count=5 → {"query": "weather", "count": 5}
    #    Typically from models that emit function-call-style rather than JSON.
    if not cleaned.startswith("{") and _RE_KV_DETECT.search(cleaned):
        kv_pairs: list[str] = []
        for m in _RE_KV_PAIRS.finditer(cleaned):
            key = m.group(1)
            val = m.group(2)
            if val.startswith("'") and val.endswith("'"):
                val = '"' + val[1:-1] + '"'
            kv_pairs.append(f'"{key}": {val}')
        if kv_pairs:
            json_str = "{" + ", ".join(kv_pairs) + "}"
            json_str = _RE_NONE_LITERAL.sub(r": null\1", json_str)
            json_str = _RE_TRUE_LITERAL.sub(r": true\1", json_str)
            json_str = _RE_FALSE_LITERAL.sub(r": false\1", json_str)
            return cast(dict, json.loads(json_str))

    # ── 5. Python → JSON literals ──────────────────────────────────────────
    #    Must happen before quote transformations to avoid corrupting strings.
    cleaned = _RE_PY_NONE.sub("null", cleaned)
    cleaned = _RE_PY_TRUE.sub("true", cleaned)
    cleaned = _RE_PY_FALSE.sub("false", cleaned)

    # ── 6. Backtick → double quote ────────────────────────────────────────
    cleaned = cleaned.replace("`", '"')

    # ── 7. Fix single-quote strings ────────────────────────────────────────
    if "'" in cleaned:
        cleaned = cleaned.replace("'", '"')

    # ── 8. Fix unquoted keys: {key: "value"} → {"key": "value"} ────────────
    cleaned = _RE_UNQUOTED_KEY.sub(r'\1"\2"\3', cleaned)

    # ── 9. Fix trailing commas before ] or } ───────────────────────────────
    cleaned = _RE_TRAILING_COMMA.sub(r"\1", cleaned)
    cleaned = cleaned.rstrip(",").strip()

    # ── 10. Fix unquoted string values: {"key": bare word} → {"key": "bare word"} ──
    cleaned = _RE_UNQUOTED_STRING.sub(
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


# Substrings that strongly indicate a tool call produced no useful output.
# Treated as "failure" for stuck-loop detection even when the tool returned
# success=True (e.g. web_search returning "No results found ..." is
# technically a successful call but semantically a dead-end).
_TOOL_FAILURE_MARKERS: tuple[str, ...] = (
    # Search backends returning nothing
    "no results found",
    "no matches for",  # grep / code_search returning empty
    # File ops on non-existent paths — the model's most common dead-end
    # when guessing import paths or searching the wrong tree.
    "file not found",
    "directory not found",
    "permission denied",
    # HTTP failures (4xx / 5xx)
    "status: 4",
    "status: 5",
    # Network / timeout
    "request timed out",
    "timed out",
    "connection refused",
    "name or service not known",
    "ssl",
    # Generic error prefixes from the tool layer
    "[error",
    "error: tool",  # invalid args, etc.
    "error: file",  # file not found wrapper
    "error: directory",
    "circuitbreakeropen",
    "execution failed:",
)


def _looks_like_failed_tool_result(result: str) -> bool:
    """Heuristic: does this tool result indicate a dead-end the LLM should
    stop retrying? Used by stuck-loop detection in _chat_stream_impl."""
    if not result:
        return True
    head = result[:300].lower()
    return any(m in head for m in _TOOL_FAILURE_MARKERS)


# Tools that produce on-disk artifacts. When the orchestration loop scans
# an agent's tool-call history, these are the calls whose `path`/`dst`
# argument names the deliverable.
_FILE_PRODUCING_TOOLS: dict[str, tuple[str, ...]] = {
    "write_file": ("path",),
    "edit_file": ("path",),
    "copy_file": ("dst", "destination"),
    "move_file": ("dst", "destination"),
}


def _extract_file_paths_from_messages(messages: list[Message]) -> list[str]:
    """Walk this task's assistant turns and pull out file paths that
    write_file / edit_file / copy_file / move_file successfully touched.

    The agent's chat reply often says only "已完成" — but the tool_calls on
    each assistant message record exactly which paths it wrote. The very
    next "tool" role message (with matching tool_call_id) tells us whether
    the call actually succeeded. We list successful-write paths in the
    order they were performed, deduplicated.
    """
    # Pair each tool_call_id with the tool result message that followed.
    tool_results: dict[str, str] = {}
    for m in messages:
        if m.role == "tool" and m.tool_call_id:
            tool_results[m.tool_call_id] = m.content or ""

    paths: list[str] = []
    seen: set[str] = set()
    for m in messages:
        if m.role != "assistant" or not m.tool_calls:
            continue
        for tc in m.tool_calls:
            name = tc.get("function", {}).get("name", "")
            arg_keys = _FILE_PRODUCING_TOOLS.get(name)
            if not arg_keys:
                continue
            raw = tc.get("function", {}).get("arguments", "")
            try:
                args = json.loads(raw) if isinstance(raw, str) else (raw or {})
            except json.JSONDecodeError:
                continue
            if not isinstance(args, dict):
                continue
            path: str | None = None
            for k in arg_keys:
                v = args.get(k)
                if isinstance(v, str) and v.strip():
                    path = v.strip()
                    break
            if not path:
                continue
            # Confirm the call actually succeeded — skip "File not found" /
            # "permission denied" results. The tool wrapper returns plain
            # strings; a successful write_file starts with "Successfully ".
            result = tool_results.get(tc.get("id", ""), "")
            if result and result.startswith(("Error:", "[Error", "Error ")):
                continue
            if path not in seen:
                seen.add(path)
                paths.append(path)
    return paths


def _enrich_response_with_artifacts(content: str, file_paths: list[str]) -> str:
    """Append a deterministic artifact footer when the agent wrote files
    during this task but its reply doesn't already mention them.

    Renders as a markdown blockquote with each path wrapped in backticks
    so Rich's Markdown renderer gives the section a distinctive left bar
    + code highlighting. Plain "- path" lines blended into the rest of
    the reply and users frequently missed them.

    Without this, an agent that writes content to disk and replies with
    a 5-char "完成了" leaves the downstream verifier and the user with
    NO way to find the actual deliverable. The footer is concise — just
    the paths — so it doesn't disrupt rich replies that already include
    full content.
    """
    if not file_paths:
        return content
    body = content or ""
    # If every path is already cited in the reply, nothing to add.
    missing = [p for p in file_paths if p not in body]
    if not missing:
        return body
    # Markdown blockquote (>) gives Rich a left-bar style; inline code
    # (`path`) gives the path a distinct color from the surrounding text.
    lines = ["", "> **Artifacts produced**"]
    for p in file_paths:
        marker = " — already cited above" if p in body else ""
        lines.append(f"> - `{p}`{marker}")
    return body.rstrip() + "\n\n" + "\n".join(lines)


# Tool-signature loop detector tuning. Window covers ~last 8 tool calls.
# Hint at 4 repeats (model has space to course-correct on the next round);
# hard-escape at 6 (model already saw the hint and ignored it).
_SIG_WINDOW = 8
_SIG_LOOP_HINT = 4
_SIG_LOOP_HARDSTOP = 6


def _tool_call_signature(tool_name: str, args: dict | None) -> str:
    """Compact, stable fingerprint of a tool call for loop detection.

    For file-mutating tools we anchor on the path: editing the same file
    repeatedly is the classic micro-tuning loop. For shell we anchor on
    the leading command word (subsequent flags vary but the intent
    doesn't). For search we anchor on the lowercased query head.
    Generic fallback is the tool name alone — coarser but safer than
    inventing a key from arguments we don't understand.
    """
    a = args or {}

    def _s(key: str, n: int = 0) -> str:
        v = a.get(key)
        if not isinstance(v, str):
            return ""
        return v.strip()[:n] if n > 0 else v.strip()

    if tool_name == "edit_file":
        # Include old_string hash so different edits to the same file
        # aren't conflated as a loop. Only truly identical edits count.
        os_val = _s('old_text')
        if os_val:
            import hashlib as _hl
            return f"edit_file:{_s('path')}#{_hl.sha1(os_val.encode()).hexdigest()[:8]}"
        return f"edit_file:{_s('path')}"
    if tool_name in ("write_file", "read_file", "delete_file"):
        return f"{tool_name}:{_s('path')}"
    if tool_name in ("copy_file", "move_file"):
        return f"{tool_name}:{_s('src') or _s('source')}->{_s('dst') or _s('destination')}"
    if tool_name in ("run_bash", "bash", "shell", "run_shell"):
        cmd = _s("command") or _s("cmd")
        if not cmd:
            return tool_name
        # First whitespace-separated token captures the program; we drop
        # arguments so "where soffice" called with cosmetic flag changes
        # still folds to the same signature.
        first = cmd.split(None, 1)[0] if cmd else ""
        return f"{tool_name}:{first[:30]}"
    if tool_name in ("web_search", "search", "search_web", "search_files", "grep"):
        q = _s("query", 40) or _s("pattern", 40)
        return f"{tool_name}:{q.lower()}"
    if tool_name == "delegate_to":
        return f"delegate_to:{_s('agent')}"
    return tool_name


def _text_similarity(a: str, b: str) -> float:
    """Cheap similarity for narration-loop detection.

    difflib's SequenceMatcher.ratio is O(N*M) which is fine for the
    typical 100-400 char per-round responses we compare. Returns 0-1.
    Short responses (< 12 chars) are always treated as dissimilar to
    avoid flagging "好的" / brief acknowledgements as loops.
    """
    if len(a) < 12 or len(b) < 12:
        return 0.0
    import difflib as _difflib

    return _difflib.SequenceMatcher(None, a, b).ratio()


def _format_args_parse_error(tool_name: str, raw_args: str) -> str:
    """Produce a tool-result error string that helps the LLM recover.

    Detects the common failure mode where the model hit max_tokens
    mid-content (huge write_file payloads, etc.) and the JSON tool args
    end in an unclosed string. In that case "Invalid JSON" is misleading
    — the args parsed fine syntactically up to where they stopped — so we
    point the model at the real fix (chunk the content or use edit_file).

    The truncation heuristic: count unescaped quotes; if odd, the last
    string was never closed. Combined with no trailing brace, that
    practically always means max_tokens cut the response.
    """
    stripped = raw_args.rstrip()
    has_closing_brace = stripped.endswith(("}", "]"))
    # Crude quote counter — treat backslash as escape one char ahead.
    in_string = False
    i = 0
    while i < len(stripped):
        ch = stripped[i]
        if ch == "\\" and in_string:
            i += 2
            continue
        if ch == '"':
            in_string = not in_string
        i += 1
    looks_truncated = in_string or not has_closing_brace

    preview = raw_args[:200] + ("...[truncated]" if len(raw_args) > 200 else "")
    if looks_truncated:
        return (
            f"Error: tool '{tool_name}' arguments were truncated by the model's "
            f"output budget (max_tokens). The JSON ended mid-value so it cannot "
            f"be parsed. For large content, split into multiple smaller calls "
            f"(write_file with a shorter chunk, then edit_file or write_file to "
            f"append the rest), or break the deliverable into pieces. "
            f"Args preview: {preview}"
        )
    return f"Error: invalid JSON in tool call arguments for '{tool_name}': {preview}"


def _suggest_tool_names(missing: str, registry: ToolRegistry, max_n: int = 3) -> list[str]:
    """Return the closest existing tool names for a hallucinated name.

    Three-tier matching, accepting the first tier that produces hits:

    1. difflib sequence-ratio match — catches typos like ``fetch_pag`` →
       ``fetch_page``.
    2. Token substring overlap on the name itself — catches misses where
       both names share at least one underscore-separated chunk.
    3. Token overlap against each tool's *description* — catches purely
       conceptual misses like ``fetch_page`` → ``http_get`` where the names
       share nothing but the description mentions "fetch a web page". This
       is what the LLM is actually reaching for; it tends to invent a name
       that matches the capability, not our internal naming.

    Without these suggestions the LLM repeats the same hallucinated name
    every iteration and burns through the tool-round budget.
    """
    import difflib as _difflib

    all_names = registry.list_names()
    if not all_names:
        return []
    close = _difflib.get_close_matches(missing, all_names, n=max_n, cutoff=0.5)
    if close:
        return close

    missing_lower = missing.lower()
    missing_chunks = [c for c in missing_lower.split("_") if len(c) >= 3]
    if not missing_chunks:
        missing_chunks = [missing_lower] if len(missing_lower) >= 3 else []

    name_scored: list[tuple[int, str]] = []
    desc_scored: list[tuple[int, str]] = []
    for n in all_names:
        nlow = n.lower()
        name_score = 0
        for chunk in missing_chunks:
            if chunk in nlow:
                name_score += 2
        for chunk in nlow.split("_"):
            if len(chunk) >= 3 and chunk in missing_lower:
                name_score += 1
        if name_score > 0:
            name_scored.append((name_score, n))

        tool = registry.get(n)
        if not tool:
            continue
        desc = tool.description.lower()
        desc_score = sum(1 for chunk in missing_chunks if chunk in desc)
        if desc_score > 0:
            desc_scored.append((desc_score, n))

    if name_scored:
        name_scored.sort(key=lambda x: x[0], reverse=True)
        return [n for _s, n in name_scored[:max_n]]
    desc_scored.sort(key=lambda x: x[0], reverse=True)
    return [n for _s, n in desc_scored[:max_n]]


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


def _synthesize_delegation_summary(delegations: list[tuple[str, bool]]) -> str:
    """Build a short fallback line shown when a turn ends with delegate_to
    calls but no plain text from the model.

    Keeps the REPL's "empty response" guard from firing while making it
    clear to the user that work happened — without leaking the target
    agent's full response into the caller's voice.
    """
    if not delegations:
        return ""
    ok = [name for name, success in delegations if success]
    failed = [name for name, success in delegations if not success]
    parts: list[str] = []
    if ok:
        parts.append("Delegated: " + ", ".join(ok))
    if failed:
        parts.append("Failed: " + ", ".join(failed))
    return "[" + " | ".join(parts) + "]"

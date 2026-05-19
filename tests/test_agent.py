"""Tests for base agent class."""

from __future__ import annotations

import asyncio

import pytest

from weather_agents.core.agent import AgentState, Task, TaskResult
from weather_agents.core.bus import Event, EventType
from weather_agents.core.skill import Skill, SkillRegistry


class TestBaseAgent:
    def test_task_dataclass(self):
        task = Task(id="1", description="test task", assigned_to="fog")
        assert task.id == "1"
        assert task.status == "pending"
        assert task.assigned_to == "fog"

    def test_task_result_dataclass(self):
        r = TaskResult(success=True, content="done", data={"key": "val"})
        assert r.success is True
        assert r.content == "done"
        assert r.data["key"] == "val"

    def test_agent_states(self):
        assert AgentState.IDLE.value == "idle"
        assert AgentState.THINKING.value == "thinking"
        assert AgentState.ACTING.value == "acting"
        assert AgentState.ERROR.value == "error"

    @pytest.mark.asyncio
    async def test_concrete_agent_init(self, app_config, mock_llm, bus, tool_registry):
        """Verify a concrete FogAgent can init without error."""
        from weather_agents.agents.fog import FogAgent

        agent = FogAgent(config=app_config, llm=mock_llm, bus=bus, tool_registry=tool_registry)
        await agent.init()
        assert agent.name == "fog"
        assert agent.state == AgentState.IDLE
        assert agent.display_name == "雾"
        await agent.close()

    @pytest.mark.asyncio
    async def test_chat_returns_response(self, app_config, mock_llm, bus, tool_registry):
        from weather_agents.agents.fog import FogAgent

        agent = FogAgent(config=app_config, llm=mock_llm, bus=bus, tool_registry=tool_registry)
        await agent.init()
        response = await agent.chat("hello")
        assert response == "test response"
        await agent.close()

    @pytest.mark.asyncio
    async def test_execute_task(self, app_config, mock_llm, bus, tool_registry):
        from weather_agents.agents.rain import RainAgent

        agent = RainAgent(config=app_config, llm=mock_llm, bus=bus, tool_registry=tool_registry)
        await agent.init()
        task = Task(id="1", description="write code", assigned_to="rain")
        result = await agent.execute_task(task)
        assert result.success is True
        assert result.content == "test response"
        await agent.close()

    @pytest.mark.asyncio
    async def test_get_status(self, app_config, mock_llm, bus, tool_registry):
        from weather_agents.agents.fog import FogAgent

        agent = FogAgent(config=app_config, llm=mock_llm, bus=bus, tool_registry=tool_registry)
        await agent.init()
        status = agent.get_status()
        assert status["name"] == "fog"
        assert status["state"] == "idle"
        await agent.close()

    @pytest.mark.asyncio
    async def test_agent_has_system_prompt(self, app_config, mock_llm, bus, tool_registry):
        from weather_agents.agents.dew import DewAgent

        agent = DewAgent(config=app_config, llm=mock_llm, bus=bus, tool_registry=tool_registry)
        assert "露" in agent.system_prompt
        assert "守护" in agent.specialty
        await agent.close()

    def test_all_agent_classes_have_required_attrs(self):
        from weather_agents.agents.dew import DewAgent
        from weather_agents.agents.fog import FogAgent
        from weather_agents.agents.frost import FrostAgent
        from weather_agents.agents.rain import RainAgent
        from weather_agents.agents.snow import SnowAgent

        for cls in [FogAgent, RainAgent, FrostAgent, SnowAgent, DewAgent]:
            assert cls.name, f"{cls.__name__} missing name"
            assert cls.display_name, f"{cls.__name__} missing display_name"
            assert cls.emoji, f"{cls.__name__} missing emoji"
            assert cls.specialty, f"{cls.__name__} missing specialty"
            assert cls.system_prompt, f"{cls.__name__} missing system_prompt"
            assert cls.skill_names, f"{cls.__name__} missing skill_names"
            assert len(cls.skill_names) >= 3, f"{cls.__name__} should have at least 3 skills"


class TestSkillSystem:
    def test_skill_dataclass(self):
        skill = Skill(name="test", description="a test skill", system_prompt="you are a test skill")
        assert skill.name == "test"
        assert skill.description == "a test skill"
        assert skill.required_tools == []

    def test_skill_registry(self):
        reg = SkillRegistry()
        skill = Skill(name="web_research", description="research", system_prompt="test")
        reg.register(skill)
        assert reg.get("web_research") is skill
        assert reg.list_names() == ["web_research"]

    def test_skill_registry_get_multiple(self):
        reg = SkillRegistry()
        reg.register(Skill(name="a", description="skill a"))
        reg.register(Skill(name="b", description="skill b"))
        skills = reg.get_skills(["a", "c"])
        assert len(skills) == 1
        assert skills[0].name == "a"

    def test_agent_skill_names_loaded(self):
        """Verify FogAgent has correct skill_names from class attribute."""
        from weather_agents.agents.fog import FogAgent

        assert "web_research" in FogAgent.skill_names
        assert "code_analysis" in FogAgent.skill_names
        assert "document_analysis" in FogAgent.skill_names

    @pytest.mark.asyncio
    async def test_activate_skill_with_registry(self, app_config, mock_llm, bus, tool_registry):
        """Activate a skill via SkillRegistry."""
        reg = SkillRegistry()
        # Register all FogAgent's skills so _load_skills picks them up
        reg.register(
            Skill(
                name="web_research",
                description="research",
                system_prompt="你擅长调研",
                required_tools=["read_file"],
            )
        )
        reg.register(Skill(name="code_analysis", description="analysis", system_prompt="分析"))
        reg.register(Skill(name="document_analysis", description="docs", system_prompt="文档"))

        # Register the required tool
        from weather_agents.core.tool import Tool, ToolParameter

        tool_registry.register(
            Tool(
                name="read_file",
                description="read",
                parameters=[ToolParameter(name="path", type="string", description="path")],
                handler=lambda **kw: "content",
            )
        )

        from weather_agents.agents.fog import FogAgent

        agent = FogAgent(
            config=app_config,
            llm=mock_llm,
            bus=bus,
            tool_registry=tool_registry,
            skill_registry=reg,
        )
        await agent.init()

        assert agent.activate_skill("web_research") is True
        assert "web_research" in agent.get_active_skills()
        await agent.close()

    @pytest.mark.asyncio
    async def test_deactivate_skill(self, app_config, mock_llm, bus, tool_registry):
        reg = SkillRegistry()
        # Use FogAgent's actual skill names
        reg.register(
            Skill(name="web_research", description="research", system_prompt="research prompt")
        )
        reg.register(
            Skill(name="code_analysis", description="analysis", system_prompt="analysis prompt")
        )
        reg.register(
            Skill(name="document_analysis", description="docs", system_prompt="docs prompt")
        )

        from weather_agents.agents.fog import FogAgent

        agent = FogAgent(
            config=app_config,
            llm=mock_llm,
            bus=bus,
            tool_registry=tool_registry,
            skill_registry=reg,
        )
        await agent.init()

        agent.activate_skill("web_research")
        assert len(agent.get_active_skills()) == 1

        agent.deactivate_all_skills()
        assert len(agent.get_active_skills()) == 0
        await agent.close()

    @pytest.mark.asyncio
    async def test_get_available_skills(self, app_config, mock_llm, bus, tool_registry):
        reg = SkillRegistry()
        # Use FogAgent's actual skill names
        reg.register(Skill(name="web_research", description="Deep web research"))
        reg.register(Skill(name="code_analysis", description="Code analysis"))
        reg.register(Skill(name="document_analysis", description="Document analysis"))

        from weather_agents.agents.fog import FogAgent

        agent = FogAgent(
            config=app_config,
            llm=mock_llm,
            bus=bus,
            tool_registry=tool_registry,
            skill_registry=reg,
        )
        await agent.init()

        available = agent.get_available_skills()
        names = {s["name"] for s in available}
        assert "web_research" in names
        assert "code_analysis" in names
        assert "document_analysis" in names
        await agent.close()

    def test_get_status_includes_skills(self):
        reg = SkillRegistry()
        reg.register(Skill(name="demo", description="demo skill"))

        from unittest.mock import Mock

        agent = Mock()
        agent.get_status.return_value = {
            "name": "test",
            "state": "idle",
            "skills": [{"name": "demo", "description": "demo skill", "active": False}],
        }
        status = agent.get_status()
        assert "skills" in status
        assert len(status["skills"]) == 1


class TestSystemPromptLanguage:
    def test_default_language_is_zh_prompt(self, app_config, mock_llm, bus, tool_registry):
        from weather_agents.agents.fog import FogAgent

        agent = FogAgent(config=app_config, llm=mock_llm, bus=bus, tool_registry=tool_registry)
        prompt = agent._resolve_system_prompt()
        assert "雾" in prompt
        assert "Fog" not in prompt or "drifting" not in prompt.lower()

    def test_english_language_selects_en_prompt(self, app_config, mock_llm, bus, tool_registry):
        from weather_agents.agents.fog import FogAgent

        app_config.llm.language = "en"
        agent = FogAgent(config=app_config, llm=mock_llm, bus=bus, tool_registry=tool_registry)
        prompt = agent._resolve_system_prompt()
        assert "drifting" in prompt.lower()

    def test_all_agents_have_english_prompt(self):
        from weather_agents.agents.dew import DewAgent
        from weather_agents.agents.fog import FogAgent
        from weather_agents.agents.frost import FrostAgent
        from weather_agents.agents.rain import RainAgent
        from weather_agents.agents.snow import SnowAgent

        for cls in (FogAgent, RainAgent, FrostAgent, SnowAgent, DewAgent):
            assert hasattr(cls, "system_prompt_en"), f"{cls.__name__} missing system_prompt_en"
            assert len(cls.system_prompt_en) > 50, f"{cls.__name__} system_prompt_en too short"


class TestSkillHandlerInjection:
    @pytest.mark.asyncio
    async def test_skill_handler_registers_tools(self, app_config, mock_llm, bus, tool_registry):
        """Activating a skill with a handler should register custom tools."""
        from weather_agents.core.skill import Skill, SkillRegistry

        reg = SkillRegistry()
        # Simulate a skill with handler tool injection
        from weather_agents.core.tool import Tool, ToolParameter

        def _inject(agent, registry):
            t = Tool(
                name="custom_tool",
                description="A handler-injected tool",
                parameters=[ToolParameter(name="arg", type="string", description="an argument")],
                handler=lambda **kw: "custom result",
            )
            registry.register(t)
            return [t]

        reg.register(
            Skill(
                name="handler_skill",
                description="Skill with handler",
                system_prompt="handler skill prompt",
                required_tools=["read_file"],
                handler=_inject,
            )
        )

        from weather_agents.agents.fog import FogAgent

        agent = FogAgent(
            config=app_config,
            llm=mock_llm,
            bus=bus,
            tool_registry=tool_registry,
            skill_registry=reg,
        )
        await agent.init()

        assert "custom_tool" not in tool_registry.list_names()

        ok = agent.activate_skill("handler_skill")
        assert ok is True
        assert "custom_tool" in tool_registry.list_names()

        # Deactivate should remove the injected tool
        agent.deactivate_skill("handler_skill")
        assert "custom_tool" not in tool_registry.list_names()
        await agent.close()

    @pytest.mark.asyncio
    async def test_deactivate_all_cleans_handler_tools(
        self, app_config, mock_llm, bus, tool_registry
    ):
        """Deactivate all skills should remove all handler-injected tools."""
        from weather_agents.core.skill import Skill, SkillRegistry
        from weather_agents.core.tool import Tool

        reg = SkillRegistry()

        def _inject_a(agent, registry):
            t = Tool(name="tool_a", description="A", handler=lambda **kw: "a")
            registry.register(t)
            return [t]

        def _inject_b(agent, registry):
            t = Tool(name="tool_b", description="B", handler=lambda **kw: "b")
            registry.register(t)
            return [t]

        reg.register(Skill(name="skill_a", description="A", handler=_inject_a))
        reg.register(Skill(name="skill_b", description="B", handler=_inject_b))

        from weather_agents.agents.fog import FogAgent

        agent = FogAgent(
            config=app_config,
            llm=mock_llm,
            bus=bus,
            tool_registry=tool_registry,
            skill_registry=reg,
        )
        await agent.init()

        agent.activate_skill("skill_a")
        agent.activate_skill("skill_b")
        assert "tool_a" in tool_registry.list_names()
        assert "tool_b" in tool_registry.list_names()

        agent.deactivate_all_skills()
        assert "tool_a" not in tool_registry.list_names()
        assert "tool_b" not in tool_registry.list_names()
        await agent.close()


class TestSkillMarkdownLoading:
    def test_from_markdown_valid(self, tmp_path):
        from weather_agents.core.skill import Skill

        md_file = tmp_path / "test_skill.md"
        md_file.write_text("""---
name: md_skill
description: A skill loaded from markdown
tools:
  - read_file
  - web_search
---

## Skill: MD Skill

This is the system prompt body.
It can have multiple lines.
""")
        skill = Skill.from_markdown(md_file)
        assert skill is not None
        assert skill.name == "md_skill"
        assert skill.description == "A skill loaded from markdown"
        assert skill.required_tools == ["read_file", "web_search"]
        assert "## Skill: MD Skill" in skill.system_prompt
        assert "system prompt body" in skill.system_prompt

    def test_from_markdown_no_frontmatter(self, tmp_path):
        from weather_agents.core.skill import Skill

        md_file = tmp_path / "no_fm.md"
        md_file.write_text("Just some text without frontmatter.")
        skill = Skill.from_markdown(md_file)
        assert skill is None

    def test_load_skills_from_directory(self, tmp_path):
        from weather_agents.core.skill import SkillRegistry

        (tmp_path / "skill_a.md").write_text("""---
name: skill_a
description: First skill
tools:
  - tool_a
---
Body A
""")
        (tmp_path / "skill_b.md").write_text("""---
name: skill_b
description: Second skill
---
Body B
""")
        (tmp_path / "_private.md").write_text("""---
name: private
description: Should be skipped
---
Body
""")

        reg = SkillRegistry()
        loaded = reg.load_skills_from_directory(tmp_path)
        assert len(loaded) == 2
        assert "skill_a" in reg.list_names()
        assert "skill_b" in reg.list_names()
        assert "private" not in reg.list_names()

    def test_load_skills_from_nonexistent_directory(self):
        from weather_agents.core.skill import SkillRegistry

        reg = SkillRegistry()
        loaded = reg.load_skills_from_directory("/nonexistent/path/12345")
        assert loaded == []

    def test_priority_skills_have_handlers(self):
        """Verify the 3 priority skills have handler functions."""
        from weather_agents.skills.code_reviewer import create_skill as _cr
        from weather_agents.skills.security_auditor import create_skill as _sa
        from weather_agents.skills.web_research import create_skill as _wr

        for factory in (_cr, _sa, _wr):
            skill = factory()
            assert skill.handler is not None, f"{skill.name} missing handler"
            assert callable(skill.handler)


class TestParseToolArgs:
    def test_valid_json(self):
        from weather_agents.core.agent import _parse_tool_args

        assert _parse_tool_args('{"query": "news"}') == {"query": "news"}

    def test_single_quotes(self):
        from weather_agents.core.agent import _parse_tool_args

        result = _parse_tool_args("{'query': 'news'}")
        assert result == {"query": "news"}

    def test_trailing_comma(self):
        from weather_agents.core.agent import _parse_tool_args

        result = _parse_tool_args('{"query": "news",}')
        assert result == {"query": "news"}

    def test_unquoted_keys(self):
        from weather_agents.core.agent import _parse_tool_args

        result = _parse_tool_args('{query: "news"}')
        assert result == {"query": "news"}

    def test_mixed_issues(self):
        from weather_agents.core.agent import _parse_tool_args

        result = _parse_tool_args("{'query': 'news', 'count': 5,}")
        assert result == {"query": "news", "count": 5}

    def test_empty_string(self):
        from weather_agents.core.agent import _parse_tool_args

        assert _parse_tool_args("") is None

    def test_whitespace_only(self):
        from weather_agents.core.agent import _parse_tool_args

        assert _parse_tool_args("   ") is None

    def test_garbage_returns_none(self):
        from weather_agents.core.agent import _parse_tool_args

        assert _parse_tool_args("not even close") is None


class TestParseToolArgsExtended:
    def test_markdown_code_fence(self):
        from weather_agents.core.agent import _parse_tool_args

        result = _parse_tool_args('```json\n{"query": "news"}\n```')
        assert result == {"query": "news"}

    def test_markdown_fence_inline(self):
        from weather_agents.core.agent import _parse_tool_args

        result = _parse_tool_args('```{"query": "news"}```')
        assert result == {"query": "news"}

    def test_python_none(self):
        from weather_agents.core.agent import _parse_tool_args

        result = _parse_tool_args('{"query": None}')
        assert result == {"query": None}

    def test_python_bool(self):
        from weather_agents.core.agent import _parse_tool_args

        assert _parse_tool_args('{"active": True, "done": False}') == {
            "active": True,
            "done": False,
        }

    def test_backtick_quotes(self):
        from weather_agents.core.agent import _parse_tool_args

        result = _parse_tool_args("{`query`: `weather`}")
        assert result == {"query": "weather"}

    def test_unquoted_string_value(self):
        from weather_agents.core.agent import _parse_tool_args

        result = _parse_tool_args("{query: hello world}")
        assert result == {"query": "hello world"}

    def test_key_equals_value_format(self):
        from weather_agents.core.agent import _parse_tool_args

        result = _parse_tool_args('query="news", num_results=5')
        assert result == {"query": "news", "num_results": 5}

    def test_key_equals_value_single_quotes(self):
        from weather_agents.core.agent import _parse_tool_args

        result = _parse_tool_args("query='hello world'")
        assert result == {"query": "hello world"}

    def test_trailing_text_after_object(self):
        from weather_agents.core.agent import _parse_tool_args

        result = _parse_tool_args('{"query": "news"} some trailing text')
        assert result == {"query": "news"}

    def text_before_json(self):
        from weather_agents.core.agent import _parse_tool_args

        result = _parse_tool_args('Here is the result: {"query": "news"}')
        assert result == {"query": "news"}

    def test_missing_closing_brace(self):
        from weather_agents.core.agent import _parse_tool_args

        result = _parse_tool_args('{"query": "news"')
        assert result == {"query": "news"}

    def test_extra_closing_brace(self):
        from weather_agents.core.agent import _parse_tool_args

        result = _parse_tool_args('{"query": "news"}}')
        assert result == {"query": "news"}

    def test_mixed_all_issues(self):
        from weather_agents.core.agent import _parse_tool_args

        result = _parse_tool_args("```\n{`query`: None, 'count': 5,}\n``` extra")
        assert result == {"query": None, "count": 5}

    def test_nested_object(self):
        from weather_agents.core.agent import _parse_tool_args

        result = _parse_tool_args('{"outer": {"inner": "value"}}')
        assert result == {"outer": {"inner": "value"}}

    def test_empty_object(self):
        from weather_agents.core.agent import _parse_tool_args

        assert _parse_tool_args("{}") == {}

    def test_null_value(self):
        from weather_agents.core.agent import _parse_tool_args

        result = _parse_tool_args('{"key": null}')
        assert result == {"key": None}

    def test_multiline_json(self):
        from weather_agents.core.agent import _parse_tool_args

        raw = '{\n  "query": "news",\n  "count": 5\n}'
        assert _parse_tool_args(raw) == {"query": "news", "count": 5}

    def test_path_with_slashes(self):
        from weather_agents.core.agent import _parse_tool_args

        result = _parse_tool_args('{"path": "/home/user/file.txt"}')
        assert result == {"path": "/home/user/file.txt"}

    def test_key_equals_none(self):
        from weather_agents.core.agent import _parse_tool_args

        result = _parse_tool_args("query=None")
        assert result == {"query": None}

    def test_unquoted_value_with_special_chars(self):
        from weather_agents.core.agent import _parse_tool_args

        result = _parse_tool_args("{path: ./src/main.py}")
        assert result == {"path": "./src/main.py"}


class TestPopLastUserMessage:
    @pytest.mark.asyncio
    async def test_pop_removes_last_user_message(self, app_config, mock_llm, bus, tool_registry):
        from weather_agents.agents.fog import FogAgent

        agent = FogAgent(config=app_config, llm=mock_llm, bus=bus, tool_registry=tool_registry)
        await agent.init()

        # Clear non-system messages to avoid interference from persisted state
        agent.memory.short_term = [m for m in agent.memory.short_term if m.role == "system"]
        user_before = sum(1 for m in agent.memory.short_term if m.role == "user")
        agent.memory.add_message("user", "new-msg-to-pop")
        agent.memory.add_message("assistant", "reply")
        agent._pop_last_user_message()

        user_after = sum(1 for m in agent.memory.short_term if m.role == "user")
        # Should have the same count as before (added one, popped one)
        assert user_after == user_before

        await agent.close()

    @pytest.mark.asyncio
    async def test_pop_no_user_does_not_crash(self, app_config, mock_llm, bus, tool_registry):
        from weather_agents.agents.fog import FogAgent

        agent = FogAgent(config=app_config, llm=mock_llm, bus=bus, tool_registry=tool_registry)
        # Don't init — short_term is empty
        assert len(agent.memory.short_term) == 0
        agent._pop_last_user_message()  # should not crash
        assert len(agent.memory.short_term) == 0
        await agent.close()


class TestAgentCommunication:
    """Tests for inter-agent request/response (request_help → awaitable)."""

    @pytest.mark.asyncio
    async def test_request_help_returns_response(self, app_config, mock_llm, bus, tool_registry):
        """Agent A requests help from Agent B → gets response content."""
        from weather_agents.agents.fog import FogAgent
        from weather_agents.agents.rain import RainAgent

        agent_a = FogAgent(config=app_config, llm=mock_llm, bus=bus, tool_registry=tool_registry)
        agent_b = RainAgent(config=app_config, llm=mock_llm, bus=bus, tool_registry=tool_registry)
        await agent_a.init()
        await agent_b.init()

        result = await agent_a.request_help("rain", "generate some code")
        assert result == "test response"

        await agent_a.close()
        await agent_b.close()

    @pytest.mark.asyncio
    async def test_request_help_publishes_agent_request_event(
        self, app_config, mock_llm, bus, tool_registry
    ):
        """Calling request_help should leave an AGENT_REQUEST in bus history."""
        from weather_agents.agents.fog import FogAgent
        from weather_agents.agents.rain import RainAgent

        agent_a = FogAgent(config=app_config, llm=mock_llm, bus=bus, tool_registry=tool_registry)
        agent_b = RainAgent(config=app_config, llm=mock_llm, bus=bus, tool_registry=tool_registry)
        await agent_a.init()
        await agent_b.init()

        await agent_a.request_help("rain", "write a test")

        events = bus.get_history(event_type=EventType.AGENT_REQUEST, limit=5)
        assert len(events) >= 1
        assert events[-1].source == "fog"
        assert events[-1].target == "rain"

        await agent_a.close()
        await agent_b.close()

    @pytest.mark.asyncio
    async def test_handle_response_resolves_future(self, app_config, mock_llm, bus, tool_registry):
        """_handle_response should resolve the matching pending future."""
        from weather_agents.agents.fog import FogAgent

        agent = FogAgent(config=app_config, llm=mock_llm, bus=bus, tool_registry=tool_registry)

        correlation_id = "test-corr-123"
        future = asyncio.get_running_loop().create_future()
        agent._pending_requests[correlation_id] = future

        event = Event(
            type=EventType.AGENT_RESPONSE,
            source="rain",
            target="fog",
            data={
                "correlation_id": correlation_id,
                "content": "response data",
                "success": True,
            },
        )
        agent._handle_response(event)
        assert future.done()
        assert future.result() == "response data"

        await agent.close()

    @pytest.mark.asyncio
    async def test_handle_response_unknown_id_does_not_crash(
        self, app_config, mock_llm, bus, tool_registry
    ):
        """_handle_response with a non-matching correlation_id is a no-op."""
        from weather_agents.agents.fog import FogAgent

        agent = FogAgent(config=app_config, llm=mock_llm, bus=bus, tool_registry=tool_registry)

        event = Event(
            type=EventType.AGENT_RESPONSE,
            source="rain",
            target="fog",
            data={"correlation_id": "nonexistent", "content": "data"},
        )
        agent._handle_response(event)  # should not crash

        await agent.close()

    @pytest.mark.asyncio
    async def test_handle_response_empty_correlation_id(
        self, app_config, mock_llm, bus, tool_registry
    ):
        """_handle_response with empty correlation_id is a no-op."""
        from weather_agents.agents.fog import FogAgent

        agent = FogAgent(config=app_config, llm=mock_llm, bus=bus, tool_registry=tool_registry)

        event = Event(
            type=EventType.AGENT_RESPONSE,
            source="rain",
            target="fog",
            data={"correlation_id": "", "content": "data"},
        )
        agent._handle_response(event)  # should not crash

        await agent.close()

    @pytest.mark.asyncio
    async def test_request_help_to_self(self, app_config, mock_llm, bus, tool_registry):
        """An agent should be able to request help from itself."""
        from weather_agents.agents.fog import FogAgent

        agent = FogAgent(config=app_config, llm=mock_llm, bus=bus, tool_registry=tool_registry)
        await agent.init()

        result = await agent.request_help("fog", "self-help task")
        assert result == "test response"

        await agent.close()


class TestApprovalGate:
    @pytest.mark.asyncio
    async def test_auto_approve_by_default(self, app_config, mock_llm, bus, tool_registry):
        """Default approval_mode='auto' should approve dangerous tools."""
        from weather_agents.agents.fog import FogAgent

        agent = FogAgent(config=app_config, llm=mock_llm, bus=bus, tool_registry=tool_registry)
        assert await agent._check_tool_approval("write_file", {"path": "/tmp/test"}) is True
        await agent.close()

    @pytest.mark.asyncio
    async def test_strict_mode_denies(self, app_config, mock_llm, bus, tool_registry):
        """approval_mode='strict' should deny all dangerous tools."""
        app_config.cli.approval_mode = "strict"
        from weather_agents.agents.fog import FogAgent

        agent = FogAgent(config=app_config, llm=mock_llm, bus=bus, tool_registry=tool_registry)
        assert await agent._check_tool_approval("write_file", {"path": "/tmp/test"}) is False
        await agent.close()

    @pytest.mark.asyncio
    async def test_interactive_mode_calls_callback(self, app_config, mock_llm, bus, tool_registry):
        """approval_mode='interactive' should delegate to the callback."""
        app_config.cli.approval_mode = "interactive"

        async def _mock_approval(name: str, args: dict) -> bool:
            return name == "safe_tool"

        from weather_agents.agents.fog import FogAgent

        agent = FogAgent(config=app_config, llm=mock_llm, bus=bus, tool_registry=tool_registry)
        agent.approval_callback = _mock_approval
        assert await agent._check_tool_approval("safe_tool", {}) is True
        assert await agent._check_tool_approval("dangerous_tool", {}) is False
        await agent.close()

    @pytest.mark.asyncio
    async def test_interactive_no_callback_denies(self, app_config, mock_llm, bus, tool_registry):
        """interactive mode with no callback set should deny."""
        app_config.cli.approval_mode = "interactive"
        from weather_agents.agents.fog import FogAgent

        agent = FogAgent(config=app_config, llm=mock_llm, bus=bus, tool_registry=tool_registry)
        assert agent.approval_callback is None
        assert await agent._check_tool_approval("write_file", {}) is False
        await agent.close()


class TestBackgroundTaskTracking:
    @pytest.mark.asyncio
    async def test_agent_request_task_is_tracked(self, app_config, mock_llm, bus, tool_registry):
        """AGENT_REQUEST creates a tracked background task."""
        from weather_agents.agents.fog import FogAgent
        from weather_agents.core.bus import Event, EventType

        agent = FogAgent(config=app_config, llm=mock_llm, bus=bus, tool_registry=tool_registry)
        await agent.init()

        event = Event(
            type=EventType.AGENT_REQUEST,
            source="rain",
            target="fog",
            data={"correlation_id": "bg-001", "description": "test", "source": "rain"},
        )
        await agent._handle_event(event)

        assert len(agent._bg_tasks) >= 1
        # Wait for background task to complete
        for t in list(agent._bg_tasks):
            await t
        assert len(agent._bg_tasks) == 0
        await agent.close()

    @pytest.mark.asyncio
    async def test_background_task_exception_is_logged(
        self, app_config, mock_llm, bus, tool_registry
    ):
        """Background task that raises does not take down the agent."""
        import asyncio as _asyncio

        from weather_agents.agents.fog import FogAgent

        agent = FogAgent(config=app_config, llm=mock_llm, bus=bus, tool_registry=tool_registry)
        await agent.init()

        async def _crash():
            raise RuntimeError("simulated crash")

        t = _asyncio.create_task(_crash())
        agent._bg_tasks.add(t)
        t.add_done_callback(agent._bg_tasks.discard)

        # The exception should be captured by the done callback
        with pytest.raises(RuntimeError, match="simulated crash"):
            await t

        assert len(agent._bg_tasks) == 0
        await agent.close()


class TestSkillAutoActivation:
    """Skills with `triggers` should auto-activate on matching user messages."""

    def test_trigger_activates_skill(self, app_config, bus, mock_llm):
        from weather_agents.agents.fog import FogAgent
        from weather_agents.core.skill import Skill, SkillRegistry
        from weather_agents.core.tool import ToolRegistry

        reg = SkillRegistry()
        reg.register(
            Skill(
                name="coder",
                description="Code review skill",
                system_prompt="Focus on quality.",
                triggers=["code review", "review the code"],
            )
        )
        agent = FogAgent(
            config=app_config,
            llm=mock_llm,
            bus=bus,
            tool_registry=ToolRegistry(),
            skill_registry=reg,
        )
        activated = agent._auto_activate_skills("please code review this PR")
        assert "coder" in activated
        assert "coder" in agent._active_skills

    def test_already_active_skill_not_re_activated(self, app_config, bus, mock_llm):
        from weather_agents.agents.fog import FogAgent
        from weather_agents.core.skill import Skill, SkillRegistry
        from weather_agents.core.tool import ToolRegistry

        reg = SkillRegistry()
        reg.register(
            Skill(name="s1", description="x", triggers=["build"]),
        )
        agent = FogAgent(
            config=app_config,
            llm=mock_llm,
            bus=bus,
            tool_registry=ToolRegistry(),
            skill_registry=reg,
        )
        agent.activate_skill("s1")
        # Already active — auto-activation must skip it
        activated = agent._auto_activate_skills("let's build it")
        assert activated == []

    def test_no_trigger_no_activation(self, app_config, bus, mock_llm):
        from weather_agents.agents.fog import FogAgent
        from weather_agents.core.skill import Skill, SkillRegistry
        from weather_agents.core.tool import ToolRegistry

        reg = SkillRegistry()
        reg.register(Skill(name="nope", description="x", triggers=["xyz"]))
        agent = FogAgent(
            config=app_config,
            llm=mock_llm,
            bus=bus,
            tool_registry=ToolRegistry(),
            skill_registry=reg,
        )
        assert agent._auto_activate_skills("hello there") == []


class TestTurnLockSerialization:
    """Concurrent chat/execute_task on the same agent must serialize,
    not interleave — otherwise short_term gets corrupted with mixed turns."""

    @pytest.mark.asyncio
    async def test_concurrent_chat_serializes(self, app_config, bus, mock_llm, tool_registry):
        from weather_agents.agents.fog import FogAgent

        agent = FogAgent(config=app_config, llm=mock_llm, bus=bus, tool_registry=tool_registry)
        await agent.init()
        # Sentinel to detect interleaving: each chat appends one user + one
        # assistant message. If serialized, short_term grows in clean pairs.
        import asyncio as _asyncio

        await _asyncio.gather(
            agent.chat("msg one"),
            agent.chat("msg two"),
            agent.chat("msg three"),
        )
        # 3 user + 3 assistant messages (plus any system). For each user
        # message the next non-system message must be an assistant — never
        # another user.
        non_sys = [m for m in agent.memory.short_term if m.role != "system"]
        for i in range(0, len(non_sys) - 1, 2):
            assert non_sys[i].role == "user", f"position {i} expected user, got {non_sys[i].role}"
            assert non_sys[i + 1].role == "assistant", (
                f"position {i + 1} expected assistant, got {non_sys[i + 1].role}"
            )
        await agent.close()

    @pytest.mark.asyncio
    async def test_close_waits_for_pending_extracts(self, app_config, bus, mock_llm, tool_registry):
        import asyncio as _asyncio

        from weather_agents.agents.fog import FogAgent

        agent = FogAgent(config=app_config, llm=mock_llm, bus=bus, tool_registry=tool_registry)
        await agent.init()

        # Inject a slow-completing fake extract task so we can verify close
        # actually waits for it.
        completed = {"ok": False}

        async def _slow_task():
            await _asyncio.sleep(0.1)
            completed["ok"] = True

        task = _asyncio.create_task(_slow_task())
        agent._pending_extracts.add(task)
        task.add_done_callback(agent._pending_extracts.discard)

        await agent.close()
        assert completed["ok"], "close() returned before _pending_extracts drained"


class TestArgsParseErrorFormatter:
    """The 'Invalid JSON' tool-result was misleading when the LLM hit
    max_tokens mid-content. _format_args_parse_error should detect that
    case and tell the model to chunk instead of parroting 'invalid JSON'."""

    def test_truncated_unclosed_string_detected(self):
        from weather_agents.core.agent import _format_args_parse_error

        # Realistic write_file truncation: content string never closes,
        # no trailing brace.
        raw = '{"path": "weather.html", "content": "<!DOCTYPE html><html><body><h1>'
        msg = _format_args_parse_error("write_file", raw)
        assert "truncated" in msg.lower()
        assert "max_tokens" in msg
        assert "chunk" in msg.lower() or "split" in msg.lower()

    def test_truncated_unclosed_object_detected(self):
        from weather_agents.core.agent import _format_args_parse_error

        # No string opened, but missing closing brace
        raw = '{"path": "x", "content": "ok"'
        msg = _format_args_parse_error("write_file", raw)
        assert "truncated" in msg.lower()

    def test_genuinely_malformed_json_not_called_truncated(self):
        from weather_agents.core.agent import _format_args_parse_error

        # Properly closed, just malformed in the middle
        raw = '{"path": "x", "content": }'
        msg = _format_args_parse_error("write_file", raw)
        assert "truncated" not in msg.lower()
        assert "invalid JSON" in msg


class TestStuckLoopDetection:
    """When tool calls keep failing, an LLM-readable recovery hint must
    appear in memory so the model stops grinding through dead-ends."""

    def test_looks_like_failed_tool_result_recognizes_common_signals(self):
        from weather_agents.core.agent import _looks_like_failed_tool_result

        assert _looks_like_failed_tool_result("No results found for 'foo'")
        assert _looks_like_failed_tool_result("Status: 403 Forbidden")
        assert _looks_like_failed_tool_result("Status: 503 Service Unavailable")
        assert _looks_like_failed_tool_result("Error: request timed out")
        assert _looks_like_failed_tool_result("[CircuitBreakerOpen] Tool 'x' is unavailable")
        # Empty / whitespace is treated as failure too
        assert _looks_like_failed_tool_result("")

    def test_real_results_not_misclassified(self):
        from weather_agents.core.agent import _looks_like_failed_tool_result

        assert not _looks_like_failed_tool_result(
            "Search results for: weather\n1. cnn.com\n2. bbc.com"
        )
        assert not _looks_like_failed_tool_result(
            "<!DOCTYPE html><html><body>real content</body></html>"
        )
        # "Status: 200 OK" looks similar to "Status: 4xx" but should NOT match
        assert not _looks_like_failed_tool_result("Status: 200 OK\nGot data")


class TestArtifactExtraction:
    """The agent often replies with "已完成" while having actually written
    files via tool calls. The orchestrator scans tool-call history and
    appends the file paths so the verifier / user can find them."""

    def test_extracts_write_file_paths(self):
        from weather_agents.core.agent import _extract_file_paths_from_messages
        from weather_agents.core.memory import Message

        msgs = [
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    {
                        "id": "c1",
                        "function": {
                            "name": "write_file",
                            "arguments": '{"path": "E:/out/a.md", "content": "..."}',
                        },
                    }
                ],
            ),
            Message(
                role="tool",
                content="Successfully wrote to E:/out/a.md",
                tool_call_id="c1",
            ),
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    {
                        "id": "c2",
                        "function": {
                            "name": "edit_file",
                            "arguments": '{"path": "E:/out/b.md", "old_text": "x", "new_text": "y"}',
                        },
                    }
                ],
            ),
            Message(
                role="tool",
                content="Successfully edited E:/out/b.md",
                tool_call_id="c2",
            ),
        ]
        paths = _extract_file_paths_from_messages(msgs)
        assert paths == ["E:/out/a.md", "E:/out/b.md"]

    def test_skips_failed_writes(self):
        from weather_agents.core.agent import _extract_file_paths_from_messages
        from weather_agents.core.memory import Message

        msgs = [
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    {
                        "id": "c1",
                        "function": {
                            "name": "write_file",
                            "arguments": '{"path": "/protected/x", "content": "..."}',
                        },
                    }
                ],
            ),
            Message(
                role="tool",
                content="Error: refusing to write to protected path: /protected/x",
                tool_call_id="c1",
            ),
        ]
        assert _extract_file_paths_from_messages(msgs) == []

    def test_enrich_appends_paths_when_missing(self):
        from weather_agents.core.agent import _enrich_response_with_artifacts

        out = _enrich_response_with_artifacts("已完成。", ["/a.md", "/b.md"])
        assert "/a.md" in out
        assert "/b.md" in out
        assert "Artifacts produced" in out
        # New format: markdown blockquote with backticked paths so Rich
        # renders the section with a left-bar + code highlighting.
        assert "> **Artifacts produced**" in out
        assert "`/a.md`" in out
        assert "`/b.md`" in out

    def test_enrich_noop_when_no_files(self):
        from weather_agents.core.agent import _enrich_response_with_artifacts

        assert _enrich_response_with_artifacts("real content", []) == "real content"

    def test_enrich_marks_already_cited(self):
        from weather_agents.core.agent import _enrich_response_with_artifacts

        body = "Saved everything to /a.md. Other notes are in /b.md."
        out = _enrich_response_with_artifacts(body, ["/a.md", "/b.md"])
        # Both already in body -> nothing appended
        assert out == body


class TestTextSimilarity:
    """The narration-loop detector uses _text_similarity to catch the
    agent rephrasing the same '现在要做 X' sentence across rounds."""

    def test_near_duplicate_zh_paraphrase_matches(self):
        from weather_agents.core.agent import _text_similarity

        a = "现在把 files/ 中的脚本、文档、数据分类移出。"
        b = "现在把 files/ 中的脚本、文档、数据分门别类移出去。"
        assert _text_similarity(a, b) >= 0.7

    def test_distinct_actions_dont_match(self):
        from weather_agents.core.agent import _text_similarity

        assert _text_similarity("清理 temp/ 目录。", "现在审查代码安全性。") < 0.5

    def test_short_strings_return_zero(self):
        from weather_agents.core.agent import _text_similarity

        # Short brief responses must not trigger loop detection
        assert _text_similarity("好", "好的") == 0.0
        assert _text_similarity("OK!", "ok.") == 0.0

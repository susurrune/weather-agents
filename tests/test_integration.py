"""Integration tests: agent init -> chat -> tool call -> response chain."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from weather_agents.core.agent import AgentState, Task
from weather_agents.core.config import AppConfig
from weather_agents.core.tool import Tool


class TestAgentInitChatToolResponse:
    """End-to-end flow: agent init, chat with tool calls, response handling."""

    @pytest.mark.asyncio
    async def test_full_chat_with_tool_call(self, tool_registry, mock_llm, bus):
        """Agent receives a message, calls a tool, and returns a response."""
        from weather_agents.agents.fog import FogAgent

        # Add a real tool
        tool_registry.register(
            Tool(
                name="read_file",
                description="Read a file",
                parameters=[],
                handler=AsyncMock(return_value="file content here"),
            )
        )

        agent = FogAgent(config=AppConfig(), llm=mock_llm, bus=bus, tool_registry=tool_registry)
        await agent.init()

        # Mock LLM: first call returns tool call, second call returns final response
        mock_llm.complete.side_effect = [
            Mock(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path": "/tmp/test.txt"}'},
                    }
                ],
                model="test-model",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
                reasoning_content=None,
            ),
            Mock(
                content="The file contains: file content here",
                tool_calls=[],
                model="test-model",
                usage={"prompt_tokens": 15, "completion_tokens": 5},
                reasoning_content=None,
            ),
        ]

        response = await agent.chat("read the file")
        assert "file content here" in response
        assert mock_llm.complete.call_count == 2
        await agent.close()

    @pytest.mark.asyncio
    async def test_execute_task_flow(self, tool_registry, mock_llm, bus):
        """Execute a task with tool calls from Rain agent."""
        from weather_agents.agents.rain import RainAgent

        tool_registry.register(
            Tool(
                name="write_file",
                description="Write content to file",
                parameters=[],
                handler=AsyncMock(return_value="Successfully wrote to /tmp/out.py"),
            )
        )

        agent = RainAgent(config=AppConfig(), llm=mock_llm, bus=bus, tool_registry=tool_registry)
        await agent.init()

        mock_llm.complete.side_effect = [
            Mock(
                content="",
                tool_calls=[
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": '{"path": "/tmp/out.py", "content": "print(1)"}',
                        },
                    }
                ],
                model="test-model",
                usage={},
                reasoning_content=None,
            ),
            Mock(
                content="File written successfully",
                tool_calls=[],
                model="test-model",
                usage={},
                reasoning_content=None,
            ),
        ]

        task = Task(id="task_1", description="create a Python script", assigned_to="rain")
        result = await agent.execute_task(task)
        assert result.success is True
        assert "written" in result.content.lower()
        await agent.close()

    @pytest.mark.asyncio
    async def test_skill_activation_flow(self, tool_registry, mock_llm, bus):
        """Activate a skill, verify it modifies behavior."""
        from weather_agents.agents.frost import FrostAgent
        from weather_agents.core.skill import Skill, SkillRegistry

        # Register a skill so activation works
        skill_reg = SkillRegistry()
        skill_reg.register(
            Skill(
                name="code_reviewer",
                description="Review code for issues",
                system_prompt="You are a code reviewer",
                required_tools=[],
            )
        )

        agent = FrostAgent(
            config=AppConfig(),
            llm=mock_llm,
            bus=bus,
            tool_registry=tool_registry,
            skill_registry=skill_reg,
        )
        await agent.init()

        # Initially no active skills
        assert len(agent._active_skills) == 0

        # Activate skill
        result = agent.activate_skill("code_reviewer")
        assert result is True
        assert "code_reviewer" in agent._active_skills

        # Deactivate
        agent.deactivate_all_skills()
        assert len(agent._active_skills) == 0

        await agent.close()

    @pytest.mark.asyncio
    async def test_snow_orchestrate_flow(self, tool_registry, mock_llm, bus):
        """Snow agent plans and returns tasks from a goal description."""
        from weather_agents.agents.snow import SnowAgent

        agent = SnowAgent(config=AppConfig(), llm=mock_llm, bus=bus, tool_registry=tool_registry)
        await agent.init()

        mock_llm.complete.return_value = Mock(
            content='```json\n{"goal": "test", "steps": [{"id": "1", "description": "analyze", "agent": "fog"}, {"id": "2", "description": "write", "agent": "rain"}]}\n```',
            tool_calls=[],
            model="test-model",
            usage={},
            reasoning_content=None,
        )

        tasks = await agent.orchestrate("build a CLI tool")
        assert len(tasks) == 2
        assert tasks[0].assigned_to == "fog"
        assert tasks[1].assigned_to == "rain"
        await agent.close()

    @pytest.mark.asyncio
    async def test_dew_execute_flow(self, tool_registry, mock_llm, bus):
        """Dew agent executes shell commands via tool calls."""
        from weather_agents.agents.dew import DewAgent

        tool_registry.register(
            Tool(
                name="shell_exec",
                description="Execute a shell command",
                parameters=[],
                handler=AsyncMock(return_value="output: README.md\n"),
            )
        )

        agent = DewAgent(config=AppConfig(), llm=mock_llm, bus=bus, tool_registry=tool_registry)
        await agent.init()

        mock_llm.complete.return_value = Mock(
            content="The directory listing shows README.md",
            tool_calls=[],
            model="test-model",
            usage={},
            reasoning_content=None,
        )

        response = await agent.chat("list files")
        assert "README.md" in response
        await agent.close()

    @pytest.mark.asyncio
    async def test_memory_persistence_flow(self, tool_registry, mock_llm, bus):
        """Messages persist in memory across chat turns."""
        from weather_agents.agents.fog import FogAgent

        agent = FogAgent(config=AppConfig(), llm=mock_llm, bus=bus, tool_registry=tool_registry)
        await agent.init()

        mock_llm.complete.return_value = Mock(
            content="first response",
            tool_calls=[],
            model="test-model",
            usage={},
            reasoning_content=None,
        )

        await agent.chat("hello")
        msgs = agent.memory.get_messages()
        # Should have system, user, assistant
        roles = [m["role"] for m in msgs]
        assert "system" in roles
        assert "user" in roles
        assert "assistant" in roles
        await agent.close()

    @pytest.mark.asyncio
    async def test_all_five_agents_init_and_chat(self, tool_registry, mock_llm, bus):
        """All agent classes can init and respond to a simple message."""
        from weather_agents.agents.dew import DewAgent
        from weather_agents.agents.fair import FairAgent
        from weather_agents.agents.fog import FogAgent
        from weather_agents.agents.frost import FrostAgent
        from weather_agents.agents.rain import RainAgent
        from weather_agents.agents.snow import SnowAgent

        mock_llm.complete.return_value = Mock(
            content="test response",
            tool_calls=[],
            model="test-model",
            usage={},
            reasoning_content=None,
        )

        for agent_cls in (FogAgent, RainAgent, FrostAgent, SnowAgent, DewAgent, FairAgent):
            agent = agent_cls(
                config=AppConfig(), llm=mock_llm, bus=bus, tool_registry=tool_registry
            )
            await agent.init()
            assert agent.state == AgentState.IDLE
            response = await agent.chat("hello")
            assert response == "test response"
            await agent.close()

    @pytest.mark.asyncio
    async def test_state_transitions(self, tool_registry, mock_llm, bus):
        """Agent state transitions: IDLE -> THINKING -> IDLE during chat."""
        from weather_agents.agents.fog import FogAgent

        agent = FogAgent(config=AppConfig(), llm=mock_llm, bus=bus, tool_registry=tool_registry)
        await agent.init()
        assert agent.state == AgentState.IDLE

        mock_llm.complete.return_value = Mock(
            content="ok",
            tool_calls=[],
            model="test",
            usage={},
            reasoning_content=None,
        )

        await agent.chat("test")
        assert agent.state == AgentState.IDLE
        await agent.close()

    @pytest.mark.asyncio
    async def test_task_status_update(self, tool_registry, mock_llm, bus):
        """Task transitions through statuses correctly."""
        task = Task(id="1", description="test", assigned_to="rain")
        assert task.status == "pending"
        assert task.result is None

        from weather_agents.agents.rain import RainAgent

        mock_llm.complete.return_value = Mock(
            content="completed",
            tool_calls=[],
            model="test",
            usage={},
            reasoning_content=None,
        )

        agent = RainAgent(config=AppConfig(), llm=mock_llm, bus=bus, tool_registry=tool_registry)
        await agent.init()

        result = await agent.execute_task(task)
        assert result.success is True
        assert task.status == "completed"
        assert task.result == "completed"
        await agent.close()

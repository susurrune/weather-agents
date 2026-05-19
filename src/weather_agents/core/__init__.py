"""Core framework for Weather Agents.

Re-exports leaf modules only. ``factory`` is intentionally NOT exposed here
because it imports ``weather_agents.tools.builtin`` — which itself needs
``weather_agents.core.tool``. Loading ``factory`` during ``core/__init__.py``
would create a circular import when any third-party code imports directly
from ``weather_agents.tools.builtin``. Import factory explicitly from
``weather_agents.core.factory`` when you need it.
"""

from weather_agents.core.agent import AgentState, BaseAgent, Task, TaskResult
from weather_agents.core.bus import Event, EventType, MessageBus
from weather_agents.core.cache import LLMCache
from weather_agents.core.config import (
    AppConfig,
    delete_config,
    load_config,
    load_model_catalog,
    set_config,
)
from weather_agents.core.llm import LLMClient, LLMResponse
from weather_agents.core.logger import get_logger, setup_logging
from weather_agents.core.mcp import MCPClient, MCPManager, MCPServerConfig
from weather_agents.core.memory import Memory
from weather_agents.core.skill import Skill, SkillRegistry, global_skill_registry
from weather_agents.core.tool import Tool, ToolParameter, ToolRegistry

__all__ = [
    "AppConfig",
    "AgentState",
    "BaseAgent",
    "Event",
    "EventType",
    "LLMCache",
    "LLMClient",
    "LLMResponse",
    "MCPClient",
    "MCPManager",
    "MCPServerConfig",
    "Memory",
    "MessageBus",
    "Skill",
    "SkillRegistry",
    "Task",
    "TaskResult",
    "Tool",
    "ToolParameter",
    "ToolRegistry",
    "delete_config",
    "get_logger",
    "global_skill_registry",
    "load_config",
    "load_model_catalog",
    "set_config",
    "setup_logging",
]

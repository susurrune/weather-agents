"""Core framework for Skyloom.

Public symbols are exposed via PEP 562 lazy loading (``__getattr__``).
Importing this package no longer pulls in ``mcp`` / ``llm`` / ``agent`` /
``memory`` eagerly — those modules (and their dependencies like ``httpx``
and ``litellm``) cost hundreds of ms at startup and are unnecessary for
``sky --help`` or any caller that only touches ``core.config``.

Reason for the change: ``cli/main.py`` does ``from weather_agents.core.config
import ...``, which still executes this ``__init__`` before the submodule.
Eager re-exports therefore tax every CLI invocation. Lazy attribute access
keeps the public surface stable for ``from weather_agents.core import X``
callers while letting ``sky --help`` skip the heavy dependency chain.

``factory`` is intentionally NOT exposed: it imports ``tools.builtin``
which itself depends on ``core.tool``. Loading ``factory`` from here would
create a circular import when third-party code touches ``tools.builtin``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Map exported name → (submodule, attribute). Resolved on first access.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "AgentState": ("agent", "AgentState"),
    "BaseAgent": ("agent", "BaseAgent"),
    "Task": ("agent", "Task"),
    "TaskResult": ("agent", "TaskResult"),
    "Event": ("bus", "Event"),
    "EventType": ("bus", "EventType"),
    "MessageBus": ("bus", "MessageBus"),
    "LLMCache": ("cache", "LLMCache"),
    "AppConfig": ("config", "AppConfig"),
    "delete_config": ("config", "delete_config"),
    "load_config": ("config", "load_config"),
    "load_model_catalog": ("config", "load_model_catalog"),
    "set_config": ("config", "set_config"),
    "LLMClient": ("llm", "LLMClient"),
    "LLMResponse": ("llm", "LLMResponse"),
    "get_logger": ("logger", "get_logger"),
    "setup_logging": ("logger", "setup_logging"),
    "MCPClient": ("mcp", "MCPClient"),
    "MCPManager": ("mcp", "MCPManager"),
    "MCPServerConfig": ("mcp", "MCPServerConfig"),
    "Memory": ("memory", "Memory"),
    "Skill": ("skill", "Skill"),
    "SkillRegistry": ("skill", "SkillRegistry"),
    "global_skill_registry": ("skill", "global_skill_registry"),
    "Tool": ("tool", "Tool"),
    "ToolParameter": ("tool", "ToolParameter"),
    "ToolRegistry": ("tool", "ToolRegistry"),
}

__all__ = [
    "AgentState",
    "AppConfig",
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


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'weather_agents.core' has no attribute {name!r}")
    module_name, attr_name = target
    from importlib import import_module

    module = import_module(f"weather_agents.core.{module_name}")
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return __all__


if TYPE_CHECKING:
    # Re-exports for IDE autocomplete / static analysis. ``__getattr__`` does
    # the real lazy work at runtime; this block is dead code at runtime.
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

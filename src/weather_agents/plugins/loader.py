"""Plugin loader for extending Skyloom."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from weather_agents.core.logger import get_logger
from weather_agents.core.tool import Tool, ToolRegistry

_log = get_logger("plugins")


class Plugin:
    """A plugin that can add tools and hooks to the system."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.tools: list[Tool] = []
        self.hooks: dict[str, Any] = {}

    def register_tool(self, tool: Tool) -> None:
        self.tools.append(tool)

    def on(self, event: str, handler: Any) -> None:
        self.hooks[event] = handler


class PluginLoader:
    """Load plugins from directories."""

    def __init__(self, tool_registry: ToolRegistry) -> None:
        self.tool_registry = tool_registry
        self.plugins: dict[str, Plugin] = {}

    def load_from_directory(self, directory: str | Path) -> list[Plugin]:
        dir_path = Path(directory).expanduser()
        if not dir_path.exists():
            return []

        loaded = []
        for py_file in dir_path.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            plugin = self._load_plugin_file(py_file)
            if plugin:
                self.plugins[plugin.name] = plugin
                for tool in plugin.tools:
                    self.tool_registry.register(tool)
                loaded.append(plugin)

        return loaded

    def _load_plugin_file(self, path: Path) -> Plugin | None:
        module_name = f"wa_plugin_{path.stem}"

        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if not spec or not spec.loader:
                return None

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Look for a create_plugin function
            if hasattr(module, "create_plugin"):
                plugin = module.create_plugin()
                if isinstance(plugin, Plugin):
                    return plugin

            # Look for a Plugin subclass
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and issubclass(attr, Plugin) and attr is not Plugin:
                    return attr()  # type: ignore[call-arg]

            return None
        except Exception as e:
            _log.warning("plugin_load_failed: %s — %s", path, e)
            return None

    def load_from_directories(self, directories: list[str]) -> list[Plugin]:
        all_loaded = []
        for d in directories:
            all_loaded.extend(self.load_from_directory(d))
        return all_loaded

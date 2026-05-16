"""Test fixtures and mocks for Weather Agents."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from weather_agents.core.bus import MessageBus
from weather_agents.core.tool import Tool, ToolRegistry


@pytest.fixture
def bus():
    return MessageBus()


@pytest.fixture
def tool_registry():
    r = ToolRegistry()
    r.register(
        Tool(
            name="test_tool",
            description="A test tool",
            parameters=[],
            handler=AsyncMock(return_value="tool result"),
        )
    )
    return r


@pytest.fixture
def mock_llm():
    llm = Mock()
    llm.complete = AsyncMock(
        return_value=Mock(
            content="test response",
            tool_calls=[],
            model="gpt-4o-mini",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            # Set reasoning_content explicitly so Mock auto-attribute doesn't
            # return a stub Mock object and end up serialized to the DB.
            reasoning_content=None,
        )
    )
    llm.stream = AsyncMock()
    llm.get_usage_stats = Mock(return_value={})
    llm.get_total_cost = Mock(return_value=0.0)
    return llm


@pytest.fixture
def app_config(tmp_path):
    """AppConfig with an isolated SQLite DB.

    Without this override, every test run hit ``~/.weather-agents/memory.db``
    — the user's real database — leaving session rows, mock-stringified
    messages, and other artefacts that polluted production data. The
    ``tmp_path`` fixture gives each test its own clean directory.
    """
    from weather_agents.core.config import AppConfig

    cfg = AppConfig()
    cfg.memory.db_path = str(tmp_path / "test_memory.db")
    return cfg


@pytest.fixture
def temp_config_dir(tmp_path):
    """Isolate config tests to a temp directory so user config is not touched."""
    user_cfg = tmp_path / ".weather-agents"
    user_cfg.mkdir()
    with patch("weather_agents.core.config.USER_CONFIG_DIR", user_cfg):
        # Also invalidate cache to pick up new dir
        from weather_agents.core.config import invalidate_cache

        invalidate_cache()
        yield user_cfg
        invalidate_cache()

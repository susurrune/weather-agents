"""Tests for MCP client and manager configuration."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from weather_agents.core.mcp import MCPClient, MCPManager, MCPServerConfig
from weather_agents.core.tool import ToolRegistry


class TestMCPServerConfig:
    def test_stdio_config(self):
        cfg = MCPServerConfig(
            name="test_server",
            command="python",
            args=["-m", "test_server"],
            env={"KEY": "val"},
        )
        assert cfg.name == "test_server"
        assert cfg.command == "python"
        assert cfg.args == ["-m", "test_server"]
        assert cfg.url is None

    def test_sse_config(self):
        cfg = MCPServerConfig(name="sse_server", url="http://localhost:8080")
        assert cfg.name == "sse_server"
        assert cfg.url == "http://localhost:8080"
        assert cfg.command is None

    def test_default_enabled(self):
        cfg = MCPServerConfig(name="x")
        assert cfg.enabled is True

    def test_disabled_config(self):
        cfg = MCPServerConfig(name="x", enabled=False)
        assert cfg.enabled is False


class TestMCPClient:
    def test_client_initialization(self):
        cfg = MCPServerConfig(name="test", command="echo")
        client = MCPClient(cfg)
        assert client.config.name == "test"
        assert client._process is None
        assert client._server_tools == []

    def test_new_id_increments(self):
        cfg = MCPServerConfig(name="test")
        client = MCPClient(cfg)
        id1 = client._new_id()
        id2 = client._new_id()
        assert id2 > id1


class TestMCPManager:
    def test_manager_empty_servers(self):
        registry = ToolRegistry()
        manager = MCPManager(registry)
        assert len(manager.clients) == 0
        assert len(manager._server_configs) == 0

    def test_configure_adds_configs(self):
        registry = ToolRegistry()
        manager = MCPManager(registry)
        manager.configure(
            [
                {"name": "echo", "command": "echo"},
            ]
        )
        assert len(manager._server_configs) == 1
        assert manager._server_configs[0].name == "echo"

    def test_configure_skips_disabled(self):
        registry = ToolRegistry()
        manager = MCPManager(registry)
        manager.configure(
            [
                {"name": "echo", "command": "echo", "enabled": False},
            ]
        )
        assert len(manager._server_configs) == 0

    def test_configure_empty_list(self):
        registry = ToolRegistry()
        manager = MCPManager(registry)
        manager.configure([])
        assert len(manager._server_configs) == 0

    @pytest.mark.asyncio
    async def test_close_all_no_clients(self):
        registry = ToolRegistry()
        manager = MCPManager(registry)
        await manager.close_all()

    @pytest.mark.asyncio
    async def test_connect_all_empty(self):
        registry = ToolRegistry()
        manager = MCPManager(registry)
        status = await manager.connect_all()
        assert status == []

    @pytest.mark.asyncio
    async def test_connect_all_with_configs(self):
        registry = ToolRegistry()
        manager = MCPManager(registry)
        manager.configure(
            [
                {"name": "mock_srv", "command": "python", "args": ["-c", "print('ok')"]},
            ]
        )

        with patch.object(MCPClient, "initialize") as mock_init:
            mock_init.return_value = [
                {
                    "name": "mock_tool",
                    "description": "A mock tool",
                    "inputSchema": {"type": "object"},
                },
            ]
            status = await manager.connect_all()
            assert len(status) == 1
            assert "mock_srv" in status[0]
            assert "mock_srv" in manager.clients
            mock_init.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_connect_all_registers_tools(self):
        registry = ToolRegistry()
        manager = MCPManager(registry)
        manager.configure(
            [
                {"name": "toolsrv", "command": "cat"},
            ]
        )

        with patch.object(MCPClient, "initialize") as mock_init:
            mock_init.return_value = [
                {
                    "name": "hello",
                    "description": "Say hello",
                    "inputSchema": {"type": "object", "properties": {}},
                },
            ]
            await manager.connect_all()
            tool = registry.get("mcp_toolsrv_hello")
            assert tool is not None
            assert tool.name == "mcp_toolsrv_hello"

    @pytest.mark.asyncio
    async def test_connect_all_isolates_per_server_failure(self):
        """A single server's init failure must NOT abort the others.
        connect_all runs initializations in parallel and catches each
        failure locally so working servers still register their tools.
        Previously this raised on the first error, taking the entire
        MCP layer down whenever any one server was misconfigured."""
        registry = ToolRegistry()
        manager = MCPManager(registry)
        manager.configure(
            [
                {"name": "bad_srv", "command": "nonexistent_binary"},
                {"name": "good_srv", "command": "real_binary"},
            ]
        )

        async def _init_side_effect(self):
            if self.config.name == "bad_srv":
                raise Exception("connection failed")
            # good_srv returns one tool
            return [{"name": "ok_tool", "description": "", "inputSchema": {}}]

        with patch.object(MCPClient, "initialize", autospec=True) as mock_init:
            mock_init.side_effect = _init_side_effect
            results = await manager.connect_all()
        # bad_srv silently skipped; good_srv tools registered
        assert any("good_srv" in r for r in results)
        assert not any("bad_srv" in r for r in results)

    @pytest.mark.asyncio
    async def test_client_tool_definitions(self):
        registry = ToolRegistry()
        manager = MCPManager(registry)
        manager.configure(
            [
                {"name": "srv", "command": "cat"},
            ]
        )

        # Manually add a client so we can test get_tool_definitions
        client = MCPClient(manager._server_configs[0])
        client._server_tools = [
            {"name": "tool_a", "description": "Tool A"},
            {"name": "tool_b", "description": "Tool B"},
        ]
        manager.clients["srv"] = client

        defs = client.get_tool_definitions()
        assert len(defs) == 2
        assert defs[0]["name"] == "tool_a"

    @pytest.mark.asyncio
    async def test_close_all_cleans_up(self):
        registry = ToolRegistry()
        manager = MCPManager(registry)
        manager.configure(
            [
                {"name": "srv", "command": "echo"},
            ]
        )

        with (
            patch.object(MCPClient, "initialize") as mock_init,
            patch.object(MCPClient, "close") as mock_close,
        ):
            mock_init.return_value = [
                {"name": "t", "description": "x", "inputSchema": {"type": "object"}},
            ]
            await manager.connect_all()
            assert len(manager.clients) == 1

            await manager.close_all()
            mock_close.assert_awaited_once()
            assert len(manager.clients) == 0


class TestSSEParsing:
    def test_parse_simple_event(self):
        from weather_agents.core.mcp import MCPClient

        event = MCPClient._parse_sse_event("event: endpoint\ndata: /messages\n")
        assert event is not None
        assert event["event"] == "endpoint"
        assert event["data"] == "/messages"

    def test_parse_event_default_type(self):
        from weather_agents.core.mcp import MCPClient

        event = MCPClient._parse_sse_event('data: {"jsonrpc": "2.0"}\n')
        assert event is not None
        assert event["event"] == "message"
        assert event["data"] == '{"jsonrpc": "2.0"}'

    def test_parse_multiline_data(self):
        from weather_agents.core.mcp import MCPClient

        event = MCPClient._parse_sse_event("data: line1\ndata: line2\n")
        assert event is not None
        assert event["data"] == "line1\nline2"

    def test_parse_skip_comments(self):
        from weather_agents.core.mcp import MCPClient

        event = MCPClient._parse_sse_event(": heartbeat\ndata: real\n")
        assert event is not None
        assert event["data"] == "real"

    def test_parse_no_data_is_none(self):
        from weather_agents.core.mcp import MCPClient

        event = MCPClient._parse_sse_event("event: heartbeat\n")
        assert event is None

    def test_parse_event_with_retry(self):
        from weather_agents.core.mcp import MCPClient

        event = MCPClient._parse_sse_event("data: msg\nretry: 3000\n")
        assert event is not None
        assert event["data"] == "msg"
        assert event.get("retry") == "3000"


class TestMCPHealthCheck:
    def test_health_no_transport(self):
        from weather_agents.core.mcp import MCPClient, MCPServerConfig

        cfg = MCPServerConfig(name="test", enabled=False)
        client = MCPClient(cfg)
        # No command, no URL — no transport
        import asyncio

        result = asyncio.run(client.health_check())
        assert result["healthy"] is False
        assert "No transport" in result["details"]

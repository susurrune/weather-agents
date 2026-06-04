"""Tests for MCP runtime self-integration tools."""

from __future__ import annotations

import pytest

from weather_agents.core import mcp
from weather_agents.core.tool import ToolRegistry

# ── Scaffold template ──


def test_scaffold_template_compiles():
    import ast

    from weather_agents.tools.mcp_tools import _SERVER_TEMPLATE

    src = _SERVER_TEMPLATE.format(name="test_srv", description="desc")
    ast.parse(src)
    # Must be valid, runnable Python.


def test_scaffold_template_roundtrip():
    from weather_agents.tools.mcp_tools import _SERVER_TEMPLATE

    src = _SERVER_TEMPLATE.format(name="hello_srv", description="test")
    ns = {}
    exec(src, ns)
    handle = ns["handle"]

    # initialize
    r = handle("initialize", {})
    assert r is not None
    assert r["protocolVersion"] == "2025-03-26"
    assert "tools" in r["capabilities"]

    # tools/list
    r = handle("tools/list", {})
    assert r is not None
    tools = r["tools"]
    assert any(t["name"] == "hello" for t in tools)

    # tools/call — hello
    r = handle("tools/call", {"name": "hello", "arguments": {"name": "Skyloom"}})
    assert r is not None
    assert "content" in r
    assert r["content"][0]["type"] == "text"
    assert "Skyloom" in r["content"][0]["text"]

    # tools/call — unknown tool returns isError
    r = handle("tools/call", {"name": "nonexistent", "arguments": {}})
    assert r is not None
    assert r.get("isError") is True

    # Unknown method returns None
    assert handle("unknown_method", {}) is None

    # notifications/ are no-ops for handle
    assert handle("notifications/initialized", {}) is None


# ── Registration ──


def test_mcp_tools_registered():
    from weather_agents.tools.mcp_tools import register_mcp_tools

    reg = ToolRegistry()
    register_mcp_tools(reg)
    names = set(reg.list_names())
    for expected in (
        "mcp_list_servers",
        "mcp_add_server",
        "mcp_remove_server",
        "mcp_scaffold_server",
    ):
        assert expected in names


def test_dangerous_ones_are_flagged():
    from weather_agents.tools.mcp_tools import register_mcp_tools

    reg = ToolRegistry()
    register_mcp_tools(reg)
    assert reg.get("mcp_add_server").dangerous is True
    assert reg.get("mcp_scaffold_server").dangerous is True
    assert reg.get("mcp_list_servers").dangerous is False


# ── Validation (no active manager needed for these) ──


@pytest.mark.asyncio
async def test_add_server_requires_name():
    from weather_agents.tools.mcp_tools import _mcp_add_server

    out = await _mcp_add_server(name="", command="x")
    assert "required" in out.lower() or "Error" in out


@pytest.mark.asyncio
async def test_add_server_no_manager_reports_unready():
    from weather_agents.tools.mcp_tools import _mcp_add_server

    out = await _mcp_add_server(name="x")
    assert "未就绪" in out or "Error" in out


@pytest.mark.asyncio
async def test_remove_server_requires_name():
    from weather_agents.tools.mcp_tools import _mcp_remove_server

    out = await _mcp_remove_server(name="")
    assert "required" in out.lower() or "Error" in out


@pytest.mark.asyncio
async def test_scaffold_invalid_name():
    from weather_agents.tools.mcp_tools import _mcp_scaffold_server

    out = await _mcp_scaffold_server(name="", description="x")
    assert "Error" in out or "alphanumeric" in out


# ── Manager lifecycle (mock subprocess) ──


@pytest.mark.asyncio
async def test_add_server_flow(monkeypatch):
    """Simulate adding an MCP server that exposes one tool."""
    reg = ToolRegistry()
    mgr = mcp.MCPManager(reg)
    mcp.set_active_manager(mgr)

    # Fake the client initialization so we don't really spawn a process.
    async def fake_init(self):
        return [{"name": "echo", "description": "echoes input"}]

    monkeypatch.setattr(mcp.MCPClient, "initialize", fake_init)

    result = await mgr.add_server({"name": "test", "command": "echo"})
    assert "✓" in result
    assert "test" in result
    assert "test" in mgr.clients
    assert "mcp_test_echo" in reg.list_names()
    # A second add of the same name should reject.
    result2 = await mgr.add_server({"name": "test", "command": "echo"})
    assert "already" in result2.lower()

    await mgr.close_all()
    mcp.set_active_manager(None)


@pytest.mark.asyncio
async def test_list_servers_empty():
    reg = ToolRegistry()
    mgr = mcp.MCPManager(reg)
    mcp.set_active_manager(mgr)
    servers = mgr.list_servers()
    assert servers == []
    mcp.set_active_manager(None)


@pytest.mark.asyncio
async def test_persist_roundtrip(tmp_path, monkeypatch):
    """Save a server config, reload it, verify it appears."""
    monkeypatch.setattr(mcp, "_persist_path", lambda: tmp_path / "mcp_servers.json")
    mcp.save_persisted_server({"name": "p_test", "command": "echo", "args": []})
    servers = mcp.load_persisted_servers()
    assert len(servers) == 1
    assert servers[0]["name"] == "p_test"
    # Removing it should clear.
    mcp.remove_persisted_server("p_test")
    assert mcp.load_persisted_servers() == []
    # Removing a non-existent server is a no-op.
    mcp.remove_persisted_server("nope")


@pytest.mark.asyncio
async def test_remove_server_flow(monkeypatch):
    reg = ToolRegistry()
    mgr = mcp.MCPManager(reg)
    mcp.set_active_manager(mgr)

    async def fake_init(self):
        return [{"name": "t1", "description": "d"}]

    monkeypatch.setattr(mcp.MCPClient, "initialize", fake_init)
    await mgr.add_server({"name": "rem", "command": "echo"})
    assert "mcp_rem_t1" in reg.list_names()

    result = await mgr.remove_server("rem")
    assert "✓" in result
    assert "rem" not in mgr.clients
    assert "mcp_rem_t1" not in reg.list_names()

    await mgr.close_all()
    mcp.set_active_manager(None)


# ── Active manager singleton ──


def test_set_get_active_manager():
    mcp.set_active_manager(None)
    assert mcp.get_active_manager() is None
    reg = ToolRegistry()
    mgr = mcp.MCPManager(reg)
    mcp.set_active_manager(mgr)
    assert mcp.get_active_manager() is mgr
    mcp.set_active_manager(None)

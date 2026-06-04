"""Runtime MCP self-integration tools.

Let an agent connect Skyloom to *any* software at runtime by adding MCP servers
on the fly — and even write its own MCP server to wrap a tool that has no
existing integration. Built on the existing MCPManager/MCPClient; new servers'
tools propagate into every agent's registry and are persisted under
``~/.skyloom/mcp_servers.json`` so they survive a restart.

Security: mcp_add_server and mcp_scaffold_server are dangerous=True — both end
up running an external process (the MCP server), so they ride the approval gate.
"""

from __future__ import annotations

import shlex

from weather_agents.core import mcp as _mcp
from weather_agents.core.tool import Tool, ToolParameter, ToolRegistry

# A minimal, dependency-free MCP stdio server the agent can extend. Implements
# just enough of JSON-RPC 2.0 / MCP to be discovered and called by our client.
_SERVER_TEMPLATE = '''"""MCP server: {name}

{description}

Scaffolded by Skyloom. Edit TOOLS + handle() to wrap your software, then it is
re-exposed to every agent as mcp_{name}_<tool>. Pure stdlib — no dependencies.
"""

import json
import sys

# Declare your tools here. Each needs name, description, and a JSON-Schema
# inputSchema. Add as many as you like.
TOOLS = [
    {{
        "name": "hello",
        "description": "Sample tool — returns a greeting. Replace with real logic.",
        "inputSchema": {{
            "type": "object",
            "properties": {{"name": {{"type": "string", "description": "who to greet"}}}},
            "required": [],
        }},
    }},
]


def handle(method, params):
    """Return a JSON-RPC *result* dict for the given method, or None if unknown."""
    if method == "initialize":
        return {{
            "protocolVersion": "2025-03-26",
            "capabilities": {{"tools": {{}}}},
            "serverInfo": {{"name": "{name}", "version": "0.1.0"}},
        }}
    if method == "tools/list":
        return {{"tools": TOOLS}}
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments", {{}})
        # ── Implement each tool below ──
        if name == "hello":
            who = args.get("name", "world")
            return {{"content": [{{"type": "text", "text": f"Hello, {{who}}! (from {name})"}}]}}
        return {{"content": [{{"type": "text", "text": f"unknown tool: {{name}}"}}], "isError": True}}
    return None


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        mid = msg.get("id")
        method = msg.get("method")
        params = msg.get("params", {{}})
        if method and method.startswith("notifications/"):
            continue  # notifications get no response
        try:
            result = handle(method, params)
            resp = (
                {{"jsonrpc": "2.0", "id": mid, "result": result}}
                if result is not None
                else {{"jsonrpc": "2.0", "id": mid, "error": {{"code": -32601, "message": "method not found"}}}}
            )
        except Exception as e:  # noqa: BLE001
            resp = {{"jsonrpc": "2.0", "id": mid, "error": {{"code": -32603, "message": str(e)}}}}
        if mid is not None:
            sys.stdout.write(json.dumps(resp) + "\\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
'''


def _mcp_servers_dir():
    from weather_agents.core import config as _cfg

    d = _cfg.USER_CONFIG_DIR / "mcp_servers"
    d.mkdir(parents=True, exist_ok=True)
    return d


async def _mcp_list_servers(**kwargs) -> str:
    mgr = _mcp.get_active_manager()
    if mgr is None:
        return "MCP 管理器未就绪（当前会话未启用 MCP）。"
    servers = mgr.list_servers()
    if not servers:
        return (
            "尚未接入任何 MCP server。用 mcp_add_server 接入，或 mcp_scaffold_server 自己写一个。"
        )
    lines = ["已接入的 MCP server:"]
    for s in servers:
        flag = "●" if s["connected"] else "○"
        lines.append(
            f"  {flag} {s['name']}  [{s['transport']}]  {s['tools']} 工具  → {s['target']}"
        )
    return "\n".join(lines)


async def _mcp_add_server(
    name: str = "", command: str = "", args: str = "", url: str = "", **kwargs
) -> str:
    name = (name or "").strip()
    if not name:
        return "Error: name is required"
    mgr = _mcp.get_active_manager()
    if mgr is None:
        return "MCP 管理器未就绪（当前会话未启用 MCP）。"
    config_dict: dict = {"name": name}
    if command.strip():
        config_dict["command"] = command.strip()
        config_dict["args"] = shlex.split(args) if args.strip() else []
    elif url.strip():
        config_dict["url"] = url.strip()
    else:
        return "Error: provide 'command' (stdio) or 'url' (SSE)"
    result = await mgr.add_server(config_dict)
    if result.startswith("✓"):
        _mcp.save_persisted_server(config_dict)
    return result


async def _mcp_remove_server(name: str = "", **kwargs) -> str:
    name = (name or "").strip()
    if not name:
        return "Error: name is required"
    mgr = _mcp.get_active_manager()
    if mgr is None:
        return "MCP 管理器未就绪。"
    result = await mgr.remove_server(name)
    _mcp.remove_persisted_server(name)
    return result


async def _mcp_scaffold_server(name: str = "", description: str = "", **kwargs) -> str:
    """Write a fresh MCP server stub, then connect it so it's usable at once."""
    import sys

    name = (name or "").strip()
    if not name or not name.replace("_", "").replace("-", "").isalnum():
        return "Error: name must be alphanumeric (underscores/hyphens ok)"
    mgr = _mcp.get_active_manager()
    if mgr is None:
        return "MCP 管理器未就绪。"
    path = _mcp_servers_dir() / f"{name}.py"
    if path.exists():
        return f"已存在: {path}（先用 mcp_remove_server 断开并删文件，或换个名字）"
    path.write_text(
        _SERVER_TEMPLATE.format(name=name, description=description or "Custom MCP server."),
        encoding="utf-8",
    )
    config_dict = {"name": name, "command": sys.executable, "args": [str(path)]}
    result = await mgr.add_server(config_dict)
    if result.startswith("✓"):
        _mcp.save_persisted_server(config_dict)
        return (
            f"✓ 已生成并接入 MCP server '{name}'：{path}\n"
            f"{result}\n"
            "现在它带一个示例工具 hello。用 edit_file 修改 TOOLS 和 handle() 加入真实逻辑，"
            f"再 mcp_remove_server {name} 然后重新 mcp_add_server 即可生效。"
        )
    return f"已生成 {path}，但接入失败：{result}"


def register_mcp_tools(reg: ToolRegistry) -> None:
    """Register the runtime MCP-management tools into *reg*."""
    tools = [
        Tool(
            name="mcp_list_servers",
            description="List connected MCP servers and their tool counts.",
            parameters=[],
            handler=_mcp_list_servers,
            cacheable=False,
        ),
        Tool(
            name="mcp_add_server",
            description=(
                "Connect a new MCP server at runtime to integrate external software. "
                "Provide either 'command' (+optional 'args' string) for a stdio server, "
                "or 'url' for an SSE server. Its tools become available to all agents "
                "immediately and persist across restarts. "
                "e.g. command='npx' args='-y @modelcontextprotocol/server-filesystem /path'."
            ),
            parameters=[
                ToolParameter(name="name", type="string", description="Unique server name"),
                ToolParameter(
                    name="command",
                    type="string",
                    description="Executable for stdio transport",
                    required=False,
                ),
                ToolParameter(
                    name="args",
                    type="string",
                    description="Space-separated args for command",
                    required=False,
                ),
                ToolParameter(
                    name="url",
                    type="string",
                    description="Base URL for SSE transport",
                    required=False,
                ),
            ],
            handler=_mcp_add_server,
            dangerous=True,
            cacheable=False,
        ),
        Tool(
            name="mcp_remove_server",
            description="Disconnect an MCP server and remove its tools.",
            parameters=[
                ToolParameter(name="name", type="string", description="Server name to remove")
            ],
            handler=_mcp_remove_server,
            cacheable=False,
        ),
        Tool(
            name="mcp_scaffold_server",
            description=(
                "Write a new minimal MCP server (pure-Python stdlib stub) and connect "
                "it — use this to wrap software that has no existing MCP server. "
                "Returns the file path; edit its TOOLS + handle() to add real logic, "
                "then reconnect. This is how Skyloom writes its own integrations."
            ),
            parameters=[
                ToolParameter(name="name", type="string", description="Server name (alphanumeric)"),
                ToolParameter(
                    name="description",
                    type="string",
                    description="What this server integrates",
                    required=False,
                ),
            ],
            handler=_mcp_scaffold_server,
            dangerous=True,
            cacheable=False,
        ),
    ]
    for tool in tools:
        reg.register(tool)

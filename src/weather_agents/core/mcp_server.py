"""MCP server — expose Skyloom agents as tools for external MCP clients.

Runs over stdio (JSON-RPC 2.0 over stdin/stdout, same protocol used by our
MCP client in ``mcp.py``).  Any MCP-compatible client (Claude Desktop, Zed,
Continue, etc.) can connect and call Skyloom agents as if they were local
tools.

Usage as a standalone process::

    python -m weather_agents.core.mcp_server
    # or:  sky mcp

Registered tools:
- mcp_chat   — single-turn question to any agent
- mcp_task   — multi-agent orchestration
- list_agents — return available agent names + specialties
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

MCP_VERSION = "2025-03-26"
SERVER_INFO = {"name": "skyloom", "version": "1.4.0"}

_TOOL_DEFS: list[dict] = [
    {
        "name": "mcp_chat",
        "description": (
            "Ask a Skyloom agent a question. Returns the agent's reply."
            " Use 'fair'(晴) for companionship, 'fog' for research,"
            " 'rain' for code generation, 'frost' for code review,"
            " 'dew' for system ops, 'snow' for planning."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "description": "Agent name: fog/rain/frost/snow/dew/fair (default: fair)",
                    "default": "fair",
                },
                "message": {
                    "type": "string",
                    "description": "What to ask the agent",
                },
            },
            "required": ["message"],
        },
    },
    {
        "name": "mcp_task",
        "description": (
            "Run multi-agent orchestration. Snow plans, then agents collaborate."
            " Use for complex, multi-step goals that span domains."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "The goal to accomplish (e.g. 'Write a URL shortener in Go')",
                },
            },
            "required": ["goal"],
        },
    },
    {
        "name": "list_agents",
        "description": "List available Skyloom agents and their specialties.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
]


class MCPServer:
    """Stdio JSON-RPC 2.0 MCP server wrapping Skyloom agents."""

    def __init__(self) -> None:
        self._initialized = False
        self._ctx = None
        self._agents: dict[str, Any] = {}

    async def run(self) -> None:
        """Read JSON-RPC from stdin, write responses to stdout.  Blocks until
        stdin closes or a fatal error occurs."""
        reader = asyncio.StreamReader()
        transport, _ = await asyncio.get_event_loop().connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(reader), sys.stdin
        )
        while True:
            line = await reader.readline()
            if not line:
                break
            line_str = line.decode("utf-8", errors="replace").strip()
            if not line_str:
                continue
            try:
                msg = json.loads(line_str)
            except json.JSONDecodeError:
                continue
            resp = await self._handle(msg)
            if resp is not None:
                self._write(resp)
        transport.close()

    def _write(self, obj: dict) -> None:
        sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    async def _handle(self, msg: dict) -> dict | None:
        mid = msg.get("id")
        method = msg.get("method", "")
        params = msg.get("params", {})

        # Notifications — no response
        if method.startswith("notifications/"):
            return None

        try:
            if method == "initialize":
                result = await self._init(params)
            elif method == "tools/list":
                result = await self._tools_list(params)
            elif method == "tools/call":
                result = await self._tools_call(params)
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": mid,
                    "error": {"code": -32601, "message": f"unknown method: {method}"},
                }
            return {"jsonrpc": "2.0", "id": mid, "result": result}
        except Exception as exc:  # noqa: BLE001
            return {
                "jsonrpc": "2.0",
                "id": mid,
                "error": {"code": -32603, "message": str(exc)},
            }

    # ── Method handlers ──

    async def _init(self, _params: dict) -> dict:
        self._initialized = True
        return {
            "protocolVersion": MCP_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": SERVER_INFO,
        }

    async def _tools_list(self, _params: dict) -> dict:
        return {"tools": _TOOL_DEFS}

    async def _tools_call(self, params: dict) -> dict:
        name = params.get("name", "")
        args = params.get("arguments", {})

        if name == "list_agents":
            return await self._handle_list_agents(args)
        if name == "mcp_chat":
            return await self._handle_chat(args)
        if name == "mcp_task":
            return await self._handle_task(args)
        return {
            "content": [{"type": "text", "text": f"Unknown tool: {name}"}],
            "isError": True,
        }

    # ── Lazy context init ──

    async def _ensure_ctx(self) -> None:
        if self._ctx is None:
            from weather_agents.core.factory import create_system_context

            self._ctx = create_system_context()
            await self._ctx.init_all()
            self._agents = self._ctx.agent_map

    # ── Tool implementations ──

    async def _handle_list_agents(self, _args: dict) -> dict:
        await self._ensure_ctx()
        lines = ["Available agents:"]
        for name, agent in self._agents.items():
            lines.append(f"- {name} ({agent.display_name}): {agent.specialty}")
        return {"content": [{"type": "text", "text": "\n".join(lines)}]}

    async def _handle_chat(self, args: dict) -> dict:
        agent_name = str(args.get("agent", "fair")).strip().lower()
        message = str(args.get("message", "")).strip()
        if not message:
            return {
                "content": [{"type": "text", "text": "Error: message is required"}],
                "isError": True,
            }

        await self._ensure_ctx()
        if agent_name not in self._agents:
            names = ", ".join(self._agents)
            return {
                "content": [
                    {"type": "text", "text": f"Unknown agent '{agent_name}'. Available: {names}"}
                ],
                "isError": True,
            }

        agent = self._agents[agent_name]
        try:
            reply = await agent.chat(message)
            return {"content": [{"type": "text", "text": reply}]}
        except Exception as exc:  # noqa: BLE001
            return {"content": [{"type": "text", "text": f"Agent error: {exc}"}], "isError": True}

    async def _handle_task(self, args: dict) -> dict:
        goal = str(args.get("goal", "")).strip()
        if not goal:
            return {
                "content": [{"type": "text", "text": "Error: goal is required"}],
                "isError": True,
            }

        await self._ensure_ctx()
        from weather_agents.core.factory import orchestrate_task

        try:
            _tasks, results, summary = await orchestrate_task(
                goal, self._agents, result_truncate=2000
            )
            ok = sum(1 for r in results if r.success)
            total = len(results)
            content = f"[{ok}/{total} tasks completed]\n" + "\n".join(
                f"## {r.agent}: {r.description}\n{r.content or '(no content)'}" for r in results
            )
            if summary:
                content += f"\n\n{summary}"
            return {"content": [{"type": "text", "text": content}]}
        except Exception as exc:  # noqa: BLE001
            return {"content": [{"type": "text", "text": f"Task error: {exc}"}], "isError": True}


async def main() -> None:
    server = MCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())

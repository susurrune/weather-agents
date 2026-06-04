"""MCP (Model Context Protocol) client — Anthropic-compatible transport.

Implements the MCP 2025-03-26 specification with two transports:
- stdio: subprocess-based with JSON-RPC over stdin/stdout
- SSE: text/event-stream via httpx.stream with jsonrpc POST endpoint
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any

import httpx

from weather_agents.core.logger import get_logger
from weather_agents.core.tool import Tool, ToolParameter, ToolRegistry

_log = get_logger("mcp")

MCP_PROTOCOL_VERSION = "2025-03-26"
CLIENT_INFO = {"name": "skyloom", "version": "1.0.0"}


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection."""

    name: str
    command: str | None = None
    args: list[str] = field(default_factory=list)
    url: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True


class MCPClient:
    """Client for connecting to a single MCP server.

    Supports both stdio (subprocess) and SSE (text/event-stream) transports.
    Uses JSON-RPC 2.0 with id-based correlation for concurrent requests.
    """

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self._process: asyncio.subprocess.Process | None = None
        self._server_tools: list[dict] = []
        self._next_id = 0
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._reader_task: asyncio.Task | None = None
        # Separate task that drains the subprocess's stderr pipe. Without
        # this an MCP server that logs more than ~64KB of debug output to
        # stderr eventually blocks on write() and the whole server hangs.
        self._stderr_task: asyncio.Task | None = None
        self._send_lock = asyncio.Lock()
        # SSE-specific state
        self._http_client: httpx.AsyncClient | None = None
        self._sse_response: httpx.Response | None = None
        self._sse_message_url: str = ""

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    # ── Initialize ──────────────────────────────────────────────────────

    async def initialize(self) -> list[dict]:
        """Connect to the MCP server and list available tools."""
        try:
            if self.config.command:
                return await self._init_stdio()
            elif self.config.url:
                return await self._init_sse()
        except (FileNotFoundError, OSError) as e:
            _log.warning(
                "mcp_server_unavailable",
                extra={"server": self.config.name, "error": str(e)},
            )
        return []

    async def health_check(self) -> dict[str, Any]:
        """Perform a health check on the MCP connection.

        Returns a dict with 'healthy' (bool) and 'details' (str).
        """
        if self.config.command:
            return await self._health_stdio()
        elif self.config.url:
            return await self._health_sse()
        return {"healthy": False, "details": "No transport configured"}

    # ── stdio transport ─────────────────────────────────────────────────

    async def _init_stdio(self) -> list[dict]:
        cmd = self.config.command
        if not cmd:
            return []

        env = {**os.environ, **(self.config.env or {})}
        self._process = await asyncio.create_subprocess_exec(
            cmd,
            *self.config.args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        self._reader_task = asyncio.create_task(self._read_stdio_loop())
        self._stderr_task = asyncio.create_task(self._drain_stderr())

        try:
            init_response = await self._request_stdio(
                "initialize",
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": CLIENT_INFO,
                },
            )
            if not init_response or "error" in init_response:
                return []

            await self._send_json(
                {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
            )

            tools_response = await self._request_stdio("tools/list", {})
            if tools_response and "result" in tools_response:
                self._server_tools = tools_response["result"].get("tools", [])

            return self._server_tools
        except Exception as e:
            _log.warning(
                "mcp_stdio_init_failed",
                extra={"server": self.config.name, "error": str(e)},
            )
            await self.close()
            return []

    async def _health_stdio(self) -> dict[str, Any]:
        if not self._process or self._process.returncode is not None:
            return {"healthy": False, "details": "stdio process not running"}
        try:
            resp = await self._request_stdio("ping", {}, timeout=5.0)
            if resp and "result" in resp:
                return {"healthy": True, "details": "ok"}
            return {"healthy": False, "details": f"unexpected ping response: {resp}"}
        except Exception as e:
            return {"healthy": False, "details": str(e)}

    async def _read_stdio_loop(self) -> None:
        if not self._process or not self._process.stdout:
            return
        stdout = self._process.stdout
        try:
            while True:
                line = await stdout.readline()
                if not line:
                    break
                msg = self._parse_json_line(line)
                if msg is None:
                    continue
                msg_id = msg.get("id")
                if msg_id is None:
                    continue
                fut = self._pending.get(msg_id)
                if fut and not fut.done():
                    fut.set_result(msg)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log.warning(
                "mcp_stdio_read_error",
                extra={"server": self.config.name, "error": str(e)},
            )
        finally:
            # Stdout EOF or read error means no further responses can arrive
            # for this client — fail every in-flight request so callers
            # unblock with CancelledError instead of waiting until their
            # asyncio.wait_for timeout fires. Cancellation belongs here (the
            # response stream is dead) and explicitly NOT in _drain_stderr
            # (which only manages the side-channel log pipe).
            for fut in self._pending.values():
                if not fut.done():
                    fut.cancel()

    async def _drain_stderr(self) -> None:
        """Continuously drain the subprocess's stderr pipe so it never
        blocks on a full kernel buffer. We log a sample so server-side
        crashes are debuggable but discard the rest to keep memory bounded.
        """
        if not self._process or not self._process.stderr:
            return
        stderr = self._process.stderr
        try:
            while True:
                line = await stderr.readline()
                if not line:
                    break
                # Only log the first few stderr lines; many MCP servers spam
                # debug output that would otherwise drown out the logs.
                _log.debug(
                    "mcp_stderr",
                    extra={
                        "server": self.config.name,
                        "line": line.decode("utf-8", errors="replace").rstrip(),
                    },
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            # Stderr drain must never crash the client; pipe closure during
            # shutdown surfaces as a generic OSError on Windows.
            return
        # NOTE: pending-future cancellation deliberately lives in
        # _read_stdio_loop's finally, not here. stderr EOF can happen while
        # stdout is still serving responses (e.g. a server that closes its
        # log channel but keeps the RPC channel alive); cancelling pending
        # from here would unnecessarily fail those in-flight requests.

    async def _request_stdio(
        self, method: str, params: dict, timeout: float = 10.0
    ) -> dict[str, Any] | None:
        if not self._process or not self._process.stdin:
            return None
        req_id = self._new_id()
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[req_id] = fut

        await self._send_json({"jsonrpc": "2.0", "method": method, "params": params, "id": req_id})
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except TimeoutError:
            _log.warning(
                "mcp_stdio_timeout",
                extra={"server": self.config.name, "method": method},
            )
            return None
        finally:
            self._pending.pop(req_id, None)

    async def _send_json(self, data: dict) -> None:
        if not self._process or not self._process.stdin:
            return
        line = json.dumps(data, ensure_ascii=False) + "\n"
        async with self._send_lock:
            self._process.stdin.write(line.encode())
            await self._process.stdin.drain()

    # ── SSE transport ───────────────────────────────────────────────────

    async def _init_sse(self) -> list[dict]:
        """Connect via SSE transport per MCP 2025-03-26 spec.

        Flow:
        1. GET /sse to open the event stream
        2. Receive `endpoint` event with the message URL
        3. POST initialize to the message endpoint
        4. POST tools/list to the message endpoint
        """
        url = (self.config.url or "").rstrip("/")
        sse_url = f"{url}/sse"

        try:
            self._http_client = httpx.AsyncClient(timeout=httpx.Timeout(30, connect=10))
            # Open SSE stream
            response = await self._http_client.send(
                self._http_client.build_request("GET", sse_url),
                stream=True,
            )
            self._sse_response = response
            if response.status_code != 200:
                _log.warning(
                    "mcp_sse_connect_failed",
                    extra={"server": self.config.name, "status": response.status_code},
                )
                return []

            # Read the endpoint event
            self._sse_message_url = ""
            sse_events = self._iter_sse_events(response)
            try:
                first_event = await asyncio.wait_for(sse_events.__anext__(), timeout=10)
            except (TimeoutError, StopAsyncIteration):
                _log.warning(
                    "mcp_sse_no_endpoint",
                    extra={"server": self.config.name},
                )
                return []

            if first_event.get("event") != "endpoint":
                _log.warning(
                    "mcp_sse_unexpected_first_event",
                    extra={"server": self.config.name, "event": first_event},
                )
                return []

            endpoint_path = first_event.get("data", "").strip()
            if not endpoint_path:
                return []
            # Resolve relative/absolute endpoint URL
            if endpoint_path.startswith("http"):
                self._sse_message_url = endpoint_path
            else:
                self._sse_message_url = f"{url}{endpoint_path}"

            # Start background SSE reader for subsequent responses
            self._reader_task = asyncio.create_task(self._read_sse_loop(response))

            # POST initialize
            init_response = await self._request_sse(
                "initialize",
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": CLIENT_INFO,
                },
            )
            if not init_response or "error" in init_response:
                return []

            # Send initialized notification
            await self._post_json(
                {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
            )

            # POST tools/list
            tools_response = await self._request_sse("tools/list", {})
            if tools_response and "result" in tools_response:
                self._server_tools = tools_response["result"].get("tools", [])

            return self._server_tools
        except Exception as e:
            _log.warning(
                "mcp_sse_init_failed",
                extra={"server": self.config.name, "error": str(e)},
            )
            await self.close()
            return []

    async def _health_sse(self) -> dict[str, Any]:
        if not self._http_client or not self._sse_message_url:
            return {"healthy": False, "details": "SSE connection not established"}
        try:
            resp = await self._request_sse("ping", {}, timeout=5.0)
            if resp and "result" in resp:
                return {"healthy": True, "details": "ok"}
            return {"healthy": False, "details": f"unexpected ping response: {resp}"}
        except Exception as e:
            return {"healthy": False, "details": str(e)}

    async def _iter_sse_events(self, response: httpx.Response):
        """Parse text/event-stream into dicts with event/data/id fields."""
        buffer = ""
        async for chunk in response.aiter_bytes():
            buffer += chunk.decode("utf-8", errors="replace")
            while "\n\n" in buffer:
                event_str, buffer = buffer.split("\n\n", 1)
                event = self._parse_sse_event(event_str)
                if event:
                    yield event

    @staticmethod
    def _parse_sse_event(text: str) -> dict[str, str] | None:
        """Parse a single SSE event block into a dict.

        Fields: event (default "message"), data, id, retry.
        """
        event: dict[str, str] = {"event": "message"}
        for line in text.split("\n"):
            line = line.strip()
            if not line or line.startswith(":"):
                continue
            if ":" not in line:
                continue
            field, _, value = line.partition(":")
            value = value.lstrip(" ")
            if field == "data":
                if "data" in event:
                    event["data"] += "\n" + value
                else:
                    event["data"] = value
            else:
                event[field] = value
        # Events with no data field are keepalive/heartbeat — skip
        if "data" not in event:
            return None
        return event

    async def _read_sse_loop(self, response: httpx.Response) -> None:
        """Background task: read SSE events and dispatch JSON-RPC responses by id."""
        try:
            async for event in self._iter_sse_events(response):
                data_str = event.get("data", "")
                if not data_str:
                    continue
                msg = self._parse_json_str(data_str)
                if msg is None:
                    continue
                msg_id = msg.get("id")
                if msg_id is None:
                    continue
                fut = self._pending.get(msg_id)
                if fut and not fut.done():
                    fut.set_result(msg)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log.warning(
                "mcp_sse_read_error",
                extra={"server": self.config.name, "error": str(e)},
            )
        finally:
            for fut in self._pending.values():
                if not fut.done():
                    fut.cancel()

    async def _request_sse(
        self, method: str, params: dict, timeout: float = 10.0
    ) -> dict[str, Any] | None:
        req_id = self._new_id()
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[req_id] = fut

        await self._post_json({"jsonrpc": "2.0", "method": method, "params": params, "id": req_id})
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except TimeoutError:
            _log.warning(
                "mcp_sse_timeout",
                extra={"server": self.config.name, "method": method},
            )
            return None
        finally:
            self._pending.pop(req_id, None)

    async def _post_json(self, data: dict) -> None:
        if not self._http_client or not self._sse_message_url:
            return
        try:
            await self._http_client.post(
                self._sse_message_url,
                json=data,
            )
        except Exception as e:
            _log.warning(
                "mcp_sse_post_error",
                extra={"server": self.config.name, "error": str(e)},
            )

    # ── Tool execution ──────────────────────────────────────────────────

    # Tools whose ``description`` or ``name`` matches any of these substrings
    # are treated as potentially long-running and granted the full 120s window.
    # Everything else gets a 15s default — quick read/list/status calls have
    # no business blocking the chat loop for a full minute when the server
    # hangs.
    _LONG_RUNNING_HINTS = (
        "deploy",
        "build",
        "render",
        "compile",
        "install",
        "train",
        "fetch",
        "download",
        "upload",
        "generate",
    )

    def _timeout_for(self, tool_name: str) -> float:
        tool_def = next(
            (t for t in self._server_tools if t.get("name") == tool_name),
            None,
        )
        haystack = tool_name.lower()
        if tool_def:
            haystack += " " + str(tool_def.get("description", "")).lower()
        for hint in self._LONG_RUNNING_HINTS:
            if hint in haystack:
                return 120.0
        return 15.0

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call a tool on the MCP server. Uses the active transport."""
        response = None
        timeout = self._timeout_for(name)
        if self.config.command:
            response = await self._request_stdio(
                "tools/call", {"name": name, "arguments": arguments}, timeout=timeout
            )
        elif self.config.url:
            response = await self._request_sse(
                "tools/call", {"name": name, "arguments": arguments}, timeout=timeout
            )

        if response and "result" in response:
            return self._extract_tool_result(response["result"])
        error = response.get("error", {}) if response else {}
        return f"MCP tool error: {error.get('message', 'unknown')}"

    @staticmethod
    def _extract_tool_result(result: dict) -> str:
        """Extract text content from an MCP tool result."""
        parts = []
        for item in result.get("content", []):
            if item.get("type") == "text":
                parts.append(item["text"])
            elif item.get("type") == "resource":
                parts.append(json.dumps(item.get("resource", {}), ensure_ascii=False))
        return "\n".join(parts) if parts else "Tool returned no content."

    # ── Cleanup ─────────────────────────────────────────────────────────

    async def close(self) -> None:
        for task_attr in ("_reader_task", "_stderr_task"):
            task = getattr(self, task_attr, None)
            if task and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            setattr(self, task_attr, None)

        proc = self._process
        if proc:
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=5)
            except TimeoutError:
                proc.kill()
            except ProcessLookupError:
                pass
            self._process = None

        sse_resp = self._sse_response
        if sse_resp:
            with contextlib.suppress(Exception):
                await sse_resp.aclose()
            self._sse_response = None

        http = self._http_client
        if http:
            with contextlib.suppress(Exception):
                await http.aclose()
            self._http_client = None

        self._sse_message_url = ""

    def get_tool_definitions(self) -> list[dict]:
        return self._server_tools

    # ── JSON helpers ────────────────────────────────────────────────────

    @staticmethod
    def _parse_json_line(line: bytes) -> dict | None:
        try:
            return json.loads(line.decode())  # type: ignore[no-any-return]
        except (json.JSONDecodeError, ValueError):
            return None

    @staticmethod
    def _parse_json_str(text: str) -> dict | None:
        try:
            return json.loads(text)  # type: ignore[no-any-return]
        except (json.JSONDecodeError, ValueError):
            return None


class MCPManager:
    """Manages multiple MCP server connections and registers their tools."""

    def __init__(self, tool_registry: ToolRegistry) -> None:
        self.tool_registry = tool_registry
        self.clients: dict[str, MCPClient] = {}
        self._server_configs: list[MCPServerConfig] = []
        # Agents bound after construction so runtime-added tools propagate into
        # each agent's own (cloned) registry and its cached tool list.
        self._agents: dict[str, Any] = {}

    def bind_agents(self, agents: dict[str, Any]) -> None:
        """Register the live agent map so runtime tool changes reach them."""
        self._agents = agents

    def configure(self, servers: list[dict]) -> None:
        """Configure MCP servers from config data."""
        self._server_configs = [
            MCPServerConfig(
                name=s["name"],
                command=s.get("command"),
                args=s.get("args", []),
                url=s.get("url"),
                env=s.get("env", {}),
                enabled=s.get("enabled", True),
            )
            for s in servers
            if s.get("enabled", True)
        ]

    async def connect_all(self) -> list[str]:
        """Connect to all configured MCP servers and register tools.

        Runs every server's ``initialize()`` in parallel — previously this
        was a serial for-loop, so 3 servers × ~500ms each meant ~1.5s of
        cold start time. Parallel init drops that to the slowest single
        server, typically ~500ms.

        Returns list of "server_name: N tools" strings.
        """
        enabled = [cfg for cfg in self._server_configs if cfg.enabled]
        if not enabled:
            return []

        async def _init_one(cfg: Any) -> tuple[Any, MCPClient, list[dict]] | None:
            client = MCPClient(cfg)
            try:
                tools = await client.initialize()
            except Exception as exc:
                _log.warning(
                    "mcp_init_failed",
                    extra={"server": cfg.name, "error": str(exc)},
                )
                return None
            if not tools:
                return None
            return cfg, client, tools

        # gather() preserves order so the result list matches config order.
        init_results = await asyncio.gather(*(_init_one(cfg) for cfg in enabled))

        results: list[str] = []
        for r in init_results:
            if r is None:
                continue
            cfg, client, tools = r
            self.clients[cfg.name] = client
            count = self._register_mcp_tools(cfg.name, tools)
            results.append(f"{cfg.name}: {count} tools")
        return results

    def _register_mcp_tools(self, server_name: str, mcp_tools: list[dict]) -> int:
        count = 0
        for mt in mcp_tools:
            name = mt.get("name", "")
            description = mt.get("description", "")
            input_schema = mt.get("inputSchema", {})

            if not name:
                continue

            parameters = []
            props = input_schema.get("properties", {}) if input_schema else {}
            required = input_schema.get("required", []) if input_schema else []

            for param_name, param_schema in props.items():
                parameters.append(
                    ToolParameter(
                        name=param_name,
                        type=param_schema.get("type", "string"),
                        description=param_schema.get("description", ""),
                        required=param_name in required,
                    )
                )

            mcp_tool_name = f"mcp_{server_name}_{name}"
            tool = Tool(
                name=mcp_tool_name,
                description=f"[MCP/{server_name}] {description}",
                parameters=parameters,
                handler=self._make_mcp_handler(server_name, name),
            )
            # Register into the base registry AND each agent's cloned registry,
            # so tools added at runtime (after agents cloned the base) actually
            # reach the agents. Refresh happens once after the loop.
            self.tool_registry.register(tool)
            for agent in self._agents.values():
                reg = getattr(agent, "tool_registry", None)
                if reg is not None:
                    reg.register(tool)
            count += 1
        for agent in self._agents.values():
            with contextlib.suppress(Exception):
                agent.refresh_tools()
        return count

    def _unregister_server_tools(self, server_name: str) -> int:
        """Remove all tools belonging to ``server_name`` from every registry."""
        prefix = f"mcp_{server_name}_"
        names = [n for n in self.tool_registry.list_names() if n.startswith(prefix)]
        for n in names:
            self.tool_registry.unregister(n)
            for agent in self._agents.values():
                reg = getattr(agent, "tool_registry", None)
                if reg is not None:
                    reg.unregister(n)
        for agent in self._agents.values():
            with contextlib.suppress(Exception):
                agent.refresh_tools()
        return len(names)

    async def add_server(self, config_dict: dict) -> str:
        """Connect to a new MCP server at runtime and register its tools.

        ``config_dict`` keys: name, command, args, url, env. Returns a human
        message. Reuses the same MCPClient used at startup, so stdio + SSE both
        work. Does NOT persist — callers handle persistence.
        """
        name = str(config_dict.get("name", "")).strip()
        if not name:
            return "Error: server name is required"
        if name in self.clients:
            return f"MCP server '{name}' is already connected"
        if not config_dict.get("command") and not config_dict.get("url"):
            return "Error: provide either 'command' (stdio) or 'url' (SSE)"
        cfg = MCPServerConfig(
            name=name,
            command=config_dict.get("command"),
            args=config_dict.get("args", []),
            url=config_dict.get("url"),
            env=config_dict.get("env", {}),
        )
        client = MCPClient(cfg)
        try:
            tools = await client.initialize()
        except Exception as e:  # noqa: BLE001
            await client.close()
            return f"连接 MCP server '{name}' 失败: {e}"
        if not tools:
            await client.close()
            return f"MCP server '{name}' 未返回任何工具（连接失败或服务无工具）"
        self.clients[name] = client
        self._server_configs.append(cfg)
        count = self._register_mcp_tools(name, tools)
        tool_names = ", ".join(t.get("name", "?") for t in tools[:10])
        return f"✓ 已接入 MCP server '{name}'，注册 {count} 个工具: {tool_names}"

    async def remove_server(self, name: str) -> str:
        """Disconnect an MCP server and unregister its tools."""
        name = (name or "").strip()
        client = self.clients.pop(name, None)
        if client is None:
            return f"未连接 MCP server '{name}'"
        await client.close()
        self._server_configs = [c for c in self._server_configs if c.name != name]
        removed = self._unregister_server_tools(name)
        return f"✓ 已断开 MCP server '{name}'，移除 {removed} 个工具"

    def list_servers(self) -> list[dict]:
        """Return the connected servers and their tool counts."""
        out = []
        for cfg in self._server_configs:
            prefix = f"mcp_{cfg.name}_"
            n = sum(1 for t in self.tool_registry.list_names() if t.startswith(prefix))
            out.append(
                {
                    "name": cfg.name,
                    "transport": "stdio" if cfg.command else "sse",
                    "target": cfg.command or cfg.url or "",
                    "tools": n,
                    "connected": cfg.name in self.clients,
                }
            )
        return out

    def _make_mcp_handler(self, server_name: str, tool_name: str):
        async def handler(**kwargs) -> str:
            client = self.clients.get(server_name)
            if not client:
                return f"Error: MCP server '{server_name}' not connected."
            return await client.call_tool(tool_name, kwargs)

        return handler

    async def close_all(self) -> None:
        # Close every client even if one of them raises during shutdown.
        # Previously a single client.close() exception would leave the rest
        # of the subprocesses orphaned.
        for client in self.clients.values():
            try:
                await client.close()
            except Exception as e:
                _log.warning(
                    "mcp_close_failed",
                    extra={"server": client.config.name, "error": str(e)},
                )
        self.clients.clear()


# ── Runtime self-integration: persistence + active-manager singleton ──────
#
# These let the running agent add/remove MCP servers itself (the "Skyloom
# connects software on its own" capability). Persisted servers live in
# ~/.skyloom/mcp_servers.json and are merged with config.yaml's mcp.servers at
# startup, so a runtime-added server survives a restart.

_ACTIVE_MANAGER: MCPManager | None = None


def set_active_manager(manager: MCPManager | None) -> None:
    global _ACTIVE_MANAGER
    _ACTIVE_MANAGER = manager


def get_active_manager() -> MCPManager | None:
    return _ACTIVE_MANAGER


def _persist_path():
    from weather_agents.core import config as _cfg

    return _cfg.USER_CONFIG_DIR / "mcp_servers.json"


def load_persisted_servers() -> list[dict]:
    """Return runtime-added MCP server configs (empty list if none/unreadable)."""
    p = _persist_path()
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:  # noqa: BLE001
        return []


def save_persisted_server(config_dict: dict) -> None:
    """Append/replace a server entry in the persisted list (by name)."""
    name = str(config_dict.get("name", "")).strip()
    if not name:
        return
    items = [s for s in load_persisted_servers() if s.get("name") != name]
    items.append(config_dict)
    p = _persist_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def remove_persisted_server(name: str) -> None:
    items = [s for s in load_persisted_servers() if s.get("name") != name]
    p = _persist_path()
    if not items and p.exists():
        with contextlib.suppress(Exception):
            p.unlink()
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)

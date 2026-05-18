"""Tests for tool system."""

from __future__ import annotations

import pytest

from weather_agents.core.tool import Tool, ToolParameter, ToolRegistry


class TestTool:
    def test_basic_tool(self):
        tool = Tool(
            name="echo",
            description="Echo input",
            parameters=[
                ToolParameter(name="msg", type="string", description="Message to echo"),
            ],
        )
        assert tool.name == "echo"
        assert len(tool.parameters) == 1

    def test_function_schema(self):
        tool = Tool(
            name="read_file",
            description="Read a file",
            parameters=[
                ToolParameter(name="path", type="string", description="File path"),
                ToolParameter(
                    name="max_lines",
                    type="number",
                    description="Max lines",
                    required=False,
                    default=100,
                ),
            ],
        )
        schema = tool.to_function_schema()
        assert schema["function"]["name"] == "read_file"
        assert "path" in schema["function"]["parameters"]["properties"]
        assert "max_lines" in schema["function"]["parameters"]["properties"]
        assert schema["function"]["parameters"]["required"] == ["path"]

    def test_execute_without_handler_returns_error(self):
        tool = Tool(name="stub", description="No handler")
        import asyncio

        result = asyncio.run(tool.execute())
        assert "has no handler" in result

    def test_execute_calls_handler(self):
        async def my_handler(**kwargs):
            return f"hello {kwargs['name']}"

        tool = Tool(name="greet", description="Greet", handler=my_handler)
        import asyncio

        result = asyncio.run(tool.execute(name="world"))
        assert result == "hello world"

    def test_retry_on_failure(self):
        call_count = 0

        async def flaky_handler(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("transient error")
            return "success"

        tool = Tool(
            name="flaky",
            description="Flaky tool",
            handler=flaky_handler,
            max_retries=3,
            retry_delay=0.01,
        )
        import asyncio

        result = asyncio.run(tool.execute())
        assert result == "success"
        assert call_count == 3

    def test_retry_exhausted(self):
        async def always_fails(**kwargs):
            raise ValueError("always fails")

        tool = Tool(
            name="bad",
            description="Bad tool",
            handler=always_fails,
            max_retries=2,
            retry_delay=0.01,
        )
        import asyncio

        result = asyncio.run(tool.execute())
        assert "Error" in result
        assert "retries" in result


class TestToolRegistry:
    def test_register_and_get(self):
        r = ToolRegistry()
        t = Tool(name="test_tool", description="Test")
        r.register(t)
        assert r.get("test_tool") is t
        assert r.get("nonexistent") is None

    def test_get_tools_by_names(self):
        r = ToolRegistry()
        r.register(Tool(name="a", description="A"))
        r.register(Tool(name="b", description="B"))
        r.register(Tool(name="c", description="C"))

        tools = r.get_tools(["a", "c"])
        assert len(tools) == 2
        assert tools[0].name == "a"
        assert tools[1].name == "c"

    def test_get_all_tools(self):
        r = ToolRegistry()
        r.register(Tool(name="x", description="X"))
        assert len(r.get_tools()) == 1

    def test_schemas(self):
        r = ToolRegistry()
        r.register(
            Tool(
                name="my_tool",
                description="My tool",
                parameters=[
                    ToolParameter(name="p", type="string", description="A param"),
                ],
            )
        )
        schemas = r.get_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "my_tool"

    def test_merge(self):
        r1 = ToolRegistry()
        r1.register(Tool(name="t1", description="T1"))
        r2 = ToolRegistry()
        r2.register(Tool(name="t2", description="T2"))
        r1.merge(r2)
        assert r1.get("t1") is not None
        assert r1.get("t2") is not None

    def test_list_names(self):
        r = ToolRegistry()
        r.register(Tool(name="alpha", description="Alpha"))
        r.register(Tool(name="beta", description="Beta"))
        names = r.list_names()
        assert "alpha" in names
        assert "beta" in names

    def test_override_on_reregister(self):
        r = ToolRegistry()
        t1 = Tool(name="t", description="v1")
        t2 = Tool(name="t", description="v2")
        r.register(t1)
        r.register(t2)
        assert r.get("t").description == "v2"

    def test_unregister(self):
        r = ToolRegistry()
        r.register(Tool(name="temp", description="Temporary"))
        assert r.get("temp") is not None

        removed = r.unregister("temp")
        assert removed is True
        assert r.get("temp") is None

    def test_unregister_nonexistent(self):
        r = ToolRegistry()
        removed = r.unregister("nonexistent")
        assert removed is False


class TestSchemaPreValidation:
    """Tool.execute must short-circuit on bad args without invoking the handler."""

    @pytest.mark.asyncio
    async def test_missing_required_arg_returns_error(self):
        from weather_agents.core.tool import Tool, ToolParameter

        called = {"n": 0}

        async def _h(**_kw):
            called["n"] += 1
            return "ok"

        t = Tool(
            name="needs_path",
            description="x",
            parameters=[ToolParameter(name="path", type="string", description="p")],
            handler=_h,
        )
        result = await t.execute()
        assert "missing required argument 'path'" in result
        assert "needs_path" in result
        assert called["n"] == 0  # handler never invoked

    @pytest.mark.asyncio
    async def test_wrong_type_returns_error(self):
        from weather_agents.core.tool import Tool, ToolParameter

        called = {"n": 0}

        async def _h(**_kw):
            called["n"] += 1
            return "ok"

        t = Tool(
            name="needs_count",
            description="x",
            parameters=[ToolParameter(name="count", type="integer", description="c")],
            handler=_h,
        )
        # dict is not coercible to integer — expect rejection
        result = await t.execute(count={"oops": True})
        assert "wrong type" in result
        assert called["n"] == 0

    @pytest.mark.asyncio
    async def test_string_to_number_coerced(self):
        """LiteLLM tool calls often pass numeric args as strings — accept them."""
        from weather_agents.core.tool import Tool, ToolParameter

        async def _h(**kw):
            return f"got {kw['n']}"

        t = Tool(
            name="numeric",
            description="x",
            parameters=[ToolParameter(name="n", type="number", description="n")],
            handler=_h,
        )
        result = await t.execute(n="3.14")
        assert "got 3.14" in result


class TestSchemaCache:
    def test_schema_is_cached(self):
        from weather_agents.core.tool import Tool, ToolParameter

        t = Tool(
            name="t1",
            description="d",
            parameters=[ToolParameter(name="a", type="string", description="a")],
        )
        s1 = t.to_function_schema()
        s2 = t.to_function_schema()
        # Identical object — the cache is being used.
        assert s1 is s2

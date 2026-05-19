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


class TestToolNameSuggestions:
    """Hallucinated tool names should be matched to real ones via name + description."""

    def test_typo_match_via_difflib(self):
        from weather_agents.core.agent import _suggest_tool_names

        r = ToolRegistry()
        r.register(Tool(name="read_file", description="Read a file"))
        r.register(Tool(name="write_file", description="Write a file"))
        assert "read_file" in _suggest_tool_names("read_fil", r)

    def test_conceptual_match_via_description(self):
        """fetch_page → http_get when only the description shares 'fetch'/'page'."""
        from weather_agents.core.agent import _suggest_tool_names

        r = ToolRegistry()
        r.register(
            Tool(
                name="http_get",
                description="Fetch a web page or HTTP URL. Download article content.",
            )
        )
        r.register(Tool(name="echo", description="Echo input back"))
        assert _suggest_tool_names("fetch_page", r) == ["http_get"]

    def test_no_match_returns_empty(self):
        from weather_agents.core.agent import _suggest_tool_names

        r = ToolRegistry()
        r.register(Tool(name="echo", description="Echo input"))
        assert _suggest_tool_names("xyzzy", r) == []


class TestCacheKeyExtra:
    """Tools with cache_key_extra (e.g. read_file with mtime) must
    invalidate cached results when their side-channel state changes."""

    @pytest.mark.asyncio
    async def test_cache_key_extra_invalidates_on_change(self):
        """Simulate read_file: same args, but the mtime-extra value
        changes between calls — second call must NOT hit the cache."""
        from weather_agents.core.tool import Tool, ToolParameter, _RESULT_STORE

        _RESULT_STORE.clear()

        call_count = {"n": 0, "extra": "v1"}

        async def _handler(**_kw):
            call_count["n"] += 1
            return f"call-{call_count['n']}"

        def _extra(_kw: dict) -> str:
            return call_count["extra"]

        t = Tool(
            name="read_stateful",
            description="x",
            parameters=[ToolParameter(name="path", type="string", description="p")],
            handler=_handler,
            cache_key_extra=_extra,
        )

        # First call — populates cache under extra=v1
        r1 = await t.execute(path="x")
        assert r1 == "call-1"
        # Same args, extra unchanged — cache hit (no new handler call)
        r2 = await t.execute(path="x")
        assert r2 == "call-1"
        assert call_count["n"] == 1
        # Now mutate the side-channel (e.g. file mtime changed)
        call_count["extra"] = "v2"
        r3 = await t.execute(path="x")
        # Must re-execute (different key) and return fresh value
        assert r3 == "call-2"
        assert call_count["n"] == 2

    @pytest.mark.asyncio
    async def test_cache_key_extra_failure_is_safe(self):
        """If cache_key_extra raises, we shouldn't crash — fall back to a
        sentinel that won't accidentally collide with a real value."""
        from weather_agents.core.tool import Tool, _RESULT_STORE

        _RESULT_STORE.clear()

        async def _h(**_):
            return "ok"

        def _broken(_kw: dict) -> str:
            raise RuntimeError("boom")

        t = Tool(
            name="break_extra",
            description="x",
            handler=_h,
            cache_key_extra=_broken,
        )
        # Must not raise — error path uses the "extra_failed" sentinel.
        result = await t.execute()
        assert result == "ok"

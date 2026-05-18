"""Tests for tool subset routing — selecting top-K relevant tools per query."""

from __future__ import annotations

from weather_agents.core.tool import Tool, ToolParameter, ToolRegistry
from weather_agents.core.tool_router import _ALWAYS_INCLUDE, select_relevant_tools


def _make_registry(tool_specs: list[tuple[str, str]]) -> ToolRegistry:
    r = ToolRegistry()
    for name, desc in tool_specs:
        r.register(
            Tool(
                name=name,
                description=desc,
                parameters=[ToolParameter(name="x", type="string", description="x")],
            )
        )
    return r


class TestSelectRelevantTools:
    def test_short_query_returns_full_set(self):
        r = _make_registry([(f"tool_{i}", f"desc {i}") for i in range(20)])
        names = r.list_names()
        # "ok" tokenizes to one short word (filtered) — no signal to filter on
        selected = select_relevant_tools(r, names, "ok", top_k=5)
        assert set(selected) == set(names)

    def test_small_catalog_returns_full_set(self):
        r = _make_registry([("a", "alpha"), ("b", "beta")])
        names = r.list_names()
        selected = select_relevant_tools(r, names, "anything goes here", top_k=12)
        assert set(selected) == set(names)

    def test_caps_at_top_k_for_large_catalog(self):
        r = _make_registry([(f"tool_{i}", f"desc {i}") for i in range(50)])
        names = r.list_names()
        selected = select_relevant_tools(r, names, "find weather data", top_k=10)
        # top_k=10 + always-include set (none of which exist in this registry
        # so they're filtered out by must_present)
        assert len(selected) <= 10

    def test_relevant_tools_score_higher(self):
        r = _make_registry(
            [
                ("read_file", "read a file from disk"),
                ("write_file", "write content to a file"),
                ("send_email", "send an email message"),
                ("fetch_url", "fetch a web URL"),
                ("query_db", "query the database"),
                ("compress", "compress some data"),
                ("encrypt", "encrypt a string"),
                ("decrypt", "decrypt a string"),
                ("hash", "compute hash"),
                ("compile", "compile code"),
                ("parse", "parse data"),
                ("validate", "validate input"),
                ("ping", "ping server"),
                ("trace", "trace network"),
            ]
        )
        names = r.list_names()
        selected = select_relevant_tools(r, names, "read this config file", top_k=3)
        # read_file should be in the top picks because both "read" and "file"
        # match its name tokens.
        assert "read_file" in selected

    def test_must_include_always_present(self):
        r = _make_registry([(f"random_{i}", f"unrelated tool {i}") for i in range(20)])
        r.register(
            Tool(name="critical_tool", description="must always be available"),
        )
        names = r.list_names()
        selected = select_relevant_tools(
            r,
            names,
            "totally unrelated query about weather",
            top_k=3,
            must_include={"critical_tool"},
        )
        assert "critical_tool" in selected

    def test_always_include_set_preserved(self):
        # Verify the module-level always-include set has the infra tools we
        # expect — guards against accidental removal.
        assert "delegate_to" in _ALWAYS_INCLUDE
        assert "read_shared_memory" in _ALWAYS_INCLUDE
        assert "list_shared_memory" in _ALWAYS_INCLUDE

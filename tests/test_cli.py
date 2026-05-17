"""Tests for CLI interactive mode, slash commands, and streaming."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from typer.testing import CliRunner

from weather_agents.cli.main import app

runner = CliRunner()


class TestCLIFlags:
    def test_version_flag(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "Weather Agents" in result.stdout

    def test_help_output(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "chat" in result.stdout


class TestSlashCommandRouting:
    @pytest.mark.asyncio
    async def test_help_command(self):
        with (
            patch("weather_agents.cli.main.create_system_context") as mock_create,
            patch("weather_agents.cli.main.console.input", side_effect=["/help", "/quit"]),
        ):
            mock_ctx = _make_ctx()
            mock_create.return_value = mock_ctx

            with patch("weather_agents.cli.main._print_help") as mock_help:
                await _run_interactive("fog")
                mock_help.assert_called()

    @pytest.mark.asyncio
    async def test_clear_command(self):
        with (
            patch("weather_agents.cli.main.create_system_context") as mock_create,
            patch("weather_agents.cli.main.console.input", side_effect=["/clear", "/quit"]),
        ):
            mock_ctx = _make_ctx()
            mock_create.return_value = mock_ctx

            with patch("weather_agents.cli.main.console.clear") as mock_clear:
                await _run_interactive("fog")
                mock_clear.assert_called()

    @pytest.mark.asyncio
    async def test_status_command(self):
        with (
            patch("weather_agents.cli.main.create_system_context") as mock_create,
            patch("weather_agents.cli.main.console.input", side_effect=["/status", "/quit"]),
        ):
            mock_ctx = _make_ctx()
            mock_create.return_value = mock_ctx

            with patch("weather_agents.cli.main._print_status") as mock_status:
                await _run_interactive("fog")
                mock_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_skills_command(self):
        with (
            patch("weather_agents.cli.main.create_system_context") as mock_create,
            patch("weather_agents.cli.main.console.input", side_effect=["/skills", "/quit"]),
        ):
            mock_ctx = _make_ctx()
            mock_create.return_value = mock_ctx

            with patch("weather_agents.cli.main._print_skills") as mock_skills:
                await _run_interactive("fog")
                mock_skills.assert_called_once()

    @pytest.mark.asyncio
    async def test_version_slash_command(self):
        with (
            patch("weather_agents.cli.main.create_system_context") as mock_create,
            patch("weather_agents.cli.main.console.input", side_effect=["/version", "/quit"]),
        ):
            mock_ctx = _make_ctx()
            mock_create.return_value = mock_ctx

            with patch("weather_agents.cli.main.console.print") as mock_print:
                await _run_interactive("fog")
                # Should print version info
                assert any("Weather Agents" in str(c) for c in mock_print.call_args_list if c.args)


class TestAgentSwitching:
    @pytest.mark.asyncio
    async def test_switch_to_rain(self):
        with (
            patch("weather_agents.cli.main.create_system_context") as mock_create,
            patch("weather_agents.cli.main.console.input", side_effect=["/rain", "/quit"]),
        ):
            mock_ctx = _make_ctx()
            mock_create.return_value = mock_ctx
            await _run_interactive("fog")
            # Should not crash

    @pytest.mark.asyncio
    async def test_switch_all_agents(self):
        for agent_name in ("fog", "rain", "frost", "snow", "dew", "fair"):
            with (
                patch("weather_agents.cli.main.create_system_context") as mock_create,
                patch(
                    "weather_agents.cli.main.console.input", side_effect=[f"/{agent_name}", "/quit"]
                ),
            ):
                mock_ctx = _make_ctx()
                mock_create.return_value = mock_ctx
                await _run_interactive("fog")
                # Should not crash for any agent switch

    @pytest.mark.asyncio
    async def test_switch_to_invalid_shows_help(self):
        with (
            patch("weather_agents.cli.main.create_system_context") as mock_create,
            patch("weather_agents.cli.main.console.input", side_effect=["/unknown", "/quit"]),
        ):
            mock_ctx = _make_ctx()
            mock_create.return_value = mock_ctx

            with patch("weather_agents.cli.main._print_help") as mock_help:
                await _run_interactive("fog")
                mock_help.assert_called()


class TestTaskCommand:
    @pytest.mark.asyncio
    async def test_task_invokes_run_task(self):
        with (
            patch("weather_agents.cli.main.create_system_context") as mock_create,
            patch(
                "weather_agents.cli.main.console.input",
                side_effect=["/task build a website", "/quit"],
            ),
        ):
            mock_ctx = _make_ctx()
            mock_create.return_value = mock_ctx

            with patch("weather_agents.cli.main._run_task") as mock_run:
                await _run_interactive("fog")
                mock_run.assert_called_once_with("build a website", mock_ctx.agent_map)

    @pytest.mark.asyncio
    async def test_task_empty_goal_ignored(self):
        with (
            patch("weather_agents.cli.main.console.input", side_effect=["/task ", "/quit"]),
            patch("weather_agents.cli.main._run_task") as mock_run,
            patch("weather_agents.cli.main.create_system_context") as mock_create,
        ):
            mock_ctx = _make_ctx()
            mock_create.return_value = mock_ctx
            await _run_interactive("fog")
            mock_run.assert_not_called()


class TestMemoryCommands:
    @pytest.mark.asyncio
    async def test_memory_status(self):
        with (
            patch("weather_agents.cli.main.create_system_context") as mock_create,
            patch("weather_agents.cli.main.console.input", side_effect=["/memory", "/quit"]),
        ):
            mock_ctx = _make_ctx()
            mock_create.return_value = mock_ctx

            with patch("weather_agents.cli.main._print_memory_status") as mock_mem:
                await _run_interactive("fog")
                mock_mem.assert_called_once()

    @pytest.mark.asyncio
    async def test_memory_clear(self):
        with (
            patch("weather_agents.cli.main.create_system_context") as mock_create,
            patch("weather_agents.cli.main.console.input", side_effect=["/memory clear", "/quit"]),
        ):
            mock_ctx = _make_ctx()
            mock_create.return_value = mock_ctx
            await _run_interactive("fog")
            # Each agent's memory.clear_short_term should be awaited
            for ag in mock_ctx.agent_map.values():
                ag.memory.clear_short_term.assert_awaited()


class TestCostCommands:
    @pytest.mark.asyncio
    async def test_cost_display(self):
        with (
            patch("weather_agents.cli.main.create_system_context") as mock_create,
            patch("weather_agents.cli.main.console.input", side_effect=["/cost", "/quit"]),
        ):
            mock_ctx = _make_ctx()
            mock_create.return_value = mock_ctx

            with patch("weather_agents.cli.main._print_cost") as mock_cost:
                await _run_interactive("fog")
                mock_cost.assert_called_once()

    @pytest.mark.asyncio
    async def test_cost_reset(self):
        with (
            patch("weather_agents.cli.main.create_system_context") as mock_create,
            patch("weather_agents.cli.main.console.input", side_effect=["/cost reset", "/quit"]),
        ):
            mock_ctx = _make_ctx()
            mock_create.return_value = mock_ctx
            await _run_interactive("fog")
            mock_ctx.llm.reset_usage_stats.assert_called_once()


class TestHistoryCommand:
    @pytest.mark.asyncio
    async def test_history(self):
        with (
            patch("weather_agents.cli.main.create_system_context") as mock_create,
            patch("weather_agents.cli.main.console.input", side_effect=["/history", "/quit"]),
        ):
            mock_ctx = _make_ctx()
            mock_create.return_value = mock_ctx

            with patch("weather_agents.cli.main._print_history") as mock_hist:
                await _run_interactive("fog")
                mock_hist.assert_called_once()


class TestMCPCommand:
    @pytest.mark.asyncio
    async def test_mcp_status(self):
        with (
            patch("weather_agents.cli.main.create_system_context") as mock_create,
            patch("weather_agents.cli.main.console.input", side_effect=["/mcp", "/quit"]),
        ):
            mock_ctx = _make_ctx()
            mock_create.return_value = mock_ctx

            with patch("weather_agents.cli.main._print_mcp_status") as mock_mcp:
                await _run_interactive("fog")
                mock_mcp.assert_called_once()


class TestModelCommand:
    @pytest.mark.asyncio
    async def test_model_list(self):
        with (
            patch("weather_agents.cli.main.create_system_context") as mock_create,
            patch("weather_agents.cli.main.console.input", side_effect=["/model list", "/quit"]),
        ):
            mock_ctx = _make_ctx()
            mock_create.return_value = mock_ctx

            with patch("weather_agents.cli.main._handle_model_command") as mock_model:
                await _run_interactive("fog")
                mock_model.assert_called_once()


class TestSkillActivation:
    @pytest.mark.asyncio
    async def test_use_skill(self):
        with (
            patch("weather_agents.cli.main.create_system_context") as mock_create,
            patch(
                "weather_agents.cli.main.console.input", side_effect=["/use code_reviewer", "/quit"]
            ),
        ):
            mock_ctx = _make_ctx()
            mock_create.return_value = mock_ctx
            await _run_interactive("frost")
            mock_ctx.agent_map["frost"].activate_skill.assert_called_once_with("code_reviewer")

    @pytest.mark.asyncio
    async def test_use_unknown_skill(self):
        with (
            patch("weather_agents.cli.main.create_system_context") as mock_create,
            patch("weather_agents.cli.main.console.input", side_effect=["/use badskill", "/quit"]),
        ):
            mock_ctx = _make_ctx()
            mock_ctx.agent_map["frost"].activate_skill.return_value = False
            mock_create.return_value = mock_ctx
            await _run_interactive("frost")
            mock_ctx.agent_map["frost"].activate_skill.assert_called_once_with("badskill")

    @pytest.mark.asyncio
    async def test_deactivate_skills(self):
        with (
            patch("weather_agents.cli.main.create_system_context") as mock_create,
            patch("weather_agents.cli.main.console.input", side_effect=["/deactivate", "/quit"]),
        ):
            mock_ctx = _make_ctx()
            mock_create.return_value = mock_ctx
            await _run_interactive("frost")
            mock_ctx.agent_map["frost"].deactivate_all_skills.assert_called_once()


class TestExitCommands:
    @pytest.mark.asyncio
    async def test_all_exit_commands(self):
        for cmd in ("/quit", "/exit", "/q"):
            with (
                patch("weather_agents.cli.main.create_system_context") as mock_create,
                patch("weather_agents.cli.main.console.input", side_effect=[cmd]),
            ):
                mock_ctx = _make_ctx()
                mock_create.return_value = mock_ctx
                await _run_interactive("fog")
                # Should exit cleanly

    @pytest.mark.asyncio
    async def test_empty_input_continues(self):
        with (
            patch("weather_agents.cli.main.create_system_context") as mock_create,
            patch("weather_agents.cli.main.console.input", side_effect=["", "  ", "/quit"]),
        ):
            mock_ctx = _make_ctx()
            mock_create.return_value = mock_ctx
            await _run_interactive("fog")
            # Empty inputs should be ignored

    @pytest.mark.asyncio
    async def test_slash_without_match_shows_help(self):
        with (
            patch("weather_agents.cli.main.create_system_context") as mock_create,
            patch("weather_agents.cli.main.console.input", side_effect=["/nonexistent", "/quit"]),
        ):
            mock_ctx = _make_ctx()
            mock_create.return_value = mock_ctx

            with patch("weather_agents.cli.main._print_help") as mock_help:
                await _run_interactive("fog")
                mock_help.assert_called()


class TestKeyboardInterrupt:
    @pytest.mark.asyncio
    async def test_ctrl_c_exits(self):
        with (
            patch("weather_agents.cli.main.create_system_context") as mock_create,
            patch("weather_agents.cli.main.console.input", side_effect=KeyboardInterrupt),
        ):
            mock_ctx = _make_ctx()
            mock_create.return_value = mock_ctx
            await _run_interactive("fog")

    @pytest.mark.asyncio
    async def test_eof_exits(self):
        with (
            patch("weather_agents.cli.main.create_system_context") as mock_create,
            patch("weather_agents.cli.main.console.input", side_effect=EOFError),
        ):
            mock_ctx = _make_ctx()
            mock_create.return_value = mock_ctx
            await _run_interactive("fog")


class TestStreamingChat:
    @pytest.mark.asyncio
    async def test_streaming_content_rendered(self):
        with (
            patch("weather_agents.cli.main.create_system_context") as mock_create,
            patch("weather_agents.cli.main.console.input", side_effect=["hello", "/quit"]),
        ):
            mock_ctx = _make_ctx()
            mock_create.return_value = mock_ctx

            mock_ctx.agent_map["fog"].chat_stream = lambda _msg: _async_iter(
                [
                    {"type": "content", "text": "Hello"},
                    {"type": "content", "text": " world!"},
                    {"type": "done"},
                ]
            )

            with patch("weather_agents.cli.main.Live") as mock_live_cls:
                mock_live = MagicMock()
                mock_live_cls.return_value = mock_live

                await _run_interactive("fog")
                mock_live.start.assert_called_once()
                # Content updates via Live
                update_calls = mock_live.update.call_args_list
                assert len(update_calls) >= 2

    @pytest.mark.asyncio
    async def test_streaming_tool_status(self):
        with (
            patch("weather_agents.cli.main.create_system_context") as mock_create,
            patch("weather_agents.cli.main.console.input", side_effect=["do x", "/quit"]),
            patch("weather_agents.cli.main.Live") as mock_live_cls,
        ):
            mock_ctx = _make_ctx()
            mock_create.return_value = mock_ctx
            mock_live = MagicMock()
            mock_live_cls.return_value = mock_live

            mock_ctx.agent_map["fog"].chat_stream = lambda _msg: _async_iter(
                [
                    {"type": "content", "text": "Checking..."},
                    {"type": "tool_status", "label": "read_file /tmp/x"},
                    {"type": "content", "text": " Done."},
                    {"type": "done"},
                ]
            )

            await _run_interactive("fog")
            mock_live.start.assert_called_once()
            # Tool status included in updates
            update_calls = mock_live.update.call_args_list
            assert len(update_calls) >= 3

    @pytest.mark.asyncio
    async def test_streaming_interrupted(self):
        with (
            patch("weather_agents.cli.main.create_system_context") as mock_create,
            patch("weather_agents.cli.main.console.input", side_effect=["long query", "/quit"]),
        ):
            mock_ctx = _make_ctx()

            async def _raise_interrupt(_msg):
                raise KeyboardInterrupt
                yield  # makes it an async generator

            mock_ctx.agent_map["fog"].chat_stream = _raise_interrupt
            mock_create.return_value = mock_ctx
            # Should handle interrupt gracefully
            await _run_interactive("fog")


class TestRunTask:
    @pytest.mark.asyncio
    async def test_complex_goal_uses_orchestrate(self):
        """Multi-step goals go through factory.orchestrate_task."""
        from weather_agents.cli.main import _run_task

        mock_ctx = _make_ctx()
        goal = "先帮我分析这段代码，然后重构它，最后写测试"

        with patch("weather_agents.core.factory.orchestrate_task") as mock_orch:
            mock_orch.return_value = ([], [], "no tasks")
            await _run_task(goal, agents=mock_ctx.agent_map)
            mock_orch.assert_called_once()
            assert mock_orch.call_args[0][0] == goal

    @pytest.mark.asyncio
    async def test_simple_goal_skips_orchestrate(self):
        """Single-purpose goals bypass orchestration — go straight to one agent."""
        from weather_agents.cli.main import _run_task

        mock_ctx = _make_ctx()

        with patch("weather_agents.core.factory.orchestrate_task") as mock_orch:
            await _run_task("你好", agents=mock_ctx.agent_map)
            mock_orch.assert_not_called()


# ── Helpers ────────────────────────────────────────────────────────────


def _make_ctx():
    """Create a minimal SystemContext mock."""
    ctx = Mock()
    ctx.config.llm.default_model = "deepseek-v4"
    ctx.agent_map = _make_agent_map()
    ctx.llm = Mock()
    ctx.llm.reset_usage_stats = Mock()
    ctx.llm.get_usage_stats = Mock(return_value={})
    ctx.llm.get_total_cost = Mock(return_value=0.0)
    ctx.init_all = AsyncMock()
    ctx.close_all = AsyncMock()
    ctx.mcp_status = []
    ctx.bus = Mock()
    ctx.bus.get_history = Mock(return_value=[])
    return ctx


def _make_agent_map():
    """Create a minimal agent map with mocks for all agents."""
    from weather_agents.core.agent import AgentState

    agents = {}
    for name in ("fog", "rain", "frost", "snow", "dew", "fair"):
        ag = Mock()
        ag.name = name
        ag.display_name = {
            "fog": "雾",
            "rain": "雨",
            "frost": "霜",
            "snow": "雪",
            "dew": "露",
            "fair": "晴",
        }[name]
        ag.emoji = {
            "fog": "~~",
            "rain": "//",
            "frost": "**",
            "snow": "..",
            "dew": ",,",
            "fair": "**",
        }[name]
        ag.state = AgentState.IDLE
        ag.chat = AsyncMock(return_value="mock response")
        ag.chat_stream = lambda _msg, _ag=ag: _async_iter(
            [
                {"type": "content", "text": f"mock response from {_ag.name}"},
                {"type": "done"},
            ]
        )
        ag.activate_skill = Mock(return_value=True)
        ag.deactivate_all_skills = Mock()
        ag._active_skills = set()
        ag._skills = []
        ag.get_available_skills = Mock(return_value=[])
        ag.get_status = Mock(
            return_value={
                "name": name,
                "display_name": ag.display_name,
                "emoji": ag.emoji,
                "specialty": "test",
            }
        )
        ag.memory = Mock()
        ag.memory.short_term = []
        ag.memory.clear_short_term = AsyncMock()
        ag.memory.get_stats = Mock(
            return_value={
                "short_term_count": 0,
                "working_count": 0,
                "long_term_count": 0,
            }
        )
        agents[name] = ag
    return agents


async def _run_interactive(agent_name: str = "fog"):
    from weather_agents.cli.main import _interactive

    await _interactive(agent_name)


def _async_iter(events):
    """Return an awaitable that acts as an async iterator yielding events."""

    async def _gen():
        for e in events:
            yield e

    return _gen()


# ── Display builder tests ─────────────────────────────────────────────────────


def _make_display_agent(name: str = "fog"):
    """Minimal agent mock for display-builder functions."""
    from weather_agents.core.factory import AGENT_COLORS, AGENT_EMOJI

    ag = Mock()
    ag.name = name
    ag.display_name = name.capitalize()
    ag.emoji = AGENT_EMOJI.get(name, "?")
    ag.color = AGENT_COLORS.get(name, "white")
    ag.context_usage = Mock(
        return_value={
            "model": "deepseek/deepseek-chat",
            "pct": 12,
            "estimated_tokens": 1200,
            "max_tokens": 128000,
            "message_count": 5,
        }
    )
    return ag


def _make_display_ctx():
    ctx = Mock()
    ctx.llm.get_total_cost = Mock(return_value=0.0042)
    ctx.llm.get_usage_stats = Mock(return_value={})
    ctx.config.mcp.servers = []
    ctx.mcp_status = []
    ctx.bus.get_history = Mock(return_value=[])
    ctx.agent_map = _make_agent_map()
    return ctx


class TestBuildStreamDisplay:
    def test_returns_table_empty(self):
        from rich.table import Table

        from weather_agents.cli.main import _build_stream_display

        ag = _make_display_agent()
        result = _build_stream_display(ag, "", "")
        assert isinstance(result, Table)

    def test_with_content_and_status(self):
        from rich.table import Table

        from weather_agents.cli.main import _build_stream_display

        ag = _make_display_agent("rain")
        result = _build_stream_display(ag, "reading file…", "Hello world")
        assert isinstance(result, Table)

    def test_with_activities(self):
        from rich.table import Table

        from weather_agents.cli.main import _build_stream_display

        ag = _make_display_agent("frost")
        result = _build_stream_display(ag, "", "content")
        assert isinstance(result, Table)

    def test_all_agents(self):
        from weather_agents.cli.main import _build_stream_display

        for name in ("fog", "rain", "frost", "snow", "dew", "fair"):
            ag = _make_display_agent(name)
            _build_stream_display(ag, "status", "md text")


class TestBuildResponsePanel:
    def test_returns_panel(self):
        from rich.panel import Panel

        from weather_agents.cli.main import _build_response_panel

        ag = _make_display_agent()
        result = _build_response_panel(ag, "# Hello\nWorld", 1.23)
        assert isinstance(result, Panel)

    def test_interrupted_flag(self):
        from rich.panel import Panel

        from weather_agents.cli.main import _build_response_panel

        ag = _make_display_agent("snow")
        result = _build_response_panel(ag, "Partial answer", 0.5, interrupted=True)
        assert isinstance(result, Panel)

    def test_all_agents(self):
        from weather_agents.cli.main import _build_response_panel

        for name in ("fog", "rain", "frost", "snow", "dew", "fair"):
            ag = _make_display_agent(name)
            _build_response_panel(ag, "response text", 2.0)


class TestBuildStatusLine:
    def test_returns_text(self):
        from rich.text import Text

        from weather_agents.cli.main import _build_status_line

        ag = _make_display_agent()
        ctx = _make_display_ctx()
        result = _build_status_line(ag, ctx)
        assert isinstance(result, Text)

    def test_high_context_usage(self):
        from rich.text import Text

        from weather_agents.cli.main import _build_status_line

        ag = _make_display_agent()
        ag.context_usage = Mock(
            return_value={
                "model": "claude-opus-4",
                "pct": 85,
                "estimated_tokens": 85000,
                "max_tokens": 100000,
                "message_count": 42,
            }
        )
        ctx = _make_display_ctx()
        ctx.llm.get_total_cost = Mock(return_value=0.25)
        result = _build_status_line(ag, ctx)
        assert isinstance(result, Text)

    def test_exception_returns_empty_text(self):
        from rich.text import Text

        from weather_agents.cli.main import _build_status_line

        ag = _make_display_agent()
        ag.context_usage = Mock(side_effect=RuntimeError("no ctx"))
        ctx = _make_display_ctx()
        result = _build_status_line(ag, ctx)
        assert isinstance(result, Text)

    def test_small_token_count(self):
        from rich.text import Text

        from weather_agents.cli.main import _build_status_line

        ag = _make_display_agent()
        ag.context_usage = Mock(
            return_value={
                "model": "gpt-4o",
                "pct": 0,
                "estimated_tokens": 50,  # < 1000, uses raw format
                "max_tokens": 8192,
                "message_count": 1,
            }
        )
        ctx = _make_display_ctx()
        result = _build_status_line(ag, ctx)
        assert isinstance(result, Text)


class TestPrintWelcome:
    def test_smoke(self):
        from weather_agents.cli.main import _print_welcome

        with patch("weather_agents.cli.main.console.print"):
            _print_welcome("deepseek/deepseek-chat")

    def test_with_workspace(self):
        from weather_agents.cli.main import _print_welcome

        with patch("weather_agents.cli.main.console.print"):
            _print_welcome("gpt-4o", "/home/user/projects/myapp")

    def test_long_workspace_path_truncated(self):
        from weather_agents.cli.main import _print_welcome

        with patch("weather_agents.cli.main.console.print"):
            _print_welcome("claude-opus-4", "/very/long/path/" + "x" * 60)


class TestPrintHelp:
    def test_smoke(self):
        from unittest.mock import MagicMock

        from weather_agents.cli.main import _print_help

        ctx = MagicMock()
        ctx.config.llm.language = "zh"
        with patch("weather_agents.cli.main.console.print"):
            _print_help(ctx)


class TestPrintStatus:
    def test_smoke(self):
        from weather_agents.cli.main import _print_status

        agents = _make_agent_map()
        # Patch get_status to return full data
        for name, ag in agents.items():
            ag.get_status.return_value = {
                "name": name,
                "display_name": name.capitalize(),
                "emoji": "~~",
                "specialty": "test",
                "state": "idle",
                "skills": [{"name": "code_reviewer", "active": True}],
                "usage": {"calls": 2, "prompt_tokens": 1000, "completion_tokens": 200},
            }
        with patch("weather_agents.cli.main.console.print"):
            _print_status(agents)

    def test_no_active_skills(self):
        from weather_agents.cli.main import _print_status

        agents = _make_agent_map()
        for name, ag in agents.items():
            ag.get_status.return_value = {
                "name": name,
                "display_name": name.capitalize(),
                "emoji": "~~",
                "specialty": "test",
                "state": "busy",
                "skills": [],
                "usage": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
            }
        with patch("weather_agents.cli.main.console.print"):
            _print_status(agents)


class TestPrintCost:
    def test_no_usage(self):
        from weather_agents.cli.main import _print_cost

        ctx = _make_display_ctx()
        ctx.llm.get_usage_stats = Mock(return_value={})
        with patch("weather_agents.cli.main.console.print"):
            _print_cost(ctx)

    def test_with_usage(self):
        from weather_agents.cli.main import _print_cost

        ctx = _make_display_ctx()
        ctx.llm.get_usage_stats = Mock(
            return_value={
                "fog": {"calls": 3, "prompt_tokens": 2000, "completion_tokens": 500, "cost": 0.005},
                "rain": {"calls": 1, "prompt_tokens": 800, "completion_tokens": 200, "cost": 0.12},
            }
        )
        with patch("weather_agents.cli.main.console.print"):
            _print_cost(ctx)


class TestPrintHistory:
    def test_no_events(self):
        from weather_agents.cli.main import _print_history

        ctx = _make_display_ctx()
        ctx.bus.get_history = Mock(return_value=[])
        with patch("weather_agents.cli.main.console.print"):
            _print_history(ctx)

    def test_with_events(self):
        from datetime import datetime

        from weather_agents.cli.main import _print_history

        ctx = _make_display_ctx()
        evt = Mock()
        evt.timestamp = datetime(2026, 5, 15, 10, 30, 0)
        evt.type.value = "tool_call"
        evt.source = "fog"
        evt.data = {"tool": "read_file", "args": {"path": "/tmp/x"}}
        ctx.bus.get_history = Mock(return_value=[evt])
        with patch("weather_agents.cli.main.console.print"):
            _print_history(ctx)


class TestSummarizeEvent:
    def test_tool_call(self):
        from weather_agents.cli.main import _summarize_event

        result = _summarize_event(
            "tool_call", {"tool": "read_file", "args": {"path": "/tmp/x", "encoding": "utf-8"}}
        )
        assert "read_file" in result

    def test_llm_call(self):
        from weather_agents.cli.main import _summarize_event

        result = _summarize_event(
            "llm_call",
            {"model": "deepseek/chat", "usage": {"prompt_tokens": 100, "completion_tokens": 50}},
        )
        assert "deepseek" in result

    def test_state_change(self):
        from weather_agents.cli.main import _summarize_event

        result = _summarize_event("state_change", {"old_state": "idle", "new_state": "busy"})
        assert "idle" in result and "busy" in result

    def test_unknown_type(self):
        from weather_agents.cli.main import _summarize_event

        result = _summarize_event("unknown_event", {"key": "value"})
        assert isinstance(result, str)

    def test_empty_data(self):
        from weather_agents.cli.main import _summarize_event

        assert _summarize_event("tool_call", None) == ""
        assert _summarize_event("tool_call", {}) == ""


class TestPrintMcpStatus:
    def test_no_servers(self):
        from weather_agents.cli.main import _print_mcp_status

        ctx = _make_display_ctx()
        ctx.config.mcp.servers = []
        with patch("weather_agents.cli.main.console.print"):
            _print_mcp_status(ctx)

    def test_with_connected_server(self):
        from weather_agents.cli.main import _print_mcp_status

        ctx = _make_display_ctx()
        ctx.config.mcp.servers = [{"name": "filesystem", "command": "mcp-fs", "enabled": True}]
        ctx.mcp_status = ["filesystem: 12 tools"]
        with patch("weather_agents.cli.main.console.print"):
            _print_mcp_status(ctx)

    def test_with_disabled_and_disconnected(self):
        from weather_agents.cli.main import _print_mcp_status

        ctx = _make_display_ctx()
        ctx.config.mcp.servers = [
            {"name": "disabled_srv", "command": "x", "enabled": False},
            {"name": "pending_srv", "enabled": True},  # no command → sse
        ]
        ctx.mcp_status = []
        with patch("weather_agents.cli.main.console.print"):
            _print_mcp_status(ctx)


class TestPrintSkills:
    def test_no_skills(self):
        from weather_agents.cli.main import _print_skills

        ag = _make_display_agent()
        ag.get_available_skills = Mock(return_value=[])
        with patch("weather_agents.cli.main.console.print"):
            _print_skills(ag)

        with patch("weather_agents.cli.main.console.print"):
            _print_skills(ag)


class TestFormatCost:
    def test_zero_cost(self):
        from weather_agents.cli.main import _format_cost

        assert _format_cost(0.0) == "$0"

    def test_tiny_cost_cents(self):
        from weather_agents.cli.main import _format_cost

        assert _format_cost(0.0005) == "0.05¢"

    def test_small_cost_four_decimals(self):
        from weather_agents.cli.main import _format_cost

        # 0.0123 is >= 0.01 → $0.0123
        assert _format_cost(0.0123) == "$0.0123"

    def test_dollar_cost_two_decimals(self):
        from weather_agents.cli.main import _format_cost

        assert _format_cost(1.5) == "$1.50"

    def test_large_cost(self):
        from weather_agents.cli.main import _format_cost

        assert _format_cost(123.456) == "$123.46"


class TestShouldAutoContinue:
    def test_cjk_keyword_continues(self):
        from weather_agents.cli.main import _should_auto_continue

        assert _should_auto_continue("接下来我们继续处理下一个任务")

    def test_ascii_keyword_word_boundary(self):
        from weather_agents.cli.main import _should_auto_continue

        assert _should_auto_continue("Let's continue with the next step")

    def test_no_false_positive_context(self):
        from weather_agents.cli.main import _should_auto_continue

        assert not _should_auto_continue("Check the context object for details")

    def test_no_false_positive_discontinue(self):
        from weather_agents.cli.main import _should_auto_continue

        assert not _should_auto_continue("We should discontinue this approach")

    def test_no_keyword_returns_false(self):
        from weather_agents.cli.main import _should_auto_continue

        assert not _should_auto_continue("The task is complete. All done.")

    def test_only_recent_tail_checked(self):
        from weather_agents.cli.main import _should_auto_continue

        # Keyword buried 7+ lines back should not trigger (only last ~6 lines inspected)
        text = "继续\nline1\nline2\nline3\nline4\nline5\nline6\nThis is the final answer."
        assert not _should_auto_continue(text)

    def test_ongoing_keyword(self):
        from weather_agents.cli.main import _should_auto_continue

        assert _should_auto_continue("This is an ongoing task")

    def test_remaining_keyword(self):
        from weather_agents.cli.main import _should_auto_continue

        assert _should_auto_continue("The remaining items are:")

    def test_empty_text_returns_false(self):
        from weather_agents.cli.main import _should_auto_continue

        assert not _should_auto_continue("")

    def test_no_false_positive_cjk(self):
        """CJK keyword '继续' does not appear, so should not continue."""
        from weather_agents.cli.main import _should_auto_continue

        assert not _should_auto_continue("所有任务已完成，无需额外操作。")

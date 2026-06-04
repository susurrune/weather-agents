"""Tests for the computer-operation tools (pure logic + mocked subprocess)."""

from __future__ import annotations

import pytest

from weather_agents.core.tool import ToolRegistry
from weather_agents.tools import computer


@pytest.fixture
def captured_run(monkeypatch):
    """Capture argv passed to computer._run and return a canned result."""
    calls: list[list[str]] = []

    async def fake_run(argv, timeout=60.0):
        calls.append(list(argv))
        return 0, "ok-output", ""

    monkeypatch.setattr(computer, "_run", fake_run)
    return calls


# ── Registration ──


def test_register_adds_expected_tools():
    reg = ToolRegistry()
    computer.register_computer_tools(reg)
    names = set(reg.list_names())
    for expected in (
        "launch_app",
        "open_path",
        "browser_open",
        "list_installed_apps",
        "system_info",
        "system_diagnose",
        "list_processes",
        "kill_process",
        "package_manager",
        "service_control",
    ):
        assert expected in names


def test_dangerous_tools_are_flagged():
    reg = ToolRegistry()
    computer.register_computer_tools(reg)
    for name in ("kill_process", "package_manager", "service_control"):
        assert reg.get(name).dangerous is True
    # Read-only / launch tools are not gated.
    for name in ("system_info", "list_processes", "launch_app"):
        assert reg.get(name).dangerous is False


# ── Input validation ──


@pytest.mark.asyncio
async def test_launch_app_requires_name():
    assert "required" in (await computer._launch_app(name="")).lower()


@pytest.mark.asyncio
async def test_open_path_missing_file():
    out = await computer._open_path(target="/no/such/path/xyz123")
    assert "not found" in out.lower()


@pytest.mark.asyncio
async def test_kill_process_requires_target():
    assert "required" in (await computer._kill_process(target="")).lower()


@pytest.mark.asyncio
async def test_package_manager_validates_action():
    out = await computer._package_manager(action="frobnicate", name="x")
    assert "action must be one of" in out


@pytest.mark.asyncio
async def test_package_manager_install_requires_name():
    out = await computer._package_manager(action="install", name="")
    assert "requires a package name" in out


@pytest.mark.asyncio
async def test_service_control_validates():
    assert "action must be one of" in await computer._service_control(action="bogus", name="x")
    assert "required" in (await computer._service_control(action="start", name="")).lower()


# ── Browser ──


@pytest.mark.asyncio
async def test_browser_open_normalizes_scheme(monkeypatch):
    opened: list[str] = []
    monkeypatch.setattr(computer.webbrowser, "open", lambda u: opened.append(u))
    out = await computer._browser_open(url="example.com")
    assert opened == ["https://example.com"]
    assert "example.com" in out


# ── Package manager detection + argv construction ──


def test_detect_pkg_manager(monkeypatch):
    monkeypatch.setattr(computer, "_SYSTEM", "Linux")
    monkeypatch.setattr(
        computer.shutil, "which", lambda n: "/usr/bin/apt-get" if n == "apt-get" else None
    )
    mgr = computer._detect_pkg_manager()
    assert mgr is not None
    assert mgr[0] == "apt-get"


def test_detect_pkg_manager_none(monkeypatch):
    monkeypatch.setattr(computer, "_SYSTEM", "Linux")
    monkeypatch.setattr(computer.shutil, "which", lambda n: None)
    assert computer._detect_pkg_manager() is None


@pytest.mark.asyncio
async def test_package_manager_builds_argv(monkeypatch, captured_run):
    monkeypatch.setattr(computer, "_SYSTEM", "Linux")
    monkeypatch.setattr(
        computer.shutil, "which", lambda n: "/usr/bin/apt-get" if n == "apt-get" else None
    )
    out = await computer._package_manager(action="install", name="htop")
    assert captured_run, "expected _run to be called"
    argv = captured_run[0]
    assert argv[0] == "apt-get"
    assert "install" in argv and "htop" in argv
    assert "✓" in out


@pytest.mark.asyncio
async def test_package_manager_no_manager(monkeypatch):
    monkeypatch.setattr(computer, "_SYSTEM", "Linux")
    monkeypatch.setattr(computer.shutil, "which", lambda n: None)
    out = await computer._package_manager(action="install", name="htop")
    assert "未检测到包管理器" in out


# ── system_info / diagnose smoke ──


@pytest.mark.asyncio
async def test_system_info_runs():
    out = await computer._system_info()
    assert "系统:" in out
    assert "CPU" in out


@pytest.mark.asyncio
async def test_system_diagnose_runs():
    out = await computer._system_diagnose()
    assert "系统诊断" in out

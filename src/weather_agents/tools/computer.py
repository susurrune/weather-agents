"""Computer-operation tools — cross-platform OS control for Skyloom.

Gives agents (chiefly 露/dew) the ability to *operate the machine*: launch apps,
open files/URLs, inspect and diagnose the system, manage processes and services,
and install/uninstall software through the platform's package manager.

Design notes (WHY):
- Cross-platform by detection, not by assuming one OS. Each handler branches on
  ``platform.system()`` and degrades to a clear message where a capability has
  no equivalent, rather than failing obscurely.
- We reuse the existing security posture: state-changing operations
  (kill/install/uninstall/service control) are registered with ``dangerous=True``
  so they flow through the agent's approval gate. Read-only inspection is not
  gated. We never shell out through a real shell (no pipes/globs); commands run
  as argv lists, matching shell_exec's hardening.
- ``psutil`` is optional. When present we get rich process/memory info; without
  it we fall back to stdlib (os/shutil/platform) so the tools still work in a
  bare install.
"""

from __future__ import annotations

import asyncio
import os
import platform
import shutil
import webbrowser

from weather_agents.core.tool import Tool, ToolParameter, ToolRegistry

_SYSTEM = platform.system()  # "Windows" | "Darwin" | "Linux"
_MAX_OUT = 8000


def _truncate(text: str, limit: int = _MAX_OUT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…(truncated, {len(text) - limit} more chars)"


async def _run(argv: list[str], timeout: float = 60.0) -> tuple[int, str, str]:
    """Run an argv list (no shell) and return (returncode, stdout, stderr)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return 127, "", f"command not found: {argv[0]}"
    except OSError as e:
        return 126, "", f"not executable: {e}"
    try:
        out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, "", f"timed out after {timeout}s"
    out = out_b.decode("utf-8", errors="replace") if out_b else ""
    err = err_b.decode("utf-8", errors="replace") if err_b else ""
    return (proc.returncode if proc.returncode is not None else 0), out, err


# ── App launching / opening ──────────────────────────────────────────────


async def _launch_app(name: str = "", args: str = "", **kwargs) -> str:
    """Launch a desktop application by name (or path)."""
    name = (name or "").strip()
    if not name:
        return "Error: app name is required"
    # Reject cmd.exe / shell metacharacters — the Windows path uses cmd /c start
    # which would interpret &, |, ^, <, >, % as command separators/operators
    # even when passed via argv list (create_subprocess_exec joins them into a
    # command-line string that cmd.exe then re-parses).
    for ch in name:
        if ch in "&|<>^%\n\r\t":
            return f"Error: unsafe character {ch!r} in app name"
    import shlex

    extra = shlex.split(args) if args else []
    try:
        if _SYSTEM == "Windows":
            # Resolve via App Paths / PATH first, falling back to start.
            exe = shutil.which(name)
            if exe:
                proc = await asyncio.create_subprocess_exec(
                    exe,
                    *extra,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                    start_new_session=True,
                )
                return f"已启动 {name} (pid {proc.pid})"
            # Fallback: cmd /c start — only for names shutil.which couldn't
            # resolve (e.g. "Microsoft Edge" registered under App Paths).
            rc, out, err = await _run(["cmd", "/c", "start", "", name, *extra])
        elif _SYSTEM == "Darwin":
            rc, out, err = await _run(["open", "-a", name, *extra])
        else:  # Linux
            exe = shutil.which(name)
            if exe:
                proc = await asyncio.create_subprocess_exec(
                    exe,
                    *extra,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                    start_new_session=True,
                )
                return f"已启动 {name} (pid {proc.pid})"
            rc, out, err = await _run(["gtk-launch", name, *extra])
        if rc == 0:
            return f"已启动 {name}"
        return f"启动失败 (exit {rc}): {err or out or 'unknown'}"
    except Exception as e:  # noqa: BLE001
        return f"启动失败: {e}"


async def _open_path(target: str = "", **kwargs) -> str:
    """Open a file, folder, or URL in the default handler."""
    target = (target or "").strip()
    if not target:
        return "Error: target is required"
    is_url = target.startswith(("http://", "https://", "mailto:", "ftp://"))
    if not is_url:
        expanded = os.path.expanduser(target)
        if not os.path.exists(expanded):
            return f"Error: path not found: {target}"
        target = expanded
    try:
        if _SYSTEM == "Windows":
            if is_url:
                webbrowser.open(target)
            else:
                os.startfile(target)  # type: ignore[attr-defined]  # Windows-only
            return f"已打开 {target}"
        opener = "open" if _SYSTEM == "Darwin" else "xdg-open"
        rc, out, err = await _run([opener, target])
        return f"已打开 {target}" if rc == 0 else f"打开失败: {err or out}"
    except Exception as e:  # noqa: BLE001
        return f"打开失败: {e}"


async def _browser_open(url: str = "", **kwargs) -> str:
    """Open a URL in the default web browser."""
    url = (url or "").strip()
    if not url:
        return "Error: url is required"
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        webbrowser.open(url)
        return f"已在浏览器打开 {url}"
    except Exception as e:  # noqa: BLE001
        return f"打开失败: {e}"


async def _list_installed_apps(filter: str = "", **kwargs) -> str:
    """List installed applications (best-effort, per platform)."""
    flt = (filter or "").strip().lower()
    names: list[str] = []
    try:
        if _SYSTEM == "Windows":
            from pathlib import Path

            _start_menu = "Microsoft/Windows/Start Menu/Programs"
            roots = [
                Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData")) / _start_menu,
                Path(os.environ.get("APPDATA", "")) / _start_menu,
            ]
            for root in roots:
                if root.is_dir():
                    names += [p.stem for p in root.rglob("*.lnk")]
        elif _SYSTEM == "Darwin":
            from pathlib import Path

            for root in (Path("/Applications"), Path.home() / "Applications"):
                if root.is_dir():
                    names += [p.stem for p in root.glob("*.app")]
        else:  # Linux
            from pathlib import Path

            for root in (
                Path("/usr/share/applications"),
                Path.home() / ".local/share/applications",
            ):
                if root.is_dir():
                    names += [p.stem for p in root.glob("*.desktop")]
    except Exception as e:  # noqa: BLE001
        return f"枚举失败: {e}"

    uniq = sorted(set(names), key=str.lower)
    if flt:
        uniq = [n for n in uniq if flt in n.lower()]
    if not uniq:
        return "未找到应用" + (f"（过滤词 {filter!r}）" if flt else "")
    shown = uniq[:200]
    head = f"共 {len(uniq)} 个应用" + ("（已截断到 200）" if len(uniq) > 200 else "")
    return head + ":\n" + "\n".join(f"- {n}" for n in shown)


# ── System info / diagnostics ────────────────────────────────────────────


def _disk_report() -> list[str]:
    lines: list[str] = []
    try:
        usage = shutil.disk_usage(os.path.abspath(os.sep))
        gb = 1024**3
        pct = usage.used / usage.total * 100 if usage.total else 0
        lines.append(
            f"磁盘(系统盘): {usage.used / gb:.1f}/{usage.total / gb:.1f} GB "
            f"({pct:.0f}% 已用, 剩余 {usage.free / gb:.1f} GB)"
        )
    except Exception:  # noqa: BLE001
        pass
    return lines


async def _system_info(**kwargs) -> str:
    """Report OS, CPU, memory, disk, and uptime."""
    lines = [
        f"系统: {platform.system()} {platform.release()} ({platform.machine()})",
        f"主机名: {platform.node()}",
        f"Python: {platform.python_version()}",
        f"CPU 核心: {os.cpu_count()}",
    ]
    try:
        import psutil  # optional

        vm = psutil.virtual_memory()
        gb = 1024**3
        lines.append(f"内存: {vm.used / gb:.1f}/{vm.total / gb:.1f} GB ({vm.percent:.0f}% 已用)")
        import datetime

        boot = datetime.datetime.fromtimestamp(psutil.boot_time())
        up = datetime.datetime.now() - boot
        lines.append(
            f"开机时长: {up.days}天 {up.seconds // 3600}小时 {(up.seconds % 3600) // 60}分"
        )
    except ImportError:
        lines.append("内存/uptime: 需安装 psutil (pip install psutil)")
    except Exception:  # noqa: BLE001
        pass
    lines += _disk_report()
    return "\n".join(lines)


async def _system_diagnose(**kwargs) -> str:
    """Run quick health checks and suggest fixes for common problems."""
    findings: list[str] = []
    fixes: list[str] = []

    # Disk pressure
    try:
        usage = shutil.disk_usage(os.path.abspath(os.sep))
        free_pct = usage.free / usage.total * 100 if usage.total else 100
        gb = 1024**3
        if free_pct < 10:
            findings.append(f"⚠ 系统盘空间不足: 仅剩 {usage.free / gb:.1f} GB ({free_pct:.0f}%)")
            fixes.append("清理临时文件/回收站；卸载不用的软件（package_manager uninstall）")
        else:
            findings.append(f"✓ 磁盘空间正常 (剩余 {free_pct:.0f}%)")
    except Exception:  # noqa: BLE001
        pass

    # Memory + top processes (needs psutil)
    try:
        import psutil

        vm = psutil.virtual_memory()
        if vm.percent > 90:
            findings.append(f"⚠ 内存吃紧: {vm.percent:.0f}% 已用")
            fixes.append("关闭占用内存最高的进程（见下方 TOP，可用 kill_process）")
        else:
            findings.append(f"✓ 内存正常 ({vm.percent:.0f}%)")

        procs = []
        for p in psutil.process_iter(["pid", "name", "memory_percent"]):
            try:
                procs.append((p.info["memory_percent"] or 0, p.info["pid"], p.info["name"]))
            except Exception:  # noqa: BLE001
                continue
        procs.sort(reverse=True)
        top = procs[:5]
        if top:
            findings.append("内存 TOP5: " + ", ".join(f"{n}({pid}) {m:.0f}%" for m, pid, n in top))
    except ImportError:
        findings.append("（安装 psutil 可检查内存与进程）")
    except Exception:  # noqa: BLE001
        pass

    report = ["== 系统诊断 =="] + findings
    if fixes:
        report += ["", "建议:"] + [f"- {f}" for f in fixes]
    else:
        report += ["", "未发现明显问题。"]
    return "\n".join(report)


# ── Process management ───────────────────────────────────────────────────


async def _list_processes(filter: str = "", limit: int = 20, **kwargs) -> str:
    """List running processes, optionally filtered by name."""
    flt = (filter or "").strip().lower()
    try:
        import psutil
    except ImportError:
        # Fallback to platform tool.
        if _SYSTEM == "Windows":
            _, out, _ = await _run(["tasklist"])
        else:
            _, out, _ = await _run(["ps", "-eo", "pid,comm,%mem,%cpu", "--sort=-%mem"])
        if flt:
            out = "\n".join(ln for ln in out.splitlines() if flt in ln.lower())
        return _truncate(out) or "无匹配进程"

    rows = []
    for p in psutil.process_iter(["pid", "name", "memory_percent", "cpu_percent"]):
        try:
            info = p.info
            name = info.get("name") or "?"
            if flt and flt not in name.lower():
                continue
            rows.append(
                (
                    info.get("memory_percent") or 0,
                    info.get("pid"),
                    name,
                    info.get("cpu_percent") or 0,
                )
            )
        except Exception:  # noqa: BLE001
            continue
    rows.sort(reverse=True)
    rows = rows[: max(1, int(limit))]
    if not rows:
        return "无匹配进程"
    lines = [f"{'PID':>7}  {'MEM%':>5}  NAME"]
    lines += [f"{pid:>7}  {mem:>5.1f}  {name}" for mem, pid, name, _cpu in rows]
    return "\n".join(lines)


async def _kill_process(target: str = "", **kwargs) -> str:
    """Terminate a process by PID or name. DANGEROUS — gated by approval."""
    target = (target or "").strip()
    if not target:
        return "Error: target (pid or name) is required"
    # Reject shell/metacharacter injection — pkill uses regex by default, and
    # unescaped '.' would match ANY process.
    for ch in target:
        if ch in "&|;`$(){}[]<>!?*+.^\\\"'%~\n\r\t":
            return f"Error: unsafe character {ch!r} in target"
    try:
        import psutil
    except ImportError:
        if _SYSTEM == "Windows":
            arg = ["/PID", target] if target.isdigit() else ["/IM", target]
            rc, out, err = await _run(["taskkill", "/F", *arg])
        else:
            rc, out, err = (
                await _run(["pkill", "-9", "--", target])
                if not target.isdigit()
                else await _run(["kill", "-9", target])
            )
        return f"已结束 {target}" if rc == 0 else f"失败: {err or out}"

    killed = []
    if target.isdigit():
        try:
            p = psutil.Process(int(target))
            p.terminate()
            killed.append(f"{p.name()}({p.pid})")
        except psutil.NoSuchProcess:
            return f"无此进程: PID {target}"
        except psutil.AccessDenied:
            return f"权限不足，无法结束 PID {target}（可能需要管理员）"
    else:
        for p in psutil.process_iter(["pid", "name"]):
            try:
                if (p.info["name"] or "").lower() == target.lower():
                    p.terminate()
                    killed.append(f"{p.info['name']}({p.info['pid']})")
            except Exception:  # noqa: BLE001
                continue
        if not killed:
            return f"未找到名为 {target!r} 的进程"
    return "已结束: " + ", ".join(killed)


# ── Package management (install / uninstall software) ────────────────────

# Per-OS manager → {action: argv-template}. First available manager wins.
_PKG_MANAGERS: dict[str, list[tuple[str, dict[str, list[str]]]]] = {
    "Windows": [
        (
            "winget",
            {
                "install": [
                    "install",
                    "--accept-package-agreements",
                    "--accept-source-agreements",
                    "-e",
                ],
                "uninstall": ["uninstall", "-e"],
                "search": ["search"],
                "upgrade": ["upgrade", "-e"],
                "list": ["list"],
            },
        ),
        (
            "scoop",
            {
                "install": ["install"],
                "uninstall": ["uninstall"],
                "search": ["search"],
                "upgrade": ["update"],
                "list": ["list"],
            },
        ),
        (
            "choco",
            {
                "install": ["install", "-y"],
                "uninstall": ["uninstall", "-y"],
                "search": ["search"],
                "upgrade": ["upgrade", "-y"],
                "list": ["list", "--local-only"],
            },
        ),
    ],
    "Darwin": [
        (
            "brew",
            {
                "install": ["install"],
                "uninstall": ["uninstall"],
                "search": ["search"],
                "upgrade": ["upgrade"],
                "list": ["list"],
            },
        ),
    ],
    "Linux": [
        (
            "apt-get",
            {
                "install": ["install", "-y"],
                "uninstall": ["remove", "-y"],
                "search": ["search"],
                "upgrade": ["upgrade", "-y"],
                "list": ["list", "--installed"],
            },
        ),
        (
            "dnf",
            {
                "install": ["install", "-y"],
                "uninstall": ["remove", "-y"],
                "search": ["search"],
                "upgrade": ["upgrade", "-y"],
                "list": ["list", "installed"],
            },
        ),
        (
            "pacman",
            {
                "install": ["-S", "--noconfirm"],
                "uninstall": ["-R", "--noconfirm"],
                "search": ["-Ss"],
                "upgrade": ["-Syu", "--noconfirm"],
                "list": ["-Q"],
            },
        ),
        (
            "brew",
            {
                "install": ["install"],
                "uninstall": ["uninstall"],
                "search": ["search"],
                "upgrade": ["upgrade"],
                "list": ["list"],
            },
        ),
    ],
}


def _detect_pkg_manager() -> tuple[str, dict[str, list[str]]] | None:
    for name, actions in _PKG_MANAGERS.get(_SYSTEM, []):
        if shutil.which(name):
            return name, actions
    return None


async def _package_manager(action: str = "", name: str = "", **kwargs) -> str:
    """Install/uninstall/search/upgrade software via the OS package manager.

    DANGEROUS — gated by approval. Note: on Linux, apt/dnf/pacman need root;
    since sudo is blocked, install/uninstall there require the agent to be run
    with sufficient privileges or will report a permissions error.
    """
    action = (action or "").strip().lower()
    name = (name or "").strip()
    valid = {"install", "uninstall", "search", "upgrade", "list"}
    if action not in valid:
        return f"Error: action must be one of {sorted(valid)}"
    if action in {"install", "uninstall", "search"} and not name:
        return f"Error: '{action}' requires a package name"

    mgr = _detect_pkg_manager()
    if mgr is None:
        avail = [m for m, _ in _PKG_MANAGERS.get(_SYSTEM, [])]
        return (
            f"未检测到包管理器。{_SYSTEM} 支持: {', '.join(avail) or '无'}。"
            "请先安装其中之一（如 Windows 的 winget / macOS 的 brew）。"
        )
    mgr_name, actions = mgr
    if action not in actions:
        return f"{mgr_name} 不支持 {action}"
    argv = [mgr_name, *actions[action]]
    if name:
        argv.append(name)
    # Install/upgrade can be slow (download + compile); give a wide window.
    timeout = 600.0 if action in {"install", "upgrade"} else 90.0
    rc, out, err = await _run(argv, timeout=timeout)
    body = _truncate((out + ("\n" + err if err else "")).strip())
    status = "✓ 成功" if rc == 0 else f"✗ 失败 (exit {rc})"
    return f"[{mgr_name} {action} {name}] {status}\n{body}"


# ── Service control ──────────────────────────────────────────────────────


async def _service_control(action: str = "", name: str = "", **kwargs) -> str:
    """Start/stop/restart/status a system service. DANGEROUS — gated."""
    action = (action or "").strip().lower()
    name = (name or "").strip()
    valid = {"start", "stop", "restart", "status"}
    if action not in valid:
        return f"Error: action must be one of {sorted(valid)}"
    if not name:
        return "Error: service name is required"
    if _SYSTEM == "Windows":
        # sc.exe verbs differ slightly from the unix verbs.
        verb = {"start": "start", "stop": "stop", "restart": "stop", "status": "query"}[action]
        rc, out, err = await _run(["sc", verb, name])
        if action == "restart" and rc == 0:
            await _run(["sc", "start", name])
    elif _SYSTEM == "Darwin":
        verb = {"start": "load", "stop": "unload", "restart": "kickstart", "status": "list"}[action]
        rc, out, err = await _run(["launchctl", verb, name])
    else:  # Linux
        rc, out, err = await _run(["systemctl", action, name])
    body = _truncate((out + ("\n" + err if err else "")).strip())
    return f"[service {action} {name}] {'✓' if rc == 0 else f'✗ exit {rc}'}\n{body}"


# ── Registration ─────────────────────────────────────────────────────────


def register_computer_tools(reg: ToolRegistry) -> None:
    """Register the computer-operation tools into *reg*."""
    tools = [
        Tool(
            name="launch_app",
            description=(
                "Launch a desktop application by name or path (cross-platform). "
                "e.g. 'notepad', 'Google Chrome', 'code'. Optional args string."
            ),
            parameters=[
                ToolParameter(name="name", type="string", description="App name or path"),
                ToolParameter(
                    name="args",
                    type="string",
                    description="Optional launch arguments",
                    required=False,
                ),
            ],
            handler=_launch_app,
            cacheable=False,
        ),
        Tool(
            name="open_path",
            description="Open a file, folder, or URL in the OS default handler.",
            parameters=[
                ToolParameter(name="target", type="string", description="File path, folder, or URL")
            ],
            handler=_open_path,
            cacheable=False,
        ),
        Tool(
            name="browser_open",
            description="Open a URL in the default web browser.",
            parameters=[ToolParameter(name="url", type="string", description="URL to open")],
            handler=_browser_open,
            cacheable=False,
        ),
        Tool(
            name="list_installed_apps",
            description="List installed desktop applications (optionally filter by substring).",
            parameters=[
                ToolParameter(
                    name="filter",
                    type="string",
                    description="Case-insensitive name filter",
                    required=False,
                )
            ],
            handler=_list_installed_apps,
        ),
        Tool(
            name="system_info",
            description="Report OS, CPU, memory, disk, and uptime of this machine.",
            parameters=[],
            handler=_system_info,
        ),
        Tool(
            name="system_diagnose",
            description=(
                "Run quick health checks (disk space, memory, top processes) and "
                "suggest fixes for common problems. Read-only."
            ),
            parameters=[],
            handler=_system_diagnose,
        ),
        Tool(
            name="list_processes",
            description="List running processes (optionally filtered by name), sorted by memory.",
            parameters=[
                ToolParameter(
                    name="filter", type="string", description="Name filter", required=False
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="Max rows (default 20)",
                    required=False,
                ),
            ],
            handler=_list_processes,
        ),
        Tool(
            name="kill_process",
            description="Terminate a process by PID or name. Use after list_processes to confirm.",
            parameters=[
                ToolParameter(
                    name="target", type="string", description="PID (number) or process name"
                )
            ],
            handler=_kill_process,
            dangerous=True,
            cacheable=False,
        ),
        Tool(
            name="package_manager",
            description=(
                "Install / uninstall / search / upgrade / list software via the OS "
                "package manager (winget/scoop/choco on Windows, brew on macOS, "
                "apt/dnf/pacman on Linux). Auto-detects the available manager."
            ),
            parameters=[
                ToolParameter(
                    name="action",
                    type="string",
                    description="install | uninstall | search | upgrade | list",
                ),
                ToolParameter(
                    name="name",
                    type="string",
                    description="Package name (not needed for list)",
                    required=False,
                ),
            ],
            handler=_package_manager,
            dangerous=True,
            cacheable=False,
        ),
        Tool(
            name="service_control",
            description=(
                "Start / stop / restart / status a system service "
                "(sc on Windows, launchctl on macOS, systemctl on Linux)."
            ),
            parameters=[
                ToolParameter(
                    name="action", type="string", description="start | stop | restart | status"
                ),
                ToolParameter(name="name", type="string", description="Service name"),
            ],
            handler=_service_control,
            dangerous=True,
            cacheable=False,
        ),
    ]
    for tool in tools:
        reg.register(tool)

"""Tests for workspace detection and management."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


class TestWorkspaceDetection:
    def test_drive_list_returns_drives(self):
        from weather_agents.core.workspace import _get_drive_list

        drives = _get_drive_list()
        assert len(drives) > 0
        for d in drives:
            assert d.path
            assert d.total_bytes > 0
            assert d.free_bytes >= 0

    def test_detect_best_workspace_root_is_absolute(self):
        from weather_agents.core.workspace import detect_best_workspace_root

        root = detect_best_workspace_root()
        assert root.is_absolute()
        assert root.name == "workspace"

    @pytest.mark.skipif(os.name != "nt", reason="Windows drive detection")
    def test_detect_skips_c_drive_when_others_exist(self):
        from weather_agents.core.workspace import DriveInfo, detect_best_workspace_root

        candidates = [
            DriveInfo("C", "C:\\", 100_000_000_000, 200_000_000_000),
            DriveInfo("D", "D:\\", 500_000_000_000, 400_000_000_000),
        ]
        with (
            patch("weather_agents.core.workspace._get_drive_list", return_value=candidates),
            patch("weather_agents.core.workspace.os.name", "nt"),
        ):
            root = detect_best_workspace_root()
            assert str(root).startswith("D:")

    @pytest.mark.skipif(os.name != "nt", reason="Windows drive detection")
    def test_detect_falls_back_to_c_when_only_c(self):
        from weather_agents.core.workspace import DriveInfo, detect_best_workspace_root

        candidates = [
            DriveInfo("C", "C:\\", 100_000_000_000, 50_000_000_000),
        ]
        with (
            patch("weather_agents.core.workspace._get_drive_list", return_value=candidates),
            patch("weather_agents.core.workspace.os.name", "nt"),
        ):
            root = detect_best_workspace_root()
            assert str(root).startswith("C:")

    @pytest.mark.skipif(os.name != "nt", reason="Windows drive detection")
    def test_detect_picks_most_free_space(self):
        from weather_agents.core.workspace import DriveInfo, detect_best_workspace_root

        candidates = [
            DriveInfo("D", "D:\\", 500_000_000_000, 100_000_000_000),
            DriveInfo("E", "E:\\", 500_000_000_000, 800_000_000_000),
            DriveInfo("F", "F:\\", 500_000_000_000, 50_000_000_000),
        ]
        with (
            patch("weather_agents.core.workspace._get_drive_list", return_value=candidates),
            patch("weather_agents.core.workspace.os.name", "nt"),
        ):
            root = detect_best_workspace_root()
            assert str(root).startswith("E:")  # most free space


class TestWorkspaceResolve:
    def test_resolve_auto(self):
        from weather_agents.core.workspace import resolve_workspace_path

        path = resolve_workspace_path("auto")
        assert path.is_absolute()
        assert path.name == "workspace"

    def test_resolve_explicit_path(self):
        from weather_agents.core.workspace import resolve_workspace_path

        path = resolve_workspace_path("~/my-agents-workspace")
        assert "my-agents-workspace" in str(path)

    def test_resolve_case_insensitive_auto(self):
        from weather_agents.core.workspace import resolve_workspace_path

        assert resolve_workspace_path("AUTO").name == "workspace"
        assert resolve_workspace_path("Auto").name == "workspace"


class TestWorkspaceInit:
    def test_init_creates_subdirs(self, tmp_path):
        from weather_agents.core.workspace import init_workspace

        ws = tmp_path / "workspace"
        result = init_workspace(ws)

        assert result == ws
        assert ws.exists()
        assert (ws / "files").is_dir()
        assert (ws / "output").is_dir()
        assert (ws / "temp").is_dir()
        assert (ws / ".workspace").exists()

    def test_init_is_idempotent(self, tmp_path):
        from weather_agents.core.workspace import init_workspace

        ws = tmp_path / "workspace"
        init_workspace(ws)
        marker_mtime = (ws / ".workspace").stat().st_mtime

        init_workspace(ws)
        assert (ws / ".workspace").stat().st_mtime == marker_mtime  # not overwritten

    def test_format_bytes(self):
        from weather_agents.core.workspace import format_bytes

        assert "B" in format_bytes(500)
        assert "KB" in format_bytes(2048)
        assert "MB" in format_bytes(5_000_000)
        assert "GB" in format_bytes(10_000_000_000)

    def test_format_bytes_petabytes(self):
        from weather_agents.core.workspace import format_bytes

        result = format_bytes(2_500_000_000_000_000)  # 2.5 PB
        assert "PB" in result

    def test_format_bytes_negative(self):
        from weather_agents.core.workspace import format_bytes

        result = format_bytes(-500)
        assert "B" in result


class TestWorkspaceDriveList:
    def test_oserror_on_drive_skipped(self):
        from weather_agents.core.workspace import _get_drive_list

        drives = _get_drive_list()
        # Should not crash; should return valid drives
        assert isinstance(drives, list)
        for d in drives:
            assert d.path


class TestWorkspaceDetectRoot:
    def test_unix_uses_home(self):
        import os as _os
        from pathlib import Path

        from weather_agents.core.workspace import detect_best_workspace_root

        if _os.name != "nt":
            root = detect_best_workspace_root()
            expected = Path.home() / "workspace"
            assert root == expected

    @pytest.mark.skipif(os.name != "posix", reason="Unix-specific code path")
    def test_get_drive_list_oserror_on_unix(self):
        from unittest.mock import patch

        from weather_agents.core.workspace import _get_drive_list

        with patch("weather_agents.core.workspace.shutil.disk_usage", side_effect=OSError("no disk")):
            drives = _get_drive_list()
            assert drives == []


class TestWorkspaceConfig:
    def test_default_config_is_auto(self):
        from weather_agents.core.config import AppConfig

        cfg = AppConfig()
        assert cfg.workspace.path == "auto"

    def test_set_and_delete_workspace_path(self, temp_config_dir):
        import os as _os

        from weather_agents.core.config import delete_config, load_config, set_config

        # Use a platform-appropriate absolute path
        test_path = "D:\\my-workspace" if _os.name == "nt" else "/custom/workspace"

        ok, msg = set_config("workspace.path", test_path)
        assert ok, msg
        cfg = load_config()
        assert cfg.workspace.path == test_path

        ok, msg = delete_config("workspace.path")
        assert ok, msg
        cfg = load_config()
        assert cfg.workspace.path == "auto"

    def test_set_workspace_rejects_relative_path(self, temp_config_dir):
        from weather_agents.core.config import set_config

        ok, msg = set_config("workspace.path", "relative/path")
        assert not ok

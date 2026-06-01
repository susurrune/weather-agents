"""Tests for built-in tools."""

from __future__ import annotations

import os
import sys

import pytest

_IS_WIN = sys.platform == "win32"


class TestFileTools:
    @pytest.mark.asyncio
    async def test_read_file(self, tmp_path):
        from weather_agents.tools.builtin import _read_file

        f = tmp_path / "test.txt"
        f.write_text("hello world")
        result = await _read_file(str(f))
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_read_file_not_found(self):
        from weather_agents.tools.builtin import _read_file

        result = await _read_file("/nonexistent/path.txt")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_write_file(self, tmp_path):
        from weather_agents.tools.builtin import _write_file

        f = tmp_path / "output.txt"
        result = await _write_file(str(f), "test content")
        assert "Successfully" in result
        assert f.read_text() == "test content"

    @pytest.mark.asyncio
    async def test_write_file_creates_dirs(self, tmp_path):
        from weather_agents.tools.builtin import _write_file

        f = tmp_path / "sub" / "dir" / "file.txt"
        result = await _write_file(str(f), "nested")
        assert "Successfully" in result
        assert f.read_text() == "nested"

    @pytest.mark.asyncio
    async def test_edit_file(self, tmp_path):
        from weather_agents.tools.builtin import _edit_file

        f = tmp_path / "edit.txt"
        f.write_text("hello world")
        result = await _edit_file(str(f), "world", "python")
        assert "Successfully" in result
        assert f.read_text() == "hello python"

    @pytest.mark.asyncio
    async def test_edit_file_text_not_found(self, tmp_path):
        from weather_agents.tools.builtin import _edit_file

        f = tmp_path / "edit2.txt"
        f.write_text("hello")
        result = await _edit_file(str(f), "nonexistent", "replacement")
        assert "Error" in result


class TestSearchTools:
    @pytest.mark.asyncio
    async def test_file_search(self, tmp_path):
        from weather_agents.tools.builtin import _file_search

        (tmp_path / "test.py").write_text("")
        (tmp_path / "test.txt").write_text("")
        result = await _file_search(str(tmp_path), "*.py")
        assert "test.py" in result

    @pytest.mark.asyncio
    async def test_file_search_no_matches(self, tmp_path):
        from weather_agents.tools.builtin import _file_search

        result = await _file_search(str(tmp_path), "*.xyz")
        assert "No files" in result


class TestShellExec:
    @pytest.mark.asyncio
    async def test_basic_command(self):
        from weather_agents.tools.builtin import _shell_exec

        cmd = "cmd /c echo hello" if _IS_WIN else "echo hello"
        result = await _shell_exec(cmd)
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_blocked_command(self):
        from weather_agents.tools.builtin import _shell_exec

        result = await _shell_exec("shutdown now")
        assert "Blocked" in result

    @pytest.mark.asyncio
    async def test_invalid_syntax(self):
        from weather_agents.tools.builtin import _shell_exec

        result = await _shell_exec("echo 'unterminated")
        assert "Invalid" in result or "Error" in result or "hello" not in result

    @pytest.mark.asyncio
    async def test_timeout(self):
        from weather_agents.tools.builtin import _shell_exec

        if _IS_WIN:
            result = await _shell_exec("ping -n 11 127.0.0.1", timeout=1)
        else:
            result = await _shell_exec("sleep 10", timeout=1)
        assert "timed out" in result


class TestWebSearch:
    @pytest.mark.asyncio
    async def test_parse_ddg_html(self):
        from weather_agents.tools.builtin import _parse_ddg_html

        html = """
        <a class="result__a" href="http://example.com">Example Title</a>
        <a class="result__snippet">This is a snippet</a>
        """
        results = _parse_ddg_html(html, 5)
        assert len(results) == 1
        assert results[0]["title"] == "Example Title"

    @pytest.mark.asyncio
    async def test_parse_ddg_empty(self):
        from weather_agents.tools.builtin import _parse_ddg_html

        results = _parse_ddg_html("<html>no results</html>", 5)
        assert results == []

    @pytest.mark.asyncio
    async def test_parse_bing_html(self):
        from weather_agents.tools.builtin import _parse_bing_html

        html = (
            '<li class="b_algo">'
            '<h2 class=""><a href="https://news.example.com/x">Big &#183; News</a></h2>'
            '<div class="b_caption"><p class="b_lineclamp2">Oct 9, 2025&ensp;summary.</p></div>'
            "</li>"
            '<li class="b_algo">'
            '<h2><a href="https://two.example.com/">Second</a></h2>'
            "<p>second snippet</p></li>"
        )
        results = _parse_bing_html(html, 5)
        assert len(results) == 2
        assert results[0]["title"] == "Big · News"  # entity decoded
        assert results[0]["url"] == "https://news.example.com/x"
        assert "summary" in results[0]["snippet"]
        assert results[1]["url"] == "https://two.example.com/"

    @pytest.mark.asyncio
    async def test_parse_bing_skips_relative_links(self):
        from weather_agents.tools.builtin import _parse_bing_html

        html = '<li class="b_algo"><h2><a href="/relative">x</a></h2></li>'
        assert _parse_bing_html(html, 5) == []

    @pytest.mark.asyncio
    async def test_race_search_returns_first_nonempty(self):
        import asyncio

        from weather_agents.tools.builtin import _race_search

        async def slow_winner():
            await asyncio.sleep(0.05)
            return [{"title": "T", "url": "https://x", "snippet": ""}]

        async def fast_empty():
            return []

        errors: list[str] = []
        # The empty backend resolves first but must NOT end the race — the
        # non-empty sibling wins.
        out = await _race_search(
            {"empty": fast_empty(), "winner": slow_winner()}, timeout=2.0, errors=errors
        )
        assert out and out[0]["title"] == "T"

    @pytest.mark.asyncio
    async def test_race_search_all_empty(self):
        from weather_agents.tools.builtin import _race_search

        async def empty():
            return []

        errors: list[str] = []
        out = await _race_search(
            {"a": empty(), "b": empty()}, timeout=2.0, errors=errors
        )
        assert out is None


class TestHttpTools:
    @pytest.mark.asyncio
    async def test_http_get_invalid_url(self):
        from weather_agents.tools.builtin import _http_get

        result = await _http_get("not-a-url")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_http_post_invalid_url(self):
        from weather_agents.tools.builtin import _http_post

        result = await _http_post("not-a-url", "data")
        assert "Error" in result


class TestDirectoryTools:
    @pytest.mark.asyncio
    async def test_list_directory(self, tmp_path):
        from weather_agents.tools.builtin import _list_directory

        (tmp_path / "file.txt").write_text("hello")
        (tmp_path / "subdir").mkdir()
        result = await _list_directory(str(tmp_path))
        assert "file.txt" in result
        assert "subdir/" in result

    @pytest.mark.asyncio
    async def test_list_directory_not_found(self):
        from weather_agents.tools.builtin import _list_directory

        result = await _list_directory("/nonexistent/path")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_list_directory_empty(self, tmp_path):
        from weather_agents.tools.builtin import _list_directory

        result = await _list_directory(str(tmp_path))
        assert "Empty" in result

    @pytest.mark.asyncio
    async def test_tree(self, tmp_path):
        from weather_agents.tools.builtin import _tree

        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "b.txt").write_text("x")
        (tmp_path / "c.txt").write_text("y")
        result = await _tree(str(tmp_path), max_depth=2)
        assert "b.txt" in result
        assert "c.txt" in result

    @pytest.mark.asyncio
    async def test_tree_max_depth(self, tmp_path):
        from weather_agents.tools.builtin import _tree

        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "b").mkdir()
        (tmp_path / "a" / "b" / "c.txt").write_text("deep")
        result = await _tree(str(tmp_path), max_depth=1)
        assert "a/" in result
        assert "c.txt" not in result


class TestFileManagementTools:
    @pytest.mark.asyncio
    async def test_move_file(self, tmp_path):
        from weather_agents.tools.builtin import _move_file

        src = tmp_path / "src.txt"
        src.write_text("content")
        dst = tmp_path / "dst.txt"
        result = await _move_file(str(src), str(dst))
        assert "Moved" in result
        assert not src.exists()
        assert dst.read_text() == "content"

    @pytest.mark.asyncio
    async def test_move_file_not_found(self, tmp_path):
        from weather_agents.tools.builtin import _move_file

        result = await _move_file("/nonexistent", str(tmp_path / "dst"))
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_copy_file(self, tmp_path):
        from weather_agents.tools.builtin import _copy_file

        src = tmp_path / "src.txt"
        src.write_text("content")
        dst = tmp_path / "copy.txt"
        result = await _copy_file(str(src), str(dst))
        assert "Copied" in result
        assert src.exists()
        assert dst.read_text() == "content"

    @pytest.mark.asyncio
    async def test_copy_directory(self, tmp_path):
        from weather_agents.tools.builtin import _copy_file

        src_dir = tmp_path / "srcdir"
        src_dir.mkdir()
        (src_dir / "a.txt").write_text("a")
        dst_dir = tmp_path / "dstdir"
        result = await _copy_file(str(src_dir), str(dst_dir))
        assert "Copied" in result
        assert (dst_dir / "a.txt").read_text() == "a"

    @pytest.mark.asyncio
    async def test_delete_file(self, tmp_path):
        from weather_agents.tools.builtin import _delete_file

        f = tmp_path / "del.txt"
        f.write_text("bye")
        result = await _delete_file(str(f))
        assert "Deleted" in result
        assert not f.exists()

    @pytest.mark.asyncio
    async def test_delete_empty_dir(self, tmp_path):
        from weather_agents.tools.builtin import _delete_file

        d = tmp_path / "emptydir"
        d.mkdir()
        result = await _delete_file(str(d))
        assert "Deleted" in result
        assert not d.exists()

    @pytest.mark.asyncio
    async def test_delete_nonempty_dir(self, tmp_path):
        from weather_agents.tools.builtin import _delete_file

        d = tmp_path / "nonempty"
        d.mkdir()
        (d / "file.txt").write_text("x")
        result = await _delete_file(str(d))
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_get_cwd(self):
        from weather_agents.tools.builtin import _get_cwd

        result = await _get_cwd()
        assert os.path.isabs(result)


class TestTruncation:
    def test_truncate_under_limit(self):
        from weather_agents.tools.builtin import _truncate

        assert _truncate("hi", 100) == "hi"

    def test_truncate_over_limit_has_marker(self):
        from weather_agents.tools.builtin import _truncate

        out = _truncate("x" * 500, 100, label="file")
        assert out.startswith("x" * 100)
        assert "truncated" in out
        assert "500" in out
        assert "file" in out


class TestShellSafety:
    def test_dangerous_rm_root(self):
        from weather_agents.tools.builtin import _is_dangerous_rm

        assert _is_dangerous_rm(["rm", "-rf", "/"])
        assert _is_dangerous_rm(["rm", "-rf", "~"])
        assert _is_dangerous_rm(["rm", "-rf", "."])
        assert _is_dangerous_rm(["rm", "-r", "--recursive", "C:\\"])

    def test_safe_rm(self):
        from weather_agents.tools.builtin import _is_dangerous_rm

        assert not _is_dangerous_rm(["rm", "file.txt"])
        assert not _is_dangerous_rm(["rm", "-rf", "build/"])
        assert not _is_dangerous_rm(["rm", "-rf", "/tmp/specific-dir"])

    @pytest.mark.asyncio
    async def test_shell_exec_blocks_sudo(self):
        from weather_agents.tools.builtin import _shell_exec

        result = await _shell_exec("sudo ls /")
        assert "Blocked" in result

    @pytest.mark.asyncio
    async def test_shell_exec_blocks_metacharacters(self):
        from weather_agents.tools.builtin import _shell_exec

        result = await _shell_exec("echo a; rm -rf /")
        # shlex may not even parse this — either way it should be refused.
        assert "Blocked" in result or "Invalid" in result or "exit" in result.lower()

    @pytest.mark.asyncio
    async def test_shell_exec_overlong_command(self):
        from weather_agents.tools.builtin import _shell_exec

        result = await _shell_exec("echo " + "x" * 5000)
        assert "too long" in result


class TestSSRFGuard:
    def test_validate_url_rejects_file_scheme(self):
        from weather_agents.tools.builtin import _validate_url

        assert _validate_url("file:///etc/passwd") is not None

    def test_validate_url_rejects_loopback(self):
        from weather_agents.tools.builtin import _validate_url

        assert _validate_url("http://127.0.0.1/") is not None
        assert _validate_url("http://localhost:6379/") is not None
        assert _validate_url("http://169.254.169.254/latest/meta-data") is not None

    def test_validate_url_rejects_private_ips(self):
        from weather_agents.tools.builtin import _validate_url

        assert _validate_url("http://10.0.0.1/") is not None
        assert _validate_url("http://192.168.1.1/") is not None
        assert _validate_url("http://172.16.0.5/") is not None

    def test_validate_url_allows_public(self):
        from weather_agents.tools.builtin import _validate_url

        assert _validate_url("https://example.com/api") is None
        assert _validate_url("https://api.openai.com/v1") is None


class TestCodeSearch:
    @pytest.mark.asyncio
    async def test_code_search_finds_text(self, tmp_path):
        from weather_agents.tools.builtin import _code_search

        (tmp_path / "a.py").write_text("def foo():\n    pass\n")
        (tmp_path / "b.py").write_text("class Bar:\n    pass\n")
        result = await _code_search(str(tmp_path), "foo")
        assert "a.py" in result
        assert "foo" in result

    @pytest.mark.asyncio
    async def test_code_search_regex_mode(self, tmp_path):
        from weather_agents.tools.builtin import _code_search

        (tmp_path / "a.py").write_text("x = 42\ny = 13\n")
        result = await _code_search(str(tmp_path), r"=\s*\d+", regex=True)
        assert "a.py" in result

    @pytest.mark.asyncio
    async def test_code_search_invalid_regex(self, tmp_path):
        from weather_agents.tools.builtin import _code_search

        (tmp_path / "a.py").write_text("x")
        result = await _code_search(str(tmp_path), "([unclosed", regex=True)
        assert "invalid regex" in result.lower() or "error" in result.lower()

    @pytest.mark.asyncio
    async def test_code_search_skips_heavy_dirs(self, tmp_path):
        from weather_agents.tools.builtin import _code_search

        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("interesting")
        (tmp_path / "a.py").write_text("nothing here")
        result = await _code_search(str(tmp_path), "interesting")
        assert ".git" not in result


class TestFileSearchPathlib:
    @pytest.mark.asyncio
    async def test_file_search_recursive(self, tmp_path):
        from weather_agents.tools.builtin import _file_search

        (tmp_path / "a.py").write_text("")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "b.py").write_text("")
        result = await _file_search(str(tmp_path), "*.py")
        assert "a.py" in result
        assert "b.py" in result

    @pytest.mark.asyncio
    async def test_file_search_no_match(self, tmp_path):
        from weather_agents.tools.builtin import _file_search

        result = await _file_search(str(tmp_path), "*.nonexistent")
        assert "No files" in result


class TestGrepTool:
    @pytest.mark.asyncio
    async def test_grep_finds_text(self, tmp_path):
        from weather_agents.tools.builtin import _grep

        (tmp_path / "a.txt").write_text("hello world\nfoo bar\n", encoding="utf-8")
        result = await _grep(str(tmp_path), "hello")
        assert "hello world" in result

    @pytest.mark.asyncio
    async def test_grep_regex_mode(self, tmp_path):
        from weather_agents.tools.builtin import _grep

        (tmp_path / "a.txt").write_text("hello world\nfoo bar\n", encoding="utf-8")
        result = await _grep(str(tmp_path), r"h.llo", regex=True)
        assert "hello world" in result

    @pytest.mark.asyncio
    async def test_grep_ignore_case(self, tmp_path):
        from weather_agents.tools.builtin import _grep

        (tmp_path / "a.txt").write_text("Hello World\n", encoding="utf-8")
        result = await _grep(str(tmp_path), "hello", ignore_case=True)
        assert "Hello World" in result

    @pytest.mark.asyncio
    async def test_grep_invalid_regex(self, tmp_path):
        from weather_agents.tools.builtin import _grep

        result = await _grep(str(tmp_path), "[invalid", regex=True)
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_grep_no_match(self, tmp_path):
        from weather_agents.tools.builtin import _grep

        (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
        result = await _grep(str(tmp_path), "nonexistent")
        assert "No matches" in result

    @pytest.mark.asyncio
    async def test_grep_with_glob_filter(self, tmp_path):
        from weather_agents.tools.builtin import _grep

        (tmp_path / "a.py").write_text("TODO: fix me\n", encoding="utf-8")
        (tmp_path / "b.txt").write_text("TODO: also\n", encoding="utf-8")
        result = await _grep(str(tmp_path), "TODO", glob="*.py")
        assert "a.py" in result
        assert "b.txt" not in result


class TestGitTools:
    @pytest.mark.asyncio
    async def test_git_outside_repo(self, tmp_path):
        from weather_agents.tools.builtin import _git_status

        result = await _git_status(repo=str(tmp_path))
        assert "not a git repository" in result

    @pytest.mark.asyncio
    async def test_git_add_refuses_all(self):
        from weather_agents.tools.builtin import _git_add

        result = await _git_add("-A")
        assert "refusing" in result.lower()

    @pytest.mark.asyncio
    async def test_git_commit_empty_message(self):
        from weather_agents.tools.builtin import _git_commit

        result = await _git_commit("")
        assert "cannot be empty" in result.lower()


class TestTaskDone:
    def test_sentinel_value(self):
        from weather_agents.tools.builtin import TASK_DONE_SENTINEL

        assert TASK_DONE_SENTINEL == "__TASK_DONE__"

    @pytest.mark.asyncio
    async def test_task_done_returns_sentinel(self):
        from weather_agents.tools.builtin import TASK_DONE_SENTINEL, _task_done

        result = await _task_done(summary="test complete")
        assert result == TASK_DONE_SENTINEL

    @pytest.mark.asyncio
    async def test_task_done_empty_summary(self):
        from weather_agents.tools.builtin import TASK_DONE_SENTINEL, _task_done

        result = await _task_done()
        assert result == TASK_DONE_SENTINEL

    def test_tool_registered(self):
        from weather_agents.core.tool import ToolRegistry
        from weather_agents.tools.builtin import register_builtin_tools

        reg = ToolRegistry()
        register_builtin_tools(reg)
        tool = reg.get("task_done")
        assert tool is not None
        assert tool.name == "task_done"
        param_names = [p.name for p in tool.parameters]
        assert "summary" in param_names

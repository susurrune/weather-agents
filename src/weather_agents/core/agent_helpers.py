"""Module-level helpers for ``core.agent`` — parsing, signatures, similarity, labels.

Extracted from the 2.8k-line ``agent.py`` so the LLM-loop class isn't sharing a
file with ~560 lines of regex + string utilities. Pure functions / constants
only: no state, no agent reference, safe to import anywhere.

Tests import these via ``weather_agents.core.agent.<helper>``; ``agent.py``
re-exports the lot to preserve those import paths.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from weather_agents.core.memory import Message
    from weather_agents.core.tool import ToolRegistry


_TOOL_LABELS: dict[str, str] = {
    "read_file": "Reading {path}",
    "write_file": "Writing {path}",
    "edit_file": "Editing {path}",
    "list_directory": "Listing {path}",
    "file_search": "Searching {directory}/{pattern}",
    "code_search": "Searching for '{query}'",
    "grep": "Grepping '{pattern}'",
    "shell_exec": "Running: {command}",
    "http_get": "GET {url}",
    "http_post": "POST {url}",
    "web_search": "Searching: {query}",
    "move_file": "Moving {src}",
    "copy_file": "Copying {src}",
    "delete_file": "Deleting {path}",
    "get_cwd": "Getting working directory",
    "tree": "Tree {directory}",
    "lint_file": "Linting {path}",
    "scan_deps": "Scanning {directory}",
    "fetch_page": "Fetching {url}",
    "delegate_to": "Delegating to {agent}: {task}",
    "use_skill": "Activating {name}",
    "list_skills": "Listing available skills",
    "git_status": "Git status",
    "git_diff": "Git diff",
    "git_log": "Git log",
    "git_add": "Git add {files}",
    "git_commit": "Git commit",
    "git_checkout": "Git checkout {branch}",
    # Computer-operation tools
    "launch_app": "Launching {name}",
    "open_path": "Opening {target}",
    "browser_open": "Opening {url} in browser",
    "list_installed_apps": "Listing installed apps",
    "system_info": "System info",
    "system_diagnose": "System diagnosis",
    "list_processes": "Listing processes",
    "kill_process": "Killing {target}",
    "package_manager": "Package {action} {name}",
    "service_control": "Service {action} {name}",
    # Runtime MCP tools
    "mcp_list_servers": "Listing MCP servers",
    "mcp_add_server": "Adding MCP server {name}",
    "mcp_remove_server": "Removing MCP server {name}",
    "mcp_scaffold_server": "Scaffolding MCP server {name}",
    # Emotional memory
    "remember": "Remembering: {note}",
}


_RE_FENCED_JSON_ARRAY = re.compile(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```")
_RE_JSON_ARRAY = re.compile(r"\[[\s\S]*?\]")
_RE_OBJ_OR_ARRAY = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)
_RE_KV_DETECT = re.compile(r"\b\w[\w\d_]*\s*=")
_RE_KV_PAIRS = re.compile(r'(\w[\w\d_]*)\s*=\s*("[^"]*"|\'[^\']*\'|[\w\d_.+-]+)')
_RE_NONE_LITERAL = re.compile(r":\s*None\s*([,}])")
_RE_TRUE_LITERAL = re.compile(r":\s*True\s*([,}])")
_RE_FALSE_LITERAL = re.compile(r":\s*False\s*([,}])")
_RE_PY_NONE = re.compile(r"\bNone\b")
_RE_PY_TRUE = re.compile(r"\bTrue\b")
_RE_PY_FALSE = re.compile(r"\bFalse\b")
_RE_UNQUOTED_KEY = re.compile(r"([{,]\s*)(\w[\w\d_]*)(\s*:)")
_RE_TRAILING_COMMA = re.compile(r",\s*([}\]])")
_RE_UNQUOTED_STRING = re.compile(r"(:\s*)([a-zA-Z_.][a-zA-Z0-9_ ./\\@.\-+#~$]*?)(\s*[,}\]])")


def _parse_tool_args(raw: str) -> dict | None:
    """Parse tool call JSON with multi-stage repair for LLM output quirks.

    Handles: markdown fences, Python literals, backtick quotes, single quotes,
    unquoted keys, trailing commas, key=value syntax, unquoted string values,
    trailing text, and unbalanced braces. All regexes are precompiled at
    module load so the repair stages don't pay re-compile cost per call —
    the fast path (stage 1, plain json.loads) is unaffected but failure-
    repair latency drops ~5-20ms.
    """
    if not raw or not raw.strip():
        return None

    cleaned = raw.strip()

    # ── 1. Direct parse ────────────────────────────────────────────────────
    try:
        return cast(dict, json.loads(cleaned))
    except json.JSONDecodeError:
        pass

    # ── 2. Strip markdown code fences ──────────────────────────────────────
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned[3:]
        cleaned = cleaned.rsplit("```", 1)[0] if "```" in cleaned else cleaned
        cleaned = cleaned.strip()
        try:
            return cast(dict, json.loads(cleaned))
        except json.JSONDecodeError:
            pass

    # ── 3. Extract first JSON object/array from surrounding text ───────────
    obj_match = _RE_OBJ_OR_ARRAY.search(cleaned)
    if obj_match:
        cleaned = obj_match.group(1)
        try:
            return cast(dict, json.loads(cleaned))
        except json.JSONDecodeError:
            pass

    # ── 4. Key=value format: query="weather", count=5 → {"query": "weather", "count": 5}
    #    Typically from models that emit function-call-style rather than JSON.
    if not cleaned.startswith("{") and _RE_KV_DETECT.search(cleaned):
        kv_pairs: list[str] = []
        for m in _RE_KV_PAIRS.finditer(cleaned):
            key = m.group(1)
            val = m.group(2)
            if val.startswith("'") and val.endswith("'"):
                val = '"' + val[1:-1] + '"'
            kv_pairs.append(f'"{key}": {val}')
        if kv_pairs:
            json_str = "{" + ", ".join(kv_pairs) + "}"
            json_str = _RE_NONE_LITERAL.sub(r": null\1", json_str)
            json_str = _RE_TRUE_LITERAL.sub(r": true\1", json_str)
            json_str = _RE_FALSE_LITERAL.sub(r": false\1", json_str)
            return cast(dict, json.loads(json_str))

    # ── 5. Python → JSON literals ──────────────────────────────────────────
    #    Must happen before quote transformations to avoid corrupting strings.
    cleaned = _RE_PY_NONE.sub("null", cleaned)
    cleaned = _RE_PY_TRUE.sub("true", cleaned)
    cleaned = _RE_PY_FALSE.sub("false", cleaned)

    # ── 6. Backtick → double quote ────────────────────────────────────────
    cleaned = cleaned.replace("`", '"')

    # ── 7. Fix single-quote strings ────────────────────────────────────────
    if "'" in cleaned:
        cleaned = cleaned.replace("'", '"')

    # ── 8. Fix unquoted keys: {key: "value"} → {"key": "value"} ────────────
    cleaned = _RE_UNQUOTED_KEY.sub(r'\1"\2"\3', cleaned)

    # ── 9. Fix trailing commas before ] or } ───────────────────────────────
    cleaned = _RE_TRAILING_COMMA.sub(r"\1", cleaned)
    cleaned = cleaned.rstrip(",").strip()

    # ── 10. Fix unquoted string values: {"key": bare word} → {"key": "bare word"} ──
    cleaned = _RE_UNQUOTED_STRING.sub(
        lambda m: (
            m.group(0)
            if m.group(2) in ("null", "true", "false")
            or m.group(2).lstrip("-").replace(".", "").isdigit()
            or m.group(2).startswith(('"', "{", "["))
            else f'{m.group(1)}"{m.group(2)}"{m.group(3)}'
        ),
        cleaned,
    )

    # ── 11. Attempt parse ──────────────────────────────────────────────────
    try:
        return cast(dict, json.loads(cleaned))
    except json.JSONDecodeError:
        pass

    # ── 12. Balanced-brace extraction ──────────────────────────────────────
    depth = 0
    start = -1
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return cast(dict, json.loads(cleaned[start : i + 1]))
                except json.JSONDecodeError:
                    pass

    # If a JSON object was started but never closed, try auto-closing
    if start >= 0 and depth > 0:
        candidate = cleaned[start:] + "}" * depth
        try:
            return cast(dict, json.loads(candidate))
        except json.JSONDecodeError:
            pass

    return None


# Substrings that strongly indicate a tool call produced no useful output.
# Treated as "failure" for stuck-loop detection even when the tool returned
# success=True (e.g. web_search returning "No results found ..." is
# technically a successful call but semantically a dead-end).
_TOOL_FAILURE_MARKERS: tuple[str, ...] = (
    # Search backends returning nothing
    "no results found",
    "no matches for",  # grep / code_search returning empty
    # File ops on non-existent paths — the model's most common dead-end
    # when guessing import paths or searching the wrong tree.
    "file not found",
    "directory not found",
    "permission denied",
    # HTTP failures (4xx / 5xx)
    "status: 4",
    "status: 5",
    # Network / timeout
    "request timed out",
    "timed out",
    "connection refused",
    "name or service not known",
    "ssl",
    # Generic error prefixes from the tool layer
    "[error",
    "error: tool",  # invalid args, etc.
    "error: file",  # file not found wrapper
    "error: directory",
    "circuitbreakeropen",
    "execution failed:",
)


def _looks_like_failed_tool_result(result: str) -> bool:
    """Heuristic: does this tool result indicate a dead-end the LLM should
    stop retrying? Used by stuck-loop detection in _chat_stream_impl."""
    if not result:
        return True
    head = result[:300].lower()
    return any(m in head for m in _TOOL_FAILURE_MARKERS)


# Tools that produce on-disk artifacts. When the orchestration loop scans
# an agent's tool-call history, these are the calls whose `path`/`dst`
# argument names the deliverable.
_FILE_PRODUCING_TOOLS: dict[str, tuple[str, ...]] = {
    "write_file": ("path",),
    "edit_file": ("path",),
    "copy_file": ("dst", "destination"),
    "move_file": ("dst", "destination"),
}


def _extract_file_paths_from_messages(messages: list[Message]) -> list[str]:
    """Walk this task's assistant turns and pull out file paths that
    write_file / edit_file / copy_file / move_file successfully touched.

    The agent's chat reply often says only "已完成" — but the tool_calls on
    each assistant message record exactly which paths it wrote. The very
    next "tool" role message (with matching tool_call_id) tells us whether
    the call actually succeeded. We list successful-write paths in the
    order they were performed, deduplicated.
    """
    # Pair each tool_call_id with the tool result message that followed.
    tool_results: dict[str, str] = {}
    for m in messages:
        if m.role == "tool" and m.tool_call_id:
            tool_results[m.tool_call_id] = m.content or ""

    paths: list[str] = []
    seen: set[str] = set()
    for m in messages:
        if m.role != "assistant" or not m.tool_calls:
            continue
        for tc in m.tool_calls:
            name = tc.get("function", {}).get("name", "")
            arg_keys = _FILE_PRODUCING_TOOLS.get(name)
            if not arg_keys:
                continue
            raw = tc.get("function", {}).get("arguments", "")
            try:
                args = json.loads(raw) if isinstance(raw, str) else (raw or {})
            except json.JSONDecodeError:
                continue
            if not isinstance(args, dict):
                continue
            path: str | None = None
            for k in arg_keys:
                v = args.get(k)
                if isinstance(v, str) and v.strip():
                    path = v.strip()
                    break
            if not path:
                continue
            # Confirm the call actually succeeded — skip "File not found" /
            # "permission denied" results. The tool wrapper returns plain
            # strings; a successful write_file starts with "Successfully ".
            result = tool_results.get(tc.get("id", ""), "")
            if result and result.startswith(("Error:", "[Error", "Error ")):
                continue
            if path not in seen:
                seen.add(path)
                paths.append(path)
    return paths


def _enrich_response_with_artifacts(content: str, file_paths: list[str]) -> str:
    """Append a deterministic artifact footer when the agent wrote files
    during this task but its reply doesn't already mention them.

    Renders as a markdown blockquote with each path wrapped in backticks
    so Rich's Markdown renderer gives the section a distinctive left bar
    + code highlighting. Plain "- path" lines blended into the rest of
    the reply and users frequently missed them.

    Without this, an agent that writes content to disk and replies with
    a 5-char "完成了" leaves the downstream verifier and the user with
    NO way to find the actual deliverable. The footer is concise — just
    the paths — so it doesn't disrupt rich replies that already include
    full content.
    """
    if not file_paths:
        return content
    body = content or ""
    # If every path is already cited in the reply, nothing to add.
    missing = [p for p in file_paths if p not in body]
    if not missing:
        return body
    # Markdown blockquote (>) gives Rich a left-bar style; inline code
    # (`path`) gives the path a distinct color from the surrounding text.
    lines = ["", "> **Artifacts produced**"]
    for p in file_paths:
        marker = " — already cited above" if p in body else ""
        lines.append(f"> - `{p}`{marker}")
    return body.rstrip() + "\n\n" + "\n".join(lines)


# Tool-signature loop detector tuning. Window covers ~last 8 tool calls.
# Hint at 4 repeats (model has space to course-correct on the next round);
# hard-escape at 6 (model already saw the hint and ignored it).
_SIG_WINDOW = 8
_SIG_LOOP_HINT = 4
_SIG_LOOP_HARDSTOP = 6


def _tool_call_signature(tool_name: str, args: dict | None) -> str:
    """Compact, stable fingerprint of a tool call for loop detection.

    For file-mutating tools we anchor on the path: editing the same file
    repeatedly is the classic micro-tuning loop. For shell we anchor on
    the leading command word (subsequent flags vary but the intent
    doesn't). For search we anchor on the lowercased query head.
    Generic fallback is the tool name alone — coarser but safer than
    inventing a key from arguments we don't understand.
    """
    a = args or {}

    def _s(key: str, n: int = 0) -> str:
        v = a.get(key)
        if not isinstance(v, str):
            return ""
        return v.strip()[:n] if n > 0 else v.strip()

    if tool_name == "edit_file":
        # Include old_string hash so different edits to the same file
        # aren't conflated as a loop. Only truly identical edits count.
        os_val = _s("old_text")
        if os_val:
            import hashlib as _hl

            return f"edit_file:{_s('path')}#{_hl.sha1(os_val.encode()).hexdigest()[:8]}"
        return f"edit_file:{_s('path')}"
    if tool_name in ("write_file", "read_file", "delete_file"):
        return f"{tool_name}:{_s('path')}"
    if tool_name in ("copy_file", "move_file"):
        return f"{tool_name}:{_s('src') or _s('source')}->{_s('dst') or _s('destination')}"
    if tool_name in ("run_bash", "bash", "shell", "run_shell"):
        cmd = _s("command") or _s("cmd")
        if not cmd:
            return tool_name
        # First whitespace-separated token captures the program; we drop
        # arguments so "where soffice" called with cosmetic flag changes
        # still folds to the same signature.
        first = cmd.split(None, 1)[0] if cmd else ""
        return f"{tool_name}:{first[:30]}"
    if tool_name in ("web_search", "search", "search_web", "search_files", "grep"):
        q = _s("query", 40) or _s("pattern", 40)
        # Normalize: strip whitespace and quotes so "良驹汽车 毕节"
        # and "良驹汽车毕节" and "\"良驹汽车\" 毕节" collapse to the
        # same signature. Without this the LLM can cycle through
        # dozens of cosmetic query variations without triggering the
        # loop detector.
        import re as _re

        q = _re.sub(r"[\s\"'「」『』" "''‘’“”]", "", q)
        return f"{tool_name}:{q.lower()[:30]}"
    if tool_name in ("fetch_page", "fetch_web_page", "http_get", "http_post"):
        # Anchor on the URL — fetching many *different* pages is normal research
        # progress, not a loop. Without this these tools fell to the generic
        # tool-name fallback below, so visiting 6 distinct URLs collapsed to one
        # signature and falsely tripped the hard-stop.
        return f"{tool_name}:{_s('url')[:60]}"
    if tool_name == "delegate_to":
        return f"delegate_to:{_s('agent')}"
    # Generic fallback: fold a short fingerprint of the args so distinct calls
    # to the same tool (different targets/params) aren't conflated into a false
    # loop. Only genuinely identical calls collapse to the same signature.
    if a:
        import hashlib as _hl
        import json as _json

        try:
            blob = _json.dumps(a, sort_keys=True, ensure_ascii=False, default=str)
        except Exception:
            blob = str(a)
        return f"{tool_name}:{_hl.sha1(blob[:300].encode()).hexdigest()[:8]}"
    return tool_name


def _text_similarity(a: str, b: str) -> float:
    """Cheap similarity for narration-loop detection.

    difflib's SequenceMatcher.ratio is O(N*M) which is fine for the
    typical 100-400 char per-round responses we compare. Returns 0-1.
    Short responses (< 12 chars) are always treated as dissimilar to
    avoid flagging "好的" / brief acknowledgements as loops.
    """
    if len(a) < 12 or len(b) < 12:
        return 0.0
    import difflib as _difflib

    return _difflib.SequenceMatcher(None, a, b).ratio()


def _format_args_parse_error(tool_name: str, raw_args: str) -> str:
    """Produce a tool-result error string that helps the LLM recover.

    Detects the common failure mode where the model hit max_tokens
    mid-content (huge write_file payloads, etc.) and the JSON tool args
    end in an unclosed string. In that case "Invalid JSON" is misleading
    — the args parsed fine syntactically up to where they stopped — so we
    point the model at the real fix (chunk the content or use edit_file).

    The truncation heuristic: count unescaped quotes; if odd, the last
    string was never closed. Combined with no trailing brace, that
    practically always means max_tokens cut the response.
    """
    stripped = raw_args.rstrip()
    has_closing_brace = stripped.endswith(("}", "]"))
    # Crude quote counter — treat backslash as escape one char ahead.
    in_string = False
    i = 0
    while i < len(stripped):
        ch = stripped[i]
        if ch == "\\" and in_string:
            i += 2
            continue
        if ch == '"':
            in_string = not in_string
        i += 1
    looks_truncated = in_string or not has_closing_brace

    preview = raw_args[:200] + ("...[truncated]" if len(raw_args) > 200 else "")
    if looks_truncated:
        return (
            f"Error: tool '{tool_name}' arguments were truncated by the model's "
            f"output budget (max_tokens). The JSON ended mid-value so it cannot "
            f"be parsed. For large content, split into multiple smaller calls "
            f"(write_file with a shorter chunk, then edit_file or write_file to "
            f"append the rest), or break the deliverable into pieces. "
            f"Args preview: {preview}"
        )
    return f"Error: invalid JSON in tool call arguments for '{tool_name}': {preview}"


def _suggest_tool_names(missing: str, registry: ToolRegistry, max_n: int = 3) -> list[str]:
    """Return the closest existing tool names for a hallucinated name.

    Three-tier matching, accepting the first tier that produces hits:

    1. difflib sequence-ratio match — catches typos like ``fetch_pag`` →
       ``fetch_page``.
    2. Token substring overlap on the name itself — catches misses where
       both names share at least one underscore-separated chunk.
    3. Token overlap against each tool's *description* — catches purely
       conceptual misses like ``fetch_page`` → ``http_get`` where the names
       share nothing but the description mentions "fetch a web page". This
       is what the LLM is actually reaching for; it tends to invent a name
       that matches the capability, not our internal naming.

    Without these suggestions the LLM repeats the same hallucinated name
    every iteration and burns through the tool-round budget.
    """
    import difflib as _difflib

    all_names = registry.list_names()
    if not all_names:
        return []
    close = _difflib.get_close_matches(missing, all_names, n=max_n, cutoff=0.5)
    if close:
        return close

    missing_lower = missing.lower()
    missing_chunks = [c for c in missing_lower.split("_") if len(c) >= 3]
    if not missing_chunks:
        missing_chunks = [missing_lower] if len(missing_lower) >= 3 else []

    name_scored: list[tuple[int, str]] = []
    desc_scored: list[tuple[int, str]] = []
    for n in all_names:
        nlow = n.lower()
        name_score = 0
        for chunk in missing_chunks:
            if chunk in nlow:
                name_score += 2
        for chunk in nlow.split("_"):
            if len(chunk) >= 3 and chunk in missing_lower:
                name_score += 1
        if name_score > 0:
            name_scored.append((name_score, n))

        tool = registry.get(n)
        if not tool:
            continue
        desc = tool.description.lower()
        desc_score = sum(1 for chunk in missing_chunks if chunk in desc)
        if desc_score > 0:
            desc_scored.append((desc_score, n))

    if name_scored:
        name_scored.sort(key=lambda x: x[0], reverse=True)
        return [n for _s, n in name_scored[:max_n]]
    desc_scored.sort(key=lambda x: x[0], reverse=True)
    return [n for _s, n in desc_scored[:max_n]]


def _tool_status_label(name: str, args: dict) -> str:
    """Build a human-readable one-liner for a tool call."""
    template = _TOOL_LABELS.get(name)
    if template:
        try:
            label = template.format_map(args)
        except (KeyError, IndexError):
            label = f"{name}..."
    else:
        label = f"{name}..."
    # Truncate long labels — keep enough room that paths are visible.
    if len(label) > 100:
        label = label[:97] + "..."
    return label


def _synthesize_delegation_summary(delegations: list[tuple[str, bool]]) -> str:
    """Build a short fallback line shown when a turn ends with delegate_to
    calls but no plain text from the model.

    Keeps the REPL's "empty response" guard from firing while making it
    clear to the user that work happened — without leaking the target
    agent's full response into the caller's voice.
    """
    if not delegations:
        return ""
    ok = [name for name, success in delegations if success]
    failed = [name for name, success in delegations if not success]
    parts: list[str] = []
    if ok:
        parts.append("Delegated: " + ", ".join(ok))
    if failed:
        parts.append("Failed: " + ", ".join(failed))
    return "[" + " | ".join(parts) + "]"

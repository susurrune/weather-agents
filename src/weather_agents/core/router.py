"""Complexity router: classify a user goal into direct / single / orchestrate.

Rules-first (no LLM), so classification stays under 1ms. The router exists
solely to keep simple goals from triggering Snow's full task-decomposition
LLM call when a single agent could answer in one shot.
"""

from __future__ import annotations

import re
from typing import Literal

Mode = Literal["direct", "single", "orchestrate"]

_MULTI_STEP_TOKENS = (
    "先",
    "再",
    "然后",
    "接着",
    "之后",
    "其次",
    "最后",
    "第一步",
    "第二步",
    "第三步",
    "步骤",
    "顺序",
    "依次",
    "首先",
    "step 1",
    "step 2",
    "first,",
    "then,",
    "after that",
    "finally",
)

_GREETING_TOKENS = (
    "你好",
    "您好",
    "hi",
    "hello",
    "hey",
    "在吗",
    "嗨",
    "早上好",
    "晚安",
    "谢谢",
    "thanks",
    "thank you",
    "再见",
    "bye",
)

# Single strong action verb implies single-agent work, not orchestration.
_SINGLE_ACTION_HINTS = (
    "解释",
    "什么是",
    "为什么",
    "如何",
    "怎么",
    "查询",
    "搜索",
    "搜一下",
    "翻译",
    "总结",
    "summarize",
    "explain",
    "what is",
    "why",
    "how do",
)

# Verbs that imply tool use / actual work — bumps short messages from direct → single.
_ACTION_VERBS = (
    "写",
    "帮我写",
    "生成",
    "创建",
    "实现",
    "做",
    "搜",
    "查",
    "找",
    "审查",
    "审计",
    "翻译",
    "重构",
    "修改",
    "改",
    "部署",
    "运行",
    "执行",
    "write",
    "create",
    "generate",
    "implement",
    "search",
    "find",
    "review",
    "translate",
    "deploy",
    "run",
)

_CODE_BLOCK = re.compile(r"```[\s\S]+?```")
_URL = re.compile(r"https?://\S+")
_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|[\\/])[\w\-./\\]+")
_NUMBERED_LIST = re.compile(r"(?:^|\n)\s*(?:\d+[.)、]|[-*])\s+", re.MULTILINE)
# Inline form: "1. X 2. Y 3. Z" on a single line (no newline anchor).
# Three+ enumerations within the same line implies a sequenced plan.
_INLINE_ENUMERATED = re.compile(r"\b\d+[.)、]\s*\S")


def classify(goal: str) -> Mode:
    """Decide the execution mode for a user goal.

    direct: short greeting / single factual question, no tools needed.
    single: clear single-purpose task, one agent + tools.
    orchestrate: multi-step plan worth decomposing into sub-tasks.
    """
    if not goal or not goal.strip():
        return "direct"

    text = goal.strip()
    lower = text.lower()
    length = len(text)

    has_code = bool(_CODE_BLOCK.search(text))
    has_url = bool(_URL.search(text))
    has_path = bool(_PATH.search(text))
    list_matches = len(_NUMBERED_LIST.findall(text))
    inline_enum_hits = len(_INLINE_ENUMERATED.findall(text))

    multi_step_hits = sum(1 for tok in _MULTI_STEP_TOKENS if tok in lower)
    greeting_hits = sum(1 for tok in _GREETING_TOKENS if tok in lower)
    single_hits = sum(1 for tok in _SINGLE_ACTION_HINTS if tok in lower)
    action_hits = sum(1 for tok in _ACTION_VERBS if tok in lower)

    if greeting_hits >= 1 and length < 30 and multi_step_hits == 0 and action_hits == 0:
        return "direct"

    if multi_step_hits >= 2 or list_matches >= 2 or inline_enum_hits >= 3:
        return "orchestrate"

    if length > 200 and multi_step_hits >= 1:
        return "orchestrate"

    if has_code and length > 150:
        return "orchestrate"

    # Tool-use signals push toward single, not direct.
    if has_path or has_url or action_hits >= 1:
        return "single"

    if length < 50 and not has_code:
        # Short factual/explanation questions with no action verb go direct.
        if single_hits >= 1 or text.endswith("?") or text.endswith("？"):
            return "direct"
        if multi_step_hits == 0:
            return "direct"

    return "single"


def pick_agent_for_goal(goal: str, available: set[str]) -> str:
    """Pick a single agent for a non-orchestrate goal, by keyword routing.

    Returns an agent name guaranteed to be in `available`, falling back to
    rain (generation generalist) then to any available agent.
    """
    lower = goal.lower()

    # Order matters: more specific buckets first.
    buckets: list[tuple[str, tuple[str, ...]]] = [
        ("frost", ("审查", "review", "漏洞", "安全", "审计", "lint", "重构建议", "code smell")),
        ("dew", ("部署", "运行", "执行命令", "shell", "deploy", "ci", "cd", "环境变量", "运维")),
        # "找" was too greedy — matched 二分查找 / 找回密码 / 找一个函数 etc.
        # Use longer, less ambiguous markers and rely on rain's bucket for code.
        (
            "fog",
            (
                "研究",
                "调研",
                "搜一下",
                "搜索",
                "查一下",
                "查资料",
                "research",
                "search",
                "调查",
                "找一下",
                "找资料",
            ),
        ),
        (
            "rain",
            (
                "写",
                "生成",
                "实现",
                "create",
                "generate",
                "写一段",
                "写个",
                "代码",
                "二分",
                "函数",
                "实现一个",
            ),
        ),
        (
            "fair",
            ("陪我", "聊天", "心情", "难过", "开心", "孤独", "倾诉", "你好", "hi", "hello", "嗨"),
        ),
    ]

    for agent, hints in buckets:
        if agent not in available:
            continue
        if any(h in lower for h in hints):
            return agent

    if "rain" in available:
        return "rain"
    return next(iter(available))

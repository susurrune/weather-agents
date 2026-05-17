"""Tests for the complexity router."""

from __future__ import annotations

import pytest

from weather_agents.core.router import classify, pick_agent_for_goal


@pytest.mark.parametrize(
    "goal",
    [
        "你好",
        "hi",
        "在吗",
        "谢谢",
        "什么是 RAG?",
        "为什么天空是蓝的？",
        "1 + 1 = ?",
        "解释一下闭包",
    ],
)
def test_direct_simple_questions(goal: str) -> None:
    assert classify(goal) == "direct"


@pytest.mark.parametrize(
    "goal",
    [
        "帮我写一个二分查找函数",
        "搜一下今天的天气",
        "审查 src/foo.py 的安全问题",
        "把这段中文翻译成英文：我喜欢猫",
        "在 src/utils/ 下找所有用到 deprecated 函数的地方",
    ],
)
def test_single_focused_tasks(goal: str) -> None:
    assert classify(goal) == "single"


@pytest.mark.parametrize(
    "goal",
    [
        "先帮我分析这段代码，然后重构它，最后写测试",
        "首先调研一下市场上有哪些方案，其次对比性能，最后给出推荐",
        "1. 创建数据库迁移\n2. 写 API\n3. 加测试\n4. 部署",
        "step 1: design the schema. step 2: write migrations. then, write tests. finally deploy.",
    ],
)
def test_orchestrate_multi_step(goal: str) -> None:
    assert classify(goal) == "orchestrate"


def test_empty_goal_is_direct() -> None:
    assert classify("") == "direct"
    assert classify("   ") == "direct"


def test_long_code_block_is_orchestrate() -> None:
    goal = (
        "帮我处理这段代码并先重构再加测试：\n"
        "```python\n"
        + "def foo():\n    return 1\n" * 30
        + "```\n"
    )
    assert classify(goal) == "orchestrate"


class TestPickAgent:
    def test_security_keyword_picks_frost(self) -> None:
        available = {"fog", "rain", "frost", "snow", "dew", "sunshine"}
        assert pick_agent_for_goal("帮我做安全审查", available) == "frost"

    def test_research_keyword_picks_fog(self) -> None:
        available = {"fog", "rain", "frost", "snow", "dew", "sunshine"}
        assert pick_agent_for_goal("搜一下最新的 React 文档", available) == "fog"

    def test_greeting_picks_sunshine(self) -> None:
        available = {"fog", "rain", "frost", "snow", "dew", "sunshine"}
        assert pick_agent_for_goal("你好啊", available) == "sunshine"

    def test_falls_back_to_rain(self) -> None:
        available = {"fog", "rain", "frost", "snow", "dew", "sunshine"}
        assert pick_agent_for_goal("处理这个东西", available) == "rain"

    def test_skips_missing_agents(self) -> None:
        available = {"rain", "snow"}
        # security would pick frost, but frost not available → fall through
        result = pick_agent_for_goal("做安全审查", available)
        assert result in available

    def test_single_agent_available(self) -> None:
        assert pick_agent_for_goal("anything", {"rain"}) == "rain"

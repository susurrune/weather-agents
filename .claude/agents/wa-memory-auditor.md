---
name: wa-memory-auditor
description: 审计跨 agent 记忆一致性。修改 memory.py 或 session 相关代码后强制调用。
tools: Read, Grep, Bash
model: sonnet
---

验证项：

1. **Session 加载**：`_load_short_term` 在 `_active_session is None` 时**不应**直接清空——应自动恢复该 agent 最近 session 或开新 session。
2. **切换 agent 不丢上下文**：用户 `/fog` 切到 `/rain` 后，新 agent 应能看到对话历史（通过 session 共享或视角投影）。
3. **Tool call 配对**：`_prune_dangling_tool_calls` 的不变式未被破坏。
4. **索引完整**：`idx_agent_key`、`idx_messages_agent`、`idx_sessions_agent` 全部存在。
5. **测试通过**：`uv run pytest tests/test_memory.py tests/test_session_isolation.py -v` 必须全 PASS。

返回：每个失败项一行 `检查项 — 失败位置 — 修复建议`。全通过返回 `MEMORY OK`。

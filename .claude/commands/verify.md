---
description: 一键 lint + typecheck + test，并行执行
allowed-tools: Bash, Agent
---

在同一条消息内并行运行：

- `uv run ruff check src tests`
- `uv run mypy src`
- `@wa-test-runner` 跑 pytest 摘要

全过：返回 `OK`。
任一失败：按 `path:line — error` 列清单，不超过 30 行。

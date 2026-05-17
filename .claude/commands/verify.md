---
description: 一键 lint + typecheck + test，并行执行
allowed-tools: Bash, Agent
---

在同一条消息内并行运行（必须**同时**跑 lint 与 format check，CI 都会跑）：

- `uv run ruff check src tests`
- `uv run ruff format --check src tests`   ← 容易漏，CI 会因此 fail
- `uv run mypy src --ignore-missing-imports`
- `@wa-test-runner` 跑 pytest 摘要（带 `--cov=src --cov-fail-under=55` 与 CI 对齐）

全过：返回 `OK`。
任一失败：
- 若是 `ruff format`：直接 `uv run ruff format src tests` 修复后 commit。
- 其余：按 `path:line — error` 列清单，不超过 30 行。

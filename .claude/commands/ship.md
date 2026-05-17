---
description: 一组修改完成后的标准收尾流程
allowed-tools: Bash, Agent
---

顺序执行：

1. `/verify`
2. `@wa-code-reviewer` 复核 `git diff`
3. 给用户列改动摘要，**等待确认**
4. 用户确认后 commit（conventional commits 格式：`feat: ...` / `fix: ...` / `refactor: ...`）

绝不：`git push`、`--no-verify`、跳过 `wa-code-reviewer` 直接 commit。

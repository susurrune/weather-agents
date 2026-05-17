---
description: 给 factory.orchestrate_task 与 cli/main.py 增加复杂度路由，简单 goal 走 fast path
allowed-tools: Read, Edit, Write, Bash, Agent
---

目标：消除 `wa task` 在简单 goal 下触发的多 agent 编排（避免 3× LLM 调用）。

步骤：

1. 读 `src/weather_agents/core/router.py` 与 `src/weather_agents/core/factory.py:orchestrate_task`。
2. 委派 `@wa-router-designer` 复核 `classify` 规则覆盖度。
3. 确认 `orchestrate_task` 入口已经调 `classify(goal)`，对 `direct/single` 走单 agent 快速路径。
4. 并行：`@wa-test-runner tests/test_router.py` + `@wa-code-reviewer`。
5. `/verify`。

不要：把 `classify` 改成调 LLM；把路由结果缓存在全局变量。

# Weather Agents — Claude 工作守则

## 项目身份
Python 3.11+ 多智能体 CLI；LiteLLM + Typer + Rich + aiosqlite。
六个 agent：fog（研究）/ rain（生成）/ frost（审查）/ snow（编排）/ dew（运维）/ sunshine（陪伴）。
入口：`src/weather_agents/cli/main.py`；核心：`src/weather_agents/core/`。

## 命令
- 开发：`uv run wa chat`
- 测试：`uv run pytest -x`
- Lint：`uv run ruff check src tests`
- 类型：`uv run mypy src`

## 不可逾越的红线
1. 不在 `Edit`/`Write` 之前挂任何 LLM 调用——会让 hook 死循环。
2. 改 `core/memory.py` 必须保持 `_prune_dangling_tool_calls` 的不变式（tool 消息前必须有匹配 tool_calls 的 assistant）。
3. 修改完毕前必须 `uv run pytest -x` 通过；失败禁止 commit。
4. 不新增对 root 的 `shell_exec`；不绕过 `WA_ALLOW_PRIVATE_NET`。
5. 严禁 `git push`、`--no-verify`、`reset --hard` 除非用户明确说。

## 修改约定
- 简单 bug 修复：直接改，不写计划文档。
- 跨模块改造（router / memory / display）：先 `/plan`，再委派 subagent。
- 任何对 `snow.py orchestrate` 或 `factory.orchestrate_task` 的修改必须过 `wa-router-designer`。
- 大块输出（pytest 全量、grep 整库）委派 `wa-test-runner` 或 Explore subagent，只收摘要。
- 简单问答（< 50 字、无多步动词链）走 router 的 `direct` / `single` 路径，**不要调 Snow**。

## 性能预算
- `wa chat` 简单问答：< 1 次 LLM 调用、< 2s 首字节。
- `wa task` 复杂目标：完整 orchestration；简单目标自动降级到 single agent。
- Snow `orchestrate` 仅在 `classify(goal) == "orchestrate"` 时被调用。

## 注释规范
- 注释只写 WHY（隐藏约束、不变式、历史教训），不写 WHAT。
- 不要在新代码里复刻 `memory.py` 那种段落级注释——单行 WHY 就够。

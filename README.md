<div align="center">

# Weather Agents

**雾 · 雨 · 霜 · 雪 · 露 · 晴**

*六位 Agent，一支团队。专精领域，默契协作。*

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/susurrune/weather-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/susurrune/weather-agents/actions)
[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/susurrune/weather-agents)
[![Tests](https://img.shields.io/badge/tests-736_🌡️-8A2BE2)](https://github.com/susurrune/weather-agents)
[![Code style](https://img.shields.io/badge/code%20style-ruff-000000)](https://github.com/astral-sh/ruff)

</div>

---

Weather Agents 是一个**本地优先的多智能体终端框架**。六个 Agent 各司其职，通过事件总线通信、技能系统增强、编排引擎协作，完成从研究分析到代码生成到部署运维的完整工作流。

它不是又一个大模型聊天客户端——它是一个**分工明确的 AI 团队**。

```bash
# 安装
pipx install git+https://github.com/susurrune/weather-agents.git

# 交互式对话
wa

# 一句话让团队协作
wa task "设计并实现一个 URL 短链接服务"
```

---

## 一、设计哲学

| 原则 | 含义 |
|:-----|:------|
| **规则优先，LLM 兜底** | 能用路由、Pipeline、关键词匹配解决的问题，绝不调 LLM。每次 LLM 调用都必须有正当理由 |
| **Agent 角色稳定** | 六位 Agent 的人格与职能不随版本漂移。新能力通过技能（Skill）和管道（Pipeline）加入 |
| **本地优先** | 核心功能不依赖云服务。Ollama 离线可用；Web 是补充，不是替代 |
| **诚实透明** | 功能描述基于代码事实，不过度承诺。费用、token、上下文使用情况均可实时查看 |

## 二、六位 Agent

| Agent | 标识 | 职责 | 核心技能 |
|:------|:-----|:------|:---------|
| **Fog** 雾 | `bright_magenta` | 探索研究 | `web_research` · `code_analysis` · `document_analysis` |
| **Rain** 雨 | `bright_blue` | 生成创造 | `code_generator` · `content_writer` · `data_transformer` |
| **Frost** 霜 | `bright_cyan` | 审查优化 | `code_reviewer` · `security_auditor` · `performance_checker` |
| **Snow** 雪 | `bright_white` | 规划编排 | `task_planner` · `arch_designer` · `workflow_designer` |
| **Dew** 露 | `bright_green` | 运维集成 | `sys_operator` · `ci_cd_manager` · `api_integrator` |
| **Fair** 晴 | `gold` | 情感陪伴 | `emotional_companion` · `self_evolve` |

### 使用场景

```bash
# 让 Fog 做调研
wa chat fog "对比 FastAPI 和 Flask 的生态"

# 让 Rain 写代码
wa chat rain "实现一个带超时的 LRU Cache"

# 全团队协作
wa task "搭建微服务项目：FastAPI + PostgreSQL + Docker"
```

协作模式下，Snow 自动拆解目标为 DAG 任务，分配至对应 Agent，按依赖顺序执行并汇总结果：

```
wa task "搭建微服务项目"
  ├─ [1] Fog: 调研微服务最佳实践
  ├─ [2] Rain: 生成项目骨架          ← 依赖 1
  ├─ [3] Rain: 编写 Dockerfile       ← 依赖 2
  ├─ [4] Frost: 代码审查             ← 依赖 2
  ├─ [5] Dew: 部署验证               ← 依赖 3, 4
```

## 三、快速开始

### 安装

```bash
pipx install git+https://github.com/susurrune/weather-agents.git
```

> 装完后 `wa` 会出现在 `~/.local/bin` 或 `%USERPROFILE%\.local\bin`。如果命令不可用，运行 `pipx ensurepath` 重开终端。

### 初始化

首次运行自动进入设置向导，选择统一模型或每个 Agent 独立配置。也可手动设置：

```bash
wa init                              # 重新跑向导
wa config set api_key.deepseek sk-xxx # 直接写入
export DEEPSEEK_API_KEY=sk-xxx        # 或环境变量
```

### 升级 / 卸载

```bash
pipx upgrade weather-agents
pipx uninstall weather-agents
```

## 四、功能总览

### 智能路由

用户输入自动分三级处理，< 1ms 决策：

| 级别 | 触发条件 | 行为 |
|:-----|:---------|:-----|
| `direct` | 问候、简短问答 | 单 Agent 直接回复，0 次额外 LLM 调用 |
| `single` | 明确单领域请求 | 关键词路由到最佳 Agent 单轮对话 |
| `orchestrate` | 复杂/跨领域目标 | Snow 拆解 → Pipeline 或 DAG 编排 |

### Pipeline 模板系统

预定义的多步骤工作流，命中后跳过 Snow 的 LLM 拆解调用：

- `code_review` — 代码审查
- `research_then_write` — 调研→写作
- `implement_and_review` — 实现→审查
- `implement_test_deploy` — 实现→测试→部署

### 技能系统

15 个可组合技能，运行时动态激活，为 Agent 注入专业能力：

```bash
# 进入 Frost，激活安全审计模式
wa chat frost
> /use security_auditor
> 审计这段 Go 代码
> /deactivate  # 回到基础模式
```

### 三层记忆

| 层级 | 范围 | 存储 | 用途 |
|:-----|:------|:------|:------|
| **Short-term** | 会话级 | SQLite | 对话上下文，自动截断 + 去悬挂 tool_call |
| **Working** | 任务级 | In-memory | 任务执行中的临时状态 |
| **Long-term** | 持久 | SQLite KV | 带分类的知识记忆，模糊搜索 |

### 15 内置工具

| 分类 | 工具 |
|:-----|:-----|
| 文件 | `read_file` · `write_file` · `edit_file` · `move_file` · `copy_file` · `delete_file` |
| 目录 | `list_directory` · `tree` · `file_search` · `code_search` |
| Shell | `shell_exec`（安全模式） |
| 网络 | `http_get` · `http_post` · `web_search`（DuckDuckGo，免 API Key） |
| Git | `git_status` · `git_diff` · `git_log` · `git_add` · `git_commit` · `git_checkout` |
| 委派 | `delegate_to`（Agent 间任务委派） |

### Multi-Provider LLM

通过 LiteLLM 接入多家模型，每个 Agent 可独立配置：

```
OpenAI:      gpt-4o · gpt-4o-mini · gpt-4.1 系列
Anthropic:   claude-opus-4-7 · claude-sonnet-4-6 · claude-haiku-4-5
DeepSeek:    deepseek-v4-flash · deepseek-v4-pro
Ollama:      llama3 · qwen2.5 · deepseek-r1（本地，离线可用）
```

### 更多功能

- **MCP 协议支持** — 通过 stdio 接入 Model Context Protocol 工具集
- **Plugin 系统** — `~/.weather-agents/plugins/` 目录自动加载自定义工具
- **语音聊天** — 晴（Fair）专属，Doubao TTS V3 HTTP + WebSocket 语音服务器
- **会话管理** — 创建/加载/删除会话，`wa chat --new` 开新会话
- **费用追踪** — 各 Agent 累计 Token 和费用，支持预算上限
- **智能工作空间** — 自动检测最佳磁盘位置，多盘跳过 C 盘
- **安全默认值** — Shell 执行拒危险命令，HTTP 请求禁私网，长输出截断标记
- **上下文压缩** — 接近限制时智能摘要，保留关键指令

## 五、CLI 参考

### 子命令

```bash
wa init              # 交互式配置向导
wa chat [agent] [msg] # 对话（默认 fog）
wa task <goal>       # 多 Agent 协作
wa status            # 所有 Agent 状态
wa config <action>   # 配置管理
wa memory <action>   # 记忆管理
```

### 交互命令

进入 `wa chat` 后：

| 命令 | 用途 |
|:-----|:------|
| `/fog` · `/rain` · `/frost` · `/snow` · `/dew` · `/fair` | 切换 Agent |
| `/task <目标>` | 多 Agent 任务编排 |
| `/skills` · `/use <skill>` · `/deactivate` | 技能管理 |
| `/status` | Agent 状态一览 |
| `/model [agent] [name]` | 查看/切换模型 |
| `/cost` / `/cost reset` | 费用追踪 |
| `/memory` / `/memory clear` | 记忆状态 |
| `/history` | 事件日志 |
| `/mcp` | MCP 服务器状态 |
| `/workspace` / `/workspace set <path>` / `/workspace auto` | 工作空间 |
| `/sessions` · `/session new|load|delete` | 会话管理 |
| `/compact` | 主动压缩上下文 |
| `/help` · `/clear` · `/quit` | 通用 |

## 六、项目状态

> 以下数据基于代码事实：

- **代码**：56 个源文件，~34k 行 Python
- **测试**：736 项，覆盖率 > 63%
- **提交**：141 个 commits
- **CI**：GitHub Actions — Ruff + MyPy + Pytest × 3 个 Python 版本

### 已落地能力

| 能力 | 状态 |
|:-----|:------|
| 6 Agent 独立角色 + 独立 LLM 配置 | ✅ 完成 |
| 三级复杂度路由（direct/single/orchestrate） | ✅ 完成，< 1ms |
| Pipeline 模板（4 个内置 DAG） | ✅ 完成，命中跳过编排 LLM 调用 |
| DAG 任务编排 + 依赖数据注入 | ✅ 完成，自动传上游产出 |
| 三层记忆（short/working/long-term） | ✅ 完成 |
| 15 内置工具 + MCP + Plugin | ✅ 完成 |
| 多 Provider LLM（OpenAI/Anthropic/DeepSeek/Ollama） | ✅ 完成 |
| 会话管理 + Session Resume | ✅ 完成 |
| 语音聊天（Doubao TTS） | ✅ 完成 |
| 费用追踪 + 预算控制 | ✅ 完成 |
| 安全守卫（Shell/HTTP/截断） | ✅ 完成 |
| 上下文压缩 + 智能摘要 | ✅ 完成 |
| Agent 间委派（delegate_to） | ✅ 完成 |
| 交互式命令弹出 + 自动补全 | ✅ 完成 |
| Claude Code 自动化骨架 | ✅ 完成 |

## 七、极致计划

> 现有功能已经可用，但离"极致"还有距离。以下路线图的目标是：**把每一项既有能力打磨到工业级品质**，不追求新功能数量，追求每项功能的可靠性、性能和用户体验。

### Phase 1 · 可靠性根基（Q3 2026）

| 项目 | 现状 | 目标 |
|:-----|:------|:------|
| **测试覆盖** | 736 tests, 63% | **90%+ 行覆盖率，-x 零容忍**。补全集成测试、边界测试、异常路径测试 |
| **属性测试** | 无 | **引入 Hypothesis**，对 memory 读写、tool 参数校验、路由分类做基于属性的随机测试 |
| **性能基准** | 无 | **CI 中嵌入基准测试**。`wa chat` 启动 < 500ms，简单问答 < 2s 首 token，编排 < 5s 出首任务 |
| **类型覆盖** | mypy 通过（196 `--strict` 错误） | 启用 `--strict`，消除所有 `Any`，补全缺失的类型标注 |
| **错误处理审计** | 分散 try/except | **统一错误类型 + 结构化日志**。每个外部调用（LLM/DB/Shell/HTTP）有明确的超时、重试、熔断策略 |

### Phase 2 · 记忆系统极致化（Q3 2026）

| 项目 | 现状 | 目标 |
|:-----|:------|:------|
| **长期记忆自动抽取** | 代码存在但未接入 | **每 N 轮对话自动提取事实**，按置信度打分，存入 SQLite |
| **长期记忆自动召回** | 代码存在但未接入 | **每次对话开头注入 top-K 相关事实**，基于语义相似度排序 |
| **跨 Agent 工作记忆** | 用 description 拼接传上游产出 | **专用 shared_working 表**，`(session_id, key) → value`，支持增量更新 |
| **上下文压缩** | 基础摘要 | **语义级压缩**：保留用户指令、关键事实、未完成任务，丢弃冗余闲聊。提供压缩预览和回退 |
| **记忆隔离验证** | 通过测试 | **压力测试**：多 Agent 并发读写、session 切换、大型上下文压缩的稳定性验证 |

### Phase 3 · Agent 协作极致化（Q4 2026）

| 项目 | 现状 | 目标 |
|:-----|:------|:------|
| **Pipeline 模板** | 4 个内置 | **15+ 模板**，覆盖开发、写作、分析、运维常见场景。用户可自定义模板 |
| **Pipeline 生成** | 固定模板匹配 | **动态 Pipeline 生成**：Snow 根据目标实时生成最优 DAG，缓存高频模式 |
| **并行执行** | 串行按依赖执行 | **真正的并行执行**：无依赖任务同时运行，资源池控制并发数 |
| **委派机制** | delegate_to 工具 | **结构化委派协议**：调用方传递上下文 + 期望产出，被调用方返回结构化结果 |
| **重试策略** | 简单重试（最多 3 轮） | **自适应重退避**：区分可重试（超时/限流）和不可重试（参数错误）失败；退避时间按失败次数指数增长 |
| **进度可见性** | Dashboard 显示任务状态 | **精确进度估算**：基于历史执行时间的 ETA，每任务完成百分比实时更新 |

### Phase 4 · 终端体验极致化（Q4 2026）

| 项目 | 现状 | 目标 |
|:-----|:------|:------|
| **CLI 架构** | main.py 3434 行 | **拆分**：`cli/interactive.py`、`cli/display.py`、`cli/commands.py`、`cli/voice.py` |
| **流式输出** | Markdown 块级更新 | **字符级流式**：平滑打字机效果，支持代码块语法高亮逐行出现 |
| **终端适配** | 基础 resize 处理 | **完美 resize**：所有 Live 显示在终端尺寸变化时自动重排，无内容截断 |
| **图片渲染** | 不支持 | **终端内渲染图表/架构图**：使用 ASCII/Unicode 块级近似或 Kitty 协议 |
| **多语言路由** | 以中文为中心的 keywords | **英文/日文等完整支持**：路由关键词、Pipeline 触发词、help 文本国际化 |
| **启动速度** | ~0.4s（已优化） | **< 200ms**：极致懒加载 + 预编译 regex + 缓存 LLM catalog |

### Phase 5 · 工具与安全极致化（2027 Q1）

| 项目 | 现状 | 目标 |
|:-----|:------|:------|
| **工具路由** | 基于关键词的 top-K 选择 | **语义选择**：Embedding 相似度 + 使用频率统计，为每个 query 动态选择最优工具子集 |
| **工具结果缓存** | 无 | **透明缓存**：相同参数的工具调用命中缓存直接返回（LLM Cache 延伸至工具层） |
| **工具调用链追踪** | 无 | **完整调用链**：记录每次工具调用的 parent_id、耗时、token 消耗，支持火焰图展示 |
| **Shell 执行** | 基础安全模式 | **沙箱执行**：可选容器隔离，输出流式返回，超时强制终止 |
| **Fuzzing** | 无 | **工具输入 Fuzzing**：随机生成边界输入测试所有工具的稳定性 |

### Phase 6 · 可观测与运维极致化（2027 Q1）

| 项目 | 现状 | 目标 |
|:-----|:------|:------|
| **日志系统** | 基础结构化日志 | **结构化追踪**：全链路 request_id，每个 Agent 调用可追溯完整的 LLM 请求/响应链 |
| **指标暴露** | 无 | **Prometheus 指标**：LLM 调用延迟/成功率、工具执行分布、Agent 切换频率 |
| **调试模式** | 无 | **交互式调试**：调试模式下每一步暂停，可检查 Agent 的当前记忆、工具状态、LLM 原始响应 |
| **确定性重放** | 无 | **种子模式**：给定种子，LLM 调用 Mock 返回预设结果，用于回归测试和 bug 复现 |

## 八、开发

```bash
# 安装 dev 依赖
pip install -e ".[dev]"

# 测试
pytest tests/ -v --cov=src/

# Lint + 类型
ruff check src/ tests/
mypy src/

# 格式化
ruff format src/ tests/
```

## 九、技术栈

| 层面 | 技术 |
|:-----|:------|
| 运行时 | Python 3.11+ · asyncio |
| LLM | LiteLLM（OpenAI / Anthropic / DeepSeek / Ollama） |
| CLI | Typer · Rich（Live / Markdown / Tables / Layout） |
| 记忆 | aiosqlite · 三层架构（Short + Working + Long-term） |
| 搜索 | DuckDuckGo（内置，免 API Key） |
| 语音 | Doubao TTS V3 HTTP/WebSocket |
| 工具 | 15 内置 · MCP 协议 · Plugin 系统 |
| CI | GitHub Actions · Ruff · MyPy · Pytest × 3 Python |

## License

[MIT](LICENSE)

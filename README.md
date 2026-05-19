<div align="center">

# Weather Agents

**雾 · 雨 · 霜 · 雪 · 露 · 晴**

*六位 Agent，一支团队。专精领域，默契协作。*

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/susurrune/weather-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/susurrune/weather-agents/actions)
[![Tests](https://img.shields.io/badge/tests-776_🌡️-8A2BE2)](https://github.com/susurrune/weather-agents)
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

| Agent | 职责 | 核心技能 |
|:------|:------|:---------|
| **Fog** 雾 | 探索研究 | `web_research` · `code_analysis` · `document_analysis` |
| **Rain** 雨 | 生成创造 | `code_generator` · `content_writer` · `data_transformer` |
| **Frost** 霜 | 审查优化 | `code_reviewer` · `security_auditor` · `performance_checker` |
| **Snow** 雪 | 规划编排 | `task_planner` · `arch_designer` · `workflow_designer` |
| **Dew** 露 | 运维集成 | `sys_operator` · `ci_cd_manager` · `api_integrator` |
| **Fair** 晴 | 情感陪伴 | `emotional_companion` · `self_evolve` |

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

### 配置

首次运行自动进入设置向导，选择统一模型或每个 Agent 独立配置。也可手动设置：

```bash
wa init                                    # 重新跑向导
wa config set api_key.deepseek sk-xxx      # 直接写入 API Key
wa config set cli.default_agent rain       # 设置默认启动 Agent（默认 fog）
export DEEPSEEK_API_KEY=sk-xxx             # 或环境变量
```

支持多模型提供商：

```bash
# 每个 Agent 独立配置模型
wa config set model.fog deepseek/deepseek-v4-flash
wa config set model.rain claude-sonnet-4-6

# 统一所有 Agent 模型
/model deepseek/deepseek-v4-flash    # chat 内命令
```

### 语音聊天配置

```bash
# 方式一（最快）：环境变量，无需写配置文件
set DOUBAO_TTS_API_KEY=你的火山引擎APIKey

# 方式二：一条命令即可
wa config set tts.api_key 你的火山引擎APIKey

# 启动语音服务器
wa voice
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

### 内置工具

| 分类 | 工具 |
|:-----|:-----|
| 文件 | `read_file` · `write_file` · `edit_file` · `move_file` · `copy_file` · `delete_file` |
| 目录 | `list_directory` · `tree` · `file_search` · `code_search` |
| Shell | `shell_exec`（安全模式） |
| 网络 | `http_get` · `http_post` · `web_search`（DuckDuckGo，免 API Key） |
| Git | `git_status` · `git_diff` · `git_log` · `git_add` · `git_commit` · `git_checkout` |
| 委派 | `delegate_to`（Agent 间任务委派） |
| 任务 | `task_done`（Agent 自主判定任务完成） |

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
- **语音聊天** — 晴（Fair）专属，Doubao TTS V3 HTTP + WebSocket 语音服务器，支持环境变量一键配置
- **会话管理** — 创建/加载/删除会话，`wa chat --new` 开新会话
- **费用追踪** — 各 Agent 累计 Token 和费用，支持预算上限
- **智能工作空间** — 自动检测最佳磁盘位置，多盘跳过 C 盘
- **安全默认值** — Shell 执行拒危险命令，HTTP 请求禁私网，长输出截断标记
- **上下文压缩** — 接近限制时智能摘要，保留关键指令
- **技能自演进** — Fair（晴）可在运行时自主创建新技能

## 五、CLI 参考

### 子命令

```bash
wa init              # 交互式配置向导
wa chat [agent] [msg] # 对话（默认由 cli.default_agent 指定，默认 fog）
wa task <goal>       # 多 Agent 协作
wa status            # 所有 Agent 状态
wa config <action>   # 配置管理
wa memory <action>   # 记忆管理
wa voice [options]   # 语音服务器（需配置 TTS）
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
| `/sessions` · `/session new\|load\|delete` | 会话管理 |
| `/compact` | 主动压缩上下文 |
| `/help` · `/clear` · `/quit` | 通用 |

## 六、项目状态

> 以下数据基于代码事实：

- **代码**：56 个源文件，~17k 行 Python
- **测试**：776 项，覆盖率 > 62%
- **提交**：193 个 commits
- **CI**：GitHub Actions — Ruff + MyPy + Pytest × 3 个 Python 版本

### 已落地能力

| 能力 | 状态 |
|:-----|:------|
| 6 Agent 独立角色 + 独立 LLM 配置 | ✅ 完成 |
| 三级复杂度路由（direct/single/orchestrate） | ✅ 完成，< 1ms |
| Pipeline 模板（4 个内置 DAG） | ✅ 完成，命中跳过编排 LLM 调用 |
| DAG 任务编排 + 依赖数据注入 | ✅ 完成，自动传上游产出 |
| 三层记忆（short/working/long-term） | ✅ 完成 |
| 内置工具 + MCP + Plugin | ✅ 完成 |
| 多 Provider LLM（OpenAI/Anthropic/DeepSeek/Ollama） | ✅ 完成 |
| 会话管理 + Session Resume | ✅ 完成 |
| 语音聊天（Doubao TTS，支持环境变量一键配置） | ✅ 完成 |
| 费用追踪 + 预算控制 | ✅ 完成 |
| 安全守卫（Shell/HTTP/截断） | ✅ 完成 |
| 上下文压缩 + 智能摘要 | ✅ 完成 |
| Agent 间委派（delegate_to） | ✅ 完成 |
| Agent 自主判定任务完成（task_done） | ✅ 完成 |
| 卡循环检测 + 自动恢复 | ✅ 完成 |
| 交互式命令弹出 + 自动补全 | ✅ 完成 |
| 技能自演进（Fair 自主创建新技能） | ✅ 完成 |
| Claude Code 自动化骨架 | ✅ 完成 |

## 七、开发

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

## 八、技术栈

| 层面 | 技术 |
|:-----|:------|
| 运行时 | Python 3.11+ · asyncio |
| LLM | LiteLLM（OpenAI / Anthropic / DeepSeek / Ollama） |
| CLI | Typer · Rich（Live / Markdown / Tables / Layout） |
| 记忆 | aiosqlite · 三层架构（Short + Working + Long-term） |
| 搜索 | DuckDuckGo（内置，免 API Key） |
| 语音 | Doubao TTS V3 HTTP/WebSocket |
| 工具 | 内置 · MCP 协议 · Plugin 系统 |
| CI | GitHub Actions · Ruff · MyPy · Pytest × 3 Python |

## License

[MIT](LICENSE)

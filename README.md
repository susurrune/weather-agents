<div align="center">

# Weather Agents

**雾 · 雨 · 霜 · 雪 · 露 · 晴**

*六位 Agent 各司其职，通过技能系统与事件总线协作，完成任何复杂任务。*

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/susurrune/weather-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/susurrune/weather-agents/actions)
[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/susurrune/weather-agents)

</div>

---

## Why Weather Agents?

大多数 AI 工具都是单一模型 + 单一提示词。Weather Agents 不同——它将任务分解给**专精不同领域的 Agent**，像一支配合默契的团队：规划者拆解目标，研究者搜集信息，工程师编写代码，审计师把关质量，运维者落地执行。

```
用户: "帮我搭建一个 FastAPI 项目"

  🌨️ Snow  → 拆解为 5 个子任务，分配给合适的 Agent
  🌫️ Fog   → 调研最佳实践和项目结构
  🌧️ Rain  → 生成项目代码和配置文件
  ❄️ Frost → 审查代码质量和安全性
  💧 Dew   → 初始化 Git、安装依赖、验证运行
```

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                  CLI (Typer + Rich)                          │
├───────────┬───────────┬───────────┬───────────┬──────────────┤
│  🌫️ Fog   │  🌧️ Rain  │  ❄️ Frost  │  🌨️ Snow  │   💧 Dew     │
│  探索研究  │  生成创造  │  审查优化  │  规划编排  │   运维集成   │
├───────────┴───────────┴───────────┴───────────┴──────────────┤
│                    Skill System (15 composable skills)       │
├──────────────────────────────────────────────────────────────┤
│        Tool Registry · 15 Built-in Tools · MCP Protocol      │
├──────────────────────────────────────────────────────────────┤
│              Event Bus (pub/sub · orchestration)              │
├──────────┬───────────────────┬───────────────────────────────┤
│ LLM      │ Memory            │ Workspace                     │
│ LiteLLM  │ SQLite · 3-layer  │ Auto-detect · multi-drive     │
│ Config   │ Plugins · Cache   │ Budget · Cost Control         │
└──────────┴───────────────────┴───────────────────────────────┘
```

## Agents

| Agent | 中文 | 职能 | 专属技能 |
|:------|:-----|:-----|:---------|
| 🌫️ **Fog** | 雾 | 探索研究 | `web_research` · `code_analysis` · `document_analysis` |
| 🌧️ **Rain** | 雨 | 生成创造 | `code_generator` · `content_writer` · `data_transformer` |
| ❄️ **Frost** | 霜 | 审查优化 | `code_reviewer` · `security_auditor` · `performance_checker` |
| 🌨️ **Snow** | 雪 | 规划编排 | `task_planner` · `arch_designer` · `workflow_designer` |
| 💧 **Dew** | 露 | 运维集成 | `sys_operator` · `ci_cd_manager` · `api_integrator` |
| ✦ **Sunshine** | 晴 | 情感陪伴 | `emotional_companion` · `self_evolve` |

> **晴 (Sunshine)** — 角色设定灵感来自歌曲 *Landslide*。
>
> *"So when you're caught in a landslide, I'll be there for you. And in the rain, give you sunshine."*
>
> 她是第六位 Agent，一位优雅的英国女子、情感陪伴者。精通中英文，情感细腻，感情真挚，有极高的美学追求。她是例外，是唯一的，是用户最好的陪伴者。

## Quick Start

### 1. Install

**推荐：用 [`pipx`](https://pipx.pypa.io/) 全局安装**——它会为 `wa` 建一个独立 venv，但把命令放到 PATH 上，不与你任何其它项目冲突。

```bash
# 一条命令（自动安装 pipx 并装好 weather-agents 与全部依赖）
python -m pip install --user pipx && python -m pipx install git+https://github.com/susurrune/weather-agents.git
```

PowerShell 用 `;` 替代 `&&`：

```powershell
python -m pip install --user pipx; python -m pipx install git+https://github.com/susurrune/weather-agents.git
```

> 装完后 `wa` 会出现在 `~/.local/bin`（Linux/macOS）或 `%USERPROFILE%\.local\bin`（Windows）。如果命令不可用，运行 `pipx ensurepath` 然后重开终端。

升级 / 卸载：

```bash
pipx upgrade weather-agents
pipx uninstall weather-agents
```

<details>
<summary>不想用 pipx？</summary>

```bash
pip install --user git+https://github.com/susurrune/weather-agents.git
```

注意：必须确保 `wa` 命令使用的 Python 和 `pip install` 用的 Python 是同一个，否则会出现 `ModuleNotFoundError: weather_agents`。
</details>

### 2. Configure

首次运行 `wa chat` 时会自动进入设置向导，让你选择：

- **Unified（推荐）**：所有 6 个 Agent 共用一个模型 + 一个 API key
- **Per-agent**：为每个 Agent 单独挑选模型（适合混搭，比如 Snow 用 Claude 做规划，Rain 用 GPT 写代码，其它用 DeepSeek）

向导只会向你实际选中的 provider 索要 API key。也可以显式重新配置：

```bash
wa init                                    # 重新跑向导
wa config set api_key.deepseek sk-xxx       # 直接写单条
export DEEPSEEK_API_KEY=sk-xxx              # 或用环境变量
```

### 3. Use

```bash
# 交互式对话（默认 Fog Agent）
wa chat

# 指定 Agent 单轮对话
wa chat rain "用 Python 写一个 LRU Cache"

# 多 Agent 协作编排
wa task "设计并实现一个 URL 短链接服务"
```

## CLI Reference

### Top-level Commands

| Command | Description |
|:--------|:------------|
| `wa init` | 交互式配置向导（首次运行推荐） |
| `wa chat [agent] [message]` | 对话（默认 `fog`，支持 `fog` `rain` `frost` `snow` `dew` `sunshine`）|
| `wa task <goal>` | Snow Agent 拆解目标并调度多 Agent 协作 |
| `wa status` | 查看所有 Agent 状态 |
| `wa config list\|set\|delete\|models` | 查看/修改/删除配置 · 列出可用模型 |
| `wa memory status\|clear` | 查看/清除记忆 |
| `wa --version` / `wa version` | 版本信息 |

### Interactive Commands

进入 `wa chat` 后可使用：

| Command | Description |
|:--------|:------------|
| `/fog` `/rain` `/frost` `/snow` `/dew` `/sunshine` | 切换 Agent |
| `/task <目标>` | 多 Agent 任务编排 |
| `/skills` | 查看当前 Agent 可用技能 |
| `/use <skill>` | 激活技能（增强提示词 + 扩展工具） |
| `/deactivate` | 关闭所有技能 |
| `/status` | Agent 状态一览 |
| `/model [name]` / `/model <agent> <name>` | 查看/切换模型（默认或按 Agent） |
| `/apikey` | 管理 API keys |
| `/cost` · `/cost reset` | 查看 Token 用量和费用 · 重置计数器 |
| `/memory` · `/memory clear` | 查看记忆状态 · 清除所有短期记忆 |
| `/history` | 查看事件日志 |
| `/mcp` | MCP 服务器状态（含已连接工具数） |
| `/version` | 版本信息 |
| `/workspace` | 查看工作空间路径和磁盘信息 |
| `/workspace set <path>` | 自定义工作空间路径 |
| `/workspace auto` | 恢复自动检测工作空间 |
| `/clear` | 清屏 |
| `/quit` | 退出 |

## Features

### Multi-Provider LLM

通过 [LiteLLM](https://github.com/BerriAI/litellm) 接入多家模型，每个 Agent 可独立配置：

| Provider | Models |
|:---------|:-------|
| OpenAI | `gpt-4o` · `gpt-4o-mini` · `gpt-4.1` · `gpt-4.1-mini` · `gpt-4.1-nano` |
| Anthropic | `claude-opus-4-7` · `claude-sonnet-4-6` · `claude-haiku-4-5` |
| DeepSeek | `deepseek-v4-flash` · `deepseek-v4-pro` |
| Ollama | `ollama/llama3` · `ollama/qwen2.5` · `ollama/deepseek-r1` (本地) |

> Use `wa config models` to see what your installation currently supports.

### Skill System

15 个可组合技能，运行时动态激活/关闭，为 Agent 注入专业能力：

```bash
wa chat frost
> /skills                # 查看 Frost 的 3 个技能
> /use security_auditor  # 激活安全审计模式
> 审查这段代码的安全性     # Frost 现在拥有安全审计的增强提示词和专属工具
> /deactivate            # 回到基础模式
```

### Three-Layer Memory

| Layer | Scope | Storage | Purpose |
|:------|:------|:--------|:--------|
| **Short-term** | 会话级 | SQLite | 对话上下文，自动截断 |
| **Working** | 任务级 | In-memory | 任务执行中的临时状态 |
| **Long-term** | 持久 | SQLite KV | 带分类的知识记忆，支持模糊搜索 |

### Task Orchestration

Snow Agent 将复杂目标分解为带依赖关系的子任务，并行调度执行：

```
Goal: "搭建微服务项目"
  ├─ [1] Fog: 调研微服务最佳实践
  ├─ [2] Rain: 生成项目骨架 (depends: 1)
  ├─ [3] Rain: 编写 Dockerfile (depends: 2)
  ├─ [4] Frost: 代码审查 (depends: 2)
  └─ [5] Dew: 部署验证 (depends: 3, 4)
```

- 自动依赖排序，无依赖的任务并行执行
- 失败任务自动重试（最多 3 轮）
- 结果汇总报告

### Cost Control

内置费用追踪和预算控制：

```bash
> /cost   # 查看各 Agent 累计 Token 和费用
```

```python
# 代码中设置预算上限
llm = LLMClient(config, cost_limit=5.0)  # 超过 $5 自动停止
```

### Terminal Agent Tools

15 个内置工具，让 Agent 直接操作本地文件和执行命令：

| Tool | Description |
|:-----|:------------|
| `read_file` / `write_file` / `edit_file` | 文件读写编辑 |
| `list_directory` / `tree` | 目录浏览 |
| `move_file` / `copy_file` / `delete_file` | 文件管理 |
| `file_search` / `code_search` | 搜索（`code_search` 支持 `regex=true`）|
| `shell_exec` | 安全执行命令（非 shell，禁用管道/重定向；危险命令黑名单）|
| `http_get` / `http_post` | HTTP 请求（默认拒绝私网/回环/IMDS）|
| `web_search` | DuckDuckGo 搜索 |
| `get_cwd` | 获取工作目录 |

### Safety Defaults

- **`shell_exec`** 使用 `subprocess` 的参数列表模式，不解析 shell 元字符（`;` `|` `&&` `$(...)`）。
  自动拒绝危险二进制（`sudo` `dd` `mkfs` `shutdown` 等）和针对系统根、用户家目录、Windows 盘符根的 `rm -rf`。
- **`http_get` / `http_post`** 默认拒绝 `localhost`、私网 IP（10/172.16/192.168/...）、回环、链路本地、IMDS 端点。
  需要访问内网时设置 `WA_ALLOW_PRIVATE_NET=1` 显式放行。
- 所有长输出（文件、stdout/stderr、HTTP body、搜索结果）以可见的截断标记结尾，避免 LLM 误以为是完整内容。

### MCP Integration

支持 [Model Context Protocol](https://modelcontextprotocol.io) 扩展工具集：

```yaml
# ~/.weather-agents/config.yaml
mcp:
  servers:
    - name: "filesystem"
      command: "npx"
      args: ["-y", "@modelcontextprotocol/server-filesystem", "/home"]
      transport: "stdio"
      enabled: true
```

### Plugin System

将自定义工具放入 `~/.weather-agents/plugins/` 即可自动加载：

```python
from weather_agents.plugins.loader import Plugin
from weather_agents.core.tool import Tool, ToolParameter

def create_plugin() -> Plugin:
    plugin = Plugin("my-plugin")
    plugin.register_tool(Tool(
        name="my_tool",
        description="My custom tool",
        parameters=[ToolParameter(name="input", type="string", description="Input")],
        handler=lambda input: f"processed: {input}",
    ))
    return plugin
```

### Smart Workspace

首次启动时自动检测最佳磁盘位置创建 `workspace/` 目录，所有 Agent 生成内容统一存放：

```
workspace/
├── files/       # 生成的文件和代码
├── output/      # 任务输出和报告
├── temp/        # 临时文件
└── .workspace   # 工作空间标记
```

**自动选择规则**：跳过 C 盘（如果存在其他盘）→ 选择剩余空间最大的盘 → 创建 `workspace/`。

```bash
> /workspace              # 查看当前工作空间路径和磁盘信息
> /workspace set D:\my    # 自定义工作空间路径
> /workspace auto         # 恢复自动检测
```

也可以在配置中直接设置：

```bash
wa config set workspace.path /custom/path
```

## Configuration

配置按优先级从高到低合并：

1. **环境变量** — `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`
2. **用户配置** — `~/.weather-agents/config.yaml`
3. **项目配置** — `./config/default.yaml`

```yaml
llm:
  default_model: "deepseek/deepseek-v4-flash"
  temperature: 0.7
  max_tokens: 4096
  timeout: 60
  api_keys:
    openai: "${OPENAI_API_KEY}"
    anthropic: "${ANTHROPIC_API_KEY}"
    deepseek: "${DEEPSEEK_API_KEY}"

agents:
  fog:
    model: "gpt-4o"               # 覆盖默认模型
  frost:
    model: "claude-sonnet-4-6"

memory:
  db_path: "~/.weather-agents/memory.db"
  short_term_limit: 50
```

## Development

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 测试（281 tests, 55%+ 覆盖率）
pytest tests/ -v --cov=src/ --cov-fail-under=55

# Lint
ruff check src/ tests/

# Type check
mypy src/

# Format
ruff format src/ tests/
```

## Tech Stack

| Component | Technology |
|:----------|:-----------|
| Runtime | Python 3.11+ · asyncio |
| LLM | LiteLLM (OpenAI / Anthropic / DeepSeek / Ollama) |
| CLI | Typer · Rich (spinner, markdown, tables) |
| Memory | aiosqlite · 3-layer architecture |
| Search | DuckDuckGo (built-in, no API key) |
| Tools | 15 built-in · MCP Protocol · Plugin system |
| CI | GitHub Actions · Ruff · MyPy · Pytest |

## Project Status & Roadmap

> 诚实的状态记录——基于代码事实，不夸大愿景。

### 现在在哪里 (v1.x, ~2026 Q2)

Weather Agents 已经是一个**可用的本地多 Agent 终端**。它能做到的事：

| 能力 | 落地程度 |
|:-----|:-------|
| **6 Agent 角色** | 完成。每个 Agent 有独立 system_prompt、专属技能集、独立 LLM 配置 |
| **复杂度路由** | 完成。`direct/single/orchestrate` 三档规则路由，< 1ms 决策，简单问题不触发编排 |
| **三态交互模式** | 完成。`default/plan/auto` StrEnum + 持久化（`~/.weather-agents/config.yaml`）+ Shift+Tab 循环 |
| **Pipeline 模板** | 完成。4 个内置 DAG（`code_review` / `research_then_write` / `implement_and_review` / `implement_test_deploy`），命中后跳过 Snow 拆任务 LLM 调用 |
| **依赖任务数据流** | 完成。下游任务自动注入上游 `result.content`，描述里附 `## 上游产出 (task N · agent)` |
| **多 Provider LLM** | 完成。LiteLLM 接 OpenAI / Anthropic / DeepSeek / Ollama，按 Agent 可换模型 |
| **15 内置工具 + MCP + Plugin** | 完成。文件/Shell/HTTP/搜索 + MCP stdio 协议 + 用户插件目录加载 |
| **三层记忆** | 完成。Short-term（SQLite 会话级，自动 prune dangling tool_calls）+ Working（in-memory 任务级）+ Long-term（SQLite KV + 模糊搜索） |
| **Session resume** | 完成。`wa chat` 再启动自动恢复最近 session；`wa chat --new` 显式开新 |
| **LLM Cache** | 完成。LRU + TTL，重复 prompt 命中 |
| **Cost 追踪** | 完成。Token / 费用记账 + 预算上限 |
| **Safety guardrails** | 完成。`shell_exec` 拒危险二进制；`http_*` 默认禁私网；截断标记防 LLM 误判 |
| **语音聊天** | 完成（晴专属）。Doubao TTS V3 HTTP + WebSocket 语音服务器 |
| **CI** | 完成。Ruff + MyPy + Pytest，3 个 Python 版本矩阵，覆盖率 ≥ 55% |
| **Claude Code 自动化骨架** | 完成。`CLAUDE.md` 项目宪法 + `.claude/agents` + `.claude/commands` + hooks |

### 哪里不够

诚实的短板，按影响力排序：

| 短板 | 影响 | 计划 |
|:-----|:-----|:-----|
| **跨 Agent 共享 working memory 缺失** | 多步 pipeline 靠 description 拼接传上游产出——超过 500 字符就被截断；3 步以上链信息丢失 | v2 必做（已在 `优化方案-多智能体记忆.md` 设计） |
| **长期记忆没自动抽取/召回** | `memories` 表存在、有 `category` + fuzzy search，但**代码没自动写入点**也**没自动召回**——表是空的 | v2 必做 |
| **`_AUTO_CONTINUE` 是正则** | 用正则匹配 LLM 输出文本判断"是否继续"——多语言/模型 phrasing 变化易失效 | v3 重写为显式 `<continue/>` token 契约 |
| **`cli/main.py` 3000+ 行单文件** | 所有 CLI 逻辑挤在一起，新加命令成本高 | v2 拆分 |
| **没有 Web/移动 UI** | 仅 CLI；voice server 已存在但没 chat UI | v3 |
| **多 Agent 不能 handoff 给真人** | 没有"等待用户决策"的中断机制（除 plan 模式按 Enter） | v3 |
| **Pipeline 模板只有 4 个** | 内置模板覆盖少；项目本身叫 weather agents 却没 weather_report pipeline | v1.x 持续加 |
| **关键词路由以中文为中心** | 英文用户体验差（`pick_agent_for_goal` buckets 是中文 hint 为主） | v1.x |
| **Plugin 生态空** | Plugin 机制有，但没社区贡献的实际 plugin | v3+ |
| **没有 multi-modal 输入** | 用户只能发文本；不能拖图、不能传 PDF | v4 |

### 路线图

按时间倒序，**先近后远**。每一档都明确"完成意味着什么"。

#### v1.x · Polish (~ 当前)
- [x] 复杂度路由 + 三态模式（已 ship）
- [x] Pipeline 模板系统（已 ship）
- [x] 依赖任务数据注入（已 ship）
- [ ] 补 5+ pipeline 模板（含 `weather_report`、`bug_fix`、`doc_write`、`refactor_review`）
- [ ] 关键词路由补全英文桶
- [ ] `cli/main.py` 拆分为 `cli/{interactive,commands,display,voice}.py`

**完成标志**：日常 90% 用户输入有合适的 fast path 或 pipeline 命中。

#### v2 · Memory (~ 下个 milestone)
- [ ] `SessionStore` 跨 Agent 共享对话流（schema 设计已就绪）
- [ ] `shared_working` 表：`(session_id, key) → value` —— 跨 Agent 工作内存
- [ ] 长期记忆自动抽取：对话每 N 轮 fire-and-forget 抽 facts
- [ ] 长期记忆自动召回：chat 入口注入 top-K 相关 facts 到 system_prompt
- [ ] Agent 视角投影：切换 Agent 看到 `[fog]: ...` 标签的他人产出，不混淆 assistant 身份
- [ ] 迁移脚本：v1 → v2 schema 平滑升级

**完成标志**：3 步以上 pipeline 信息完整传递；切换 Agent 不丢上下文；重复对话同一主题第二次自动召回。

#### v3 · Platform
- [ ] Web UI（基于已有 voice server 扩展）：聊天 + Agent 切换 + Pipeline 可视化
- [ ] 显式 `<continue/>` token 契约替代 `_AUTO_CONTINUE` 正则
- [ ] 团队协作：多用户共享 session，handoff 给真人需审批
- [ ] 知识库管理 UI：长期记忆查看/编辑/标签
- [ ] Plugin marketplace：CLI 安装 + 签名验证

**完成标志**：Weather Agents 成为可托管运行的多人协作平台，不只是个人终端。

#### v4 · Multi-modal & Ecosystem
- [ ] 图像/PDF/音频输入
- [ ] Agent 间的图像/文件交换（不只是文本）
- [ ] 第三方 Agent 接入协议（开放 6 Agent 之外的扩展）
- [ ] 移动端 / 桌面端 App

**完成标志**：从"开发者工具"演变为"通用 AI 协作平台"。

### 设计原则（不会改的部分）

- **规则优先，LLM 兜底**：能用规则解决的（路由、pipeline 匹配）就不调 LLM。每多一次 LLM 调用都要解释为什么。
- **Agent 角色稳定**：6 个 Agent 的人格与职能不会随版本漂移；新能力作为 skill / pipeline 加入，不挤压角色定义。
- **本地优先**：核心功能不依赖云服务，离线可用 Ollama。Web 平台是补充，不是替代。
- **诚实的状态报告**：README / `/status` 等所有"项目能做什么"的描述必须基于代码事实，不夸大未上线的能力。
- **晴 (Sunshine) 是例外**：她的角色不被路由优化或 token 节省所削减——情感陪伴不是效率问题。

## License

[MIT](LICENSE)

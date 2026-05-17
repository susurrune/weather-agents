# Weather Agents — 功能与测试文档

> 版本: 1.0.0 · 测试日期: 2026-05-16 · 测试总数: 374 passed

---

## 目录

1. [项目概述](#1-项目概述)
2. [系统架构](#2-系统架构)
3. [核心模块](#3-核心模块)
4. [六 Agent 系统](#4-六-agent-系统)
5. [CLI 命令参考](#5-cli-命令参考)
6. [配置系统](#6-配置系统)
7. [工具系统](#7-工具系统)
8. [技能系统](#8-技能系统)
9. [测试覆盖分析](#9-测试覆盖分析)
10. [已发现并修复的 Bug](#10-已发现并修复的-bug)

---

## 1. 项目概述

Weather Agents 是一个天气主题的多智能体 AI 编排框架，包含 **6 个专业 Agent**：

| Agent | 代号 | 专长 |
|-------|------|------|
| 雾 Fog | `~` | 探索研究 |
| 雨 Rain | `/` | 生成创造 |
| 霜 Frost | `+` | 审查优化 |
| 雪 Snow | `·` | 规划编排 |
| 露 Dew | `,` | 运维集成 |
| 晴 Fair | `*` | 情感陪伴 |

**技术栈**: Python 3.11+, LiteLLM, Rich (CLI), Typer, aiosqlite, httpx, pyyaml

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────┐
│                    CLI (main.py)                      │
│    Typer app · Rich Live · 交互式输入 · 流式输出      │
├─────────────────────────────────────────────────────┤
│              System Context (factory.py)              │
│    全局注册表 · 消息总线 · LLM 客户端 · MCP 管理器    │
├────────────┬───────────┬───────────┬────────────────┤
│  Agent 层   │  Tool 层   │  Skill 层 │  Plugin 层     │
│ fog/rain/  │ read_file │ code_     │ 动态加载 .py   │
│ frost/     │ write_    │ reviewer  │ 文件注入工具    │
│ snow/dew/  │ shell_    │ web_      │                │
│ fair       │ http_     │ research  │                │
├────────────┴───────────┴───────────┴────────────────┤
│                  Core 基础设施                        │
│  Config · Bus · Memory · Cache · MCP · Workspace    │
└─────────────────────────────────────────────────────┘
```

### 核心数据流

```
用户输入 → CLI → Agent.chat_stream() → LLM (LiteLLM)
                                        ↓
                                   工具调用 (并行 asyncio.gather)
                                        ↓
                                   结果返回 → 流式输出 → Rich Live
```

---

## 3. 核心模块

### 3.1 config.py — 配置管理

- **AppConfig**: 分层配置数据类（LLM、Agent、Bus、Memory、Web、Workspace、Plugin、MCP）
- **load_config()**: 默认配置 → 用户覆盖 → 环境变量，2 秒 TTL 缓存
- **set_config() / delete_config()**: 运行时配置修改并持久化到 `~/.weather-agents/config.yaml`
- **模型目录**: `models.yaml` 提供可用模型列表，含 context_window、定价、fallback 链
- **AGENT_NAMES**: 单一真相来源 `("fog", "rain", "frost", "snow", "dew", "fair")`

### 3.2 agent.py — Agent 基类

| 方法 | 功能 |
|------|------|
| `init()` | 初始化记忆、注入系统提示词、注册工具 |
| `chat()` | 通用对话（非流式），含自动工具循环 |
| `chat_stream()` | 流式对话，支持工具调用并行执行 |
| `compact()` | LLM 驱动的上下文压缩 |
| `execute_task()` | 执行编排任务 |
| `activate_skill()` | 激活技能并注入系统提示词 |

**系统提示词注入链**:
1. `_resolve_system_prompt()` — Agent 特定提示词
2. `_inject_workspace_info()` — 工作空间路径
3. `_inject_behavior_rules()` — 行为守则（不赘述、不装饰线等）
4. `_inject_programming_wisdom()` — 工程能力指导
5. `_current_time_tag()` — 当前时间标记（30 秒类级缓存）

### 3.3 llm.py — LLM 抽象层

- 基于 LiteLLM，支持 14+ provider
- **Fallback 链**: 主模型失败后自动降级（如 `deepseek-v4-flash` → `gpt-4.1-mini`）
- **重试策略**: 仅重试可恢复错误（408/425/429/5xx/超时/连接错误）
- **缓存**: 相同请求在 120 秒 TTL 内直接返回缓存结果
- **成本追踪**: 按 Agent 统计 token 用量和估算费用
- **预算控制**: 可选成本上限

### 3.4 bus.py — 消息总线

事件驱动架构，支持：
- 点对点发送 (target 指定)
- 广播 (target=None)
- 状态变更监听 (state_change)
- 事件历史记录 (最多 2000 条)

### 3.5 memory.py — 三层记忆系统

| 层级 | 存储 | 功能 |
|------|------|------|
| Short-term | SQLite + 内存 | 对话上下文，自动裁剪 |
| Working | SQLite + 内存 | 任务级临时状态 |
| Long-term | SQLite | 持久化键值存储，按分类检索 |

- **会话管理**: 创建/切换/删除会话，会话内消息隔离
- **自动裁剪**: 超出 `max_persisted_messages`（默认 1000）时自动删除旧消息
- **孤儿工具消息清理**: 自动移除不完整的 tool_call/tool 消息对

### 3.6 workspace.py — 工作空间管理

- 自动检测最佳磁盘（Windows: 跳过 C:，选剩余空间最大的盘；Unix: `~/workspace`）
- 创建子目录: `files/`, `output/`, `temp/`
- 支持自定义路径

---

## 4. 六 Agent 系统

### 4.1 Fog (雾) — 探索研究

- **专长**: 信息检索、代码分析、趋势洞察
- **工具**: 全部内置工具 + delegate_to
- **技能**: code_analysis, web_research, data_transformer, self_evolve

### 4.2 Rain (雨) — 生成创造

- **专长**: 代码编写、内容创作、数据转换
- **工具**: 全部内置工具 + delegate_to
- **技能**: code_generator, content_writer, document_analysis, self_evolve

### 4.3 Frost (霜) — 审查优化

- **专长**: 代码审查、安全审计、性能检测
- **工具**: 全部内置工具 + delegate_to
- **技能**: code_reviewer, security_auditor, performance_checker, self_evolve

### 4.4 Snow (雪) — 规划编排

- **专长**: 任务编排、架构设计、流程管理
- **特有方法**: `orchestrate()` — 将目标分解为子任务并分配给其他 Agent
- **技能**: task_planner, arch_designer, workflow_designer, self_evolve

### 4.5 Dew (露) — 运维集成

- **专长**: 命令执行、部署操作、API 集成
- **工具**: 全部内置工具 + delegate_to
- **技能**: sys_operator, ci_cd_manager, api_integrator, self_evolve

### 4.6 Fair (晴) — 情感陪伴

- **专长**: 温暖倾听、情感支持、深度对话、创意灵感
- **提示词风格**: 诗歌化、双语（中英）、美学导向
- **技能**: emotional_companion, self_evolve

---

## 5. CLI 命令参考

### 启动命令

```bash
wa                              # 默认进入 Fog 交互模式
wa --help                       # 显示帮助
wa --version                    # 显示版本
wa init                         # 首次配置向导
wa chat [agent] [message]       # 单次对话
wa task "<goal>"                # 多 Agent 编排任务
wa status                       # 所有 Agent 状态
wa config list                  # 显示配置
wa memory status                # 记忆状态
```

### 交互模式命令 (/ 命令)

| 命令 | 功能 |
|------|------|
| `/help` | 显示帮助 |
| `/clear` | 清屏 |
| `/quit` / `/exit` | 退出 |
| `/fog` / `/rain` / `/frost` / `/snow` / `/dew` / `/fair` | 切换 Agent |
| `/task <goal>` | 多 Agent 编排 |
| `/model` | 查看/设置模型 |
| `/apikey set <prov> <key>` | 添加 API key |
| `/apikey del <prov>` | 删除 API key |
| `/workspace` | 工作空间信息 |
| `/skills` / `/use` / `/deactivate` | 技能管理 |
| `/cost` | 用量与费用 |
| `/compact` | 压缩上下文 |
| `/memory` | 记忆状态 |
| `/sessions` / `/session new/load/delete` | 会话管理 |
| `/mcp` | MCP 服务器状态 |
| `/version` | 版本信息 |

### 交互特性

| 特性 | 说明 |
|------|------|
| `/` 命令补全 | 输入 `/` 显示命令弹窗，↑/↓ 选择，Tab 自动补全 |
| ↑/↓ 历史记录 | 最多 50 条输入历史 |
| Esc 中断 | 流式输出时按 Esc 中断 |
| Shift+Tab | 切换 Auto/Plan 模式 |
| Plan 模式 | 先生成计划，确认后执行 |
| Auto 模式 | 自主推理 + 自动继续（最多 3 轮） |

---

## 6. 配置系统

### 配置文件位置

| 文件 | 说明 |
|------|------|
| `config/default.yaml` | 默认配置（随包安装） |
| `~/.weather-agents/config.yaml` | 用户覆盖配置 |
| `~/.weather-agents/.env` | 环境变量文件 |
| `config/models.yaml` | 模型目录 |

### 支持配置的 Key

| Key | 类型 | 说明 |
|-----|------|------|
| `default_model` | str | 默认模型 |
| `temperature` | float | 温度 [0.0, 2.0] |
| `max_tokens` | int | 最大输出 [1, 200000] |
| `timeout` | int | 超时秒数 [1, 600] |
| `model.<agent>` | str | 特定 Agent 的模型覆盖 |
| `api_key.<provider>` | str | API Key |
| `workspace.path` | str | 工作空间路径 |

### 支持的模型 Provider

OpenAI, Anthropic, DeepSeek, Google/Gemini, Ollama (本地)

---

## 7. 工具系统

### 内置工具

| 工具 | 类型 | 说明 |
|------|------|------|
| `read_file` | 读取 | 读取文件（最多 50KB），支持行范围 |
| `write_file` | 写入 ⚠️ | 写入文件，自动创建目录 |
| `edit_file` | 写入 ⚠️ | 查找替换编辑文件 |
| `move_file` | 写入 ⚠️ | 移动/重命名 |
| `copy_file` | 写入 ⚠️ | 复制文件或目录 |
| `delete_file` | 写入 ⚠️ | 删除文件或空目录 |
| `list_directory` | 读取 | 列出目录内容 |
| `tree` | 读取 | 目录树 |
| `file_search` | 读取 | 按 glob 模式搜索文件 |
| `code_search` | 读取 | 按文本/正则搜索代码 |
| `shell_exec` | 执行 ⚠️ | 安全 shell 执行 |
| `http_get` | 网络 | HTTP GET 请求 |
| `http_post` | 网络 | HTTP POST 请求 |
| `web_search` | 网络 | DuckDuckGo 网络搜索 |
| `get_cwd` | 读取 | 当前工作目录 |
| `delegate_to` | 编排 | 委托任务给其他 Agent |

⚠️ = 危险操作（有保护机制）

### 安全保护

- **写保护路径**: 拒绝写入系统目录（/etc, C:\Windows 等）
- **命令黑名单**: 阻止 dd, mkfs, rm -rf 等危险命令
- **SSRF 防护**: 默认拒绝访问内网 IP/localhost（可通过 `WA_ALLOW_PRIVATE_NET=1` 覆盖）
- **shell 元字符检测**: 阻止 `;`, `&&`, ` `` `, `$()` 等注入

### MCP (Model Context Protocol)

支持 stdio 和 SSE 两种传输方式，兼容 MCP 2025-03-26 规范。

---

## 8. 技能系统

每个 Agent 内置若干技能，通过 `/use <skill>` 激活：

| 技能 | 适用 Agent | 功能 |
|------|-----------|------|
| code_analysis | Fog | 代码分析 |
| web_research | Fog | 网络研究 |
| data_transformer | Fog, Rain | 数据转换 |
| code_generator | Rain | 代码生成 |
| content_writer | Rain | 内容创作 |
| document_analysis | Rain | 文档分析 |
| code_reviewer | Frost | 代码审查 |
| security_auditor | Frost | 安全审计 |
| performance_checker | Frost | 性能检查 |
| task_planner | Snow | 任务规划 |
| arch_designer | Snow | 架构设计 |
| workflow_designer | Snow | 流程设计 |
| sys_operator | Dew | 系统操作 |
| ci_cd_manager | Dew | CI/CD 管理 |
| api_integrator | Dew | API 集成 |
| emotional_companion | Fair | 情感陪伴 |
| self_evolve | 全部 | 自我进化 |

---

## 9. 测试覆盖分析

### 测试套件概况

| 测试文件 | 行数 | 测试内容 |
|----------|------|----------|
| test_agent.py | 658 | Agent 生命周期、方法、状态机 |
| test_cli.py | 983 | CLI 路由、输入解析、格式化 |
| test_integration.py | 304 | 端到端：agent chat、skill、编排 |
| test_memory.py | 323 | 三层记忆、SQLite 持久化、会话管理 |
| test_tool.py | 175 | Tool/ToolRegistry 注册、执行、重试 |
| test_builtin_tools.py | 418 | 所有内置工具函数 |
| test_config.py | 239 | 配置加载、验证、持久化 |
| test_delegate.py | 190 | 委托工具、深度防护 |
| test_factory.py | 275 | 系统上下文、编排 |
| test_llm.py | 159 | LLM 客户端、fallback、成本 |
| test_mcp.py | 270 | MCP 客户端、stdio/SSE 传输 |
| test_bus.py | 114 | 事件总线、发布/订阅 |
| test_workspace.py | 158 | 工作空间检测、初始化 |
| test_plugins.py | 152 | 插件加载 |
| test_snow.py | 64 | Snow 编排、JSON 解析 |
| conftest.py | 67 | 测试夹具和模拟 |

**总计: 16 文件, 4549 行, 374 测试, 全部通过**

### 测试覆盖维度

| 维度 | 覆盖情况 |
|------|----------|
| Agent 初始化 | 全部 6 个 Agent 的 init/chat/close |
| 状态转换 | IDLE → THINKING → ACTING → IDLE |
| 工具调用 | 单个、多个并行、参数解析错误 |
| 流式输出 | content/tool_status/tool_done/done 事件 |
| CLI 命令 | 全部 `/` 命令路由 |
| 配置验证 | temperature/max_tokens/timeout 边界值 |
| 错误处理 | 无效 key、超时、权限、文件不存在 |
| 安全防护 | 写保护路径检测、命令黑名单、SSRF 防护 |
| 记忆持久化 | 写入、读取、裁剪、清除 |
| 会话管理 | 创建、加载、删除 |
| 委托深度 | 嵌套委托防护 |
| Fallback 链 | 模型降级 |
| 并发执行 | asyncio.gather 并行工具调用 |

### 未直接测试的模块

以下模块通过集成测试间接覆盖：
- **logger.py**: 日志格式化、请求 ID
- **skill.py**: Skill/SkillRegistry 数据类
- **cache.py**: LLMCache LRU 缓存
- **icons.py**: 图标路径和文本映射

---

## 10. 已发现并修复的 Bug

### Bug 1: Plan 模式消息重复 (已修复)

**位置**: `src/weather_agents/cli/main.py` 第 1120-1123 行

**症状**: 在 Plan 模式下，用户确认执行后，`[PLAN] {inp}` 消息保留在记忆中，同时 `inp` 作为新用户消息再次添加，导致用户输入在对话历史中出现两次。

**修复**: 确认执行后调用 `agent._pop_last_user_message()` 移除计划阶段的用户消息，再进入实际对话流。

**影响**: 仅影响 Plan 模式（`INTERACTIVE_MODE == "plan"`）。

### Bug 2: Fair 颜色 "gold" 无效 (已修复 — 此前会话)

**位置**: `src/weather_agents/core/icons.py`

**症状**: Rich 不识别命名颜色 `"gold"`，抛出 `MissingStyle` 错误。

**修复**: 改为 `"#FFD700"` 十六进制颜色值。

### Bug 3: `_history_idx` UnboundLocalError (已修复 — 此前会话)

**位置**: `src/weather_agents/cli/main.py`

**症状**: `_read_line_with_popup` 中 `_history_idx -= 1` 缺少 `global` 声明。

**修复**: 添加 `global _history_idx`。

### 剩余潜在问题

以下问题被评估为"不是缺陷"，不影响功能：
- `_COMMAND_LOOKUP` 定义后未使用（死代码）
- `_handle_apikey_command` 中 `os.environ.pop` 重复调用（无害）
- `up`/`down` 键在 popup 可见时存在两个处理分支（第二个为死代码，无害）

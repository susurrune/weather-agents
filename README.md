<div align="center">

# Skyloom

**雾 · 雨 · 霜 · 雪 · 露 · 晴**

*六位 Agent，一支团队。*

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/susurrune/skyloom/actions/workflows/ci.yml/badge.svg)](https://github.com/susurrune/skyloom/actions)
[![Tests](https://img.shields.io/badge/tests-969_🌡️-8A2BE2)](https://github.com/susurrune/skyloom)

</div>

---

**Skyloom** 是一个本地优先的多智能体终端框架。六个 Agent 各司其职——研究、生成、
审查、编排、运维、陪伴——通过事件总线、技能系统、DAG 编排引擎协作完成完整工作流。

不是又一个 LLM 聊天客户端，而是一个**分工明确的 AI 团队**。

```bash
pipx install git+https://github.com/susurrune/skyloom.git
sky                                       # 交互式对话
sky task "用 Go 实现 URL 短链接服务"      # 多 Agent 协作
sky web                                   # 语音聊天（晴专属）
```

---

## Agent 团队

| Agent | 角色 | 核心技能 |
|:------|:-----|:---------|
| **Fog** 雾 · research | 探索研究 | `web_research` · `code_analysis` · `document_analysis` |
| **Rain** 雨 · codegen | 生成实现 | `code_generator` · `content_writer` · `data_transformer` |
| **Frost** 霜 · review | 审查优化 | `code_reviewer` · `security_auditor` · `performance_checker` |
| **Snow** 雪 · planning | 规划编排 | `task_planner` · `arch_designer` · `workflow_designer` |
| **Dew** 露 · devops | 运维执行 | `sys_operator` · `ci_cd_manager` · `api_integrator` |
| **Fair** 晴 · companion | 情感陪伴 | `emotional_companion` · `self_evolve` |

每个 Agent 有独立人格、独立 LLM 配置、独立技能集。对话中切换：`/fog` `/rain` `/frost` `/snow` `/dew` `/fair`。

---

## 快速开始

```bash
# 安装
pipx install git+https://github.com/susurrune/skyloom.git

# 首次配置（向导选模型 + 填 API Key + 设画像）
sky init

# 进入对话
sky
```

支持 OpenAI / Anthropic / DeepSeek / Ollama 等，通过 LiteLLM 接入。

```bash
sky config set api_key.deepseek sk-xxx       # 设置 API Key
sky config set cli.default_agent rain        # 默认启动 Agent
sky config set model.fog deepseek/deepseek-v4-flash   # 每 Agent 独立模型
```

---

## 核心能力

### 智能路由

输入自动分三级，< 1ms 决策：

| 级别 | 触发 | 行为 |
|:-----|:-----|:-----|
| `direct` | 问候、短问答 | 单 Agent 直接回复，0 额外 LLM 调用 |
| `single` | 明确单领域请求 | 关键词路由到最佳 Agent |
| `orchestrate` | 复杂/跨领域 | Snow 拆解为 DAG → 多 Agent 协作 |

### Pipeline 模板

预定义工作流，命中后跳过 Snow 的 LLM 拆解：

`code_review` · `research_then_write` · `implement_and_review` · `implement_test_deploy`

### 三层记忆

| 层 | 范围 | 存储 |
|:---|:-----|:-----|
| Short-term | 会话 | SQLite（自动截断 + 去悬挂 tool_call） |
| Working | 任务 | 内存 |
| Long-term | 持久 | SQLite KV（检索注入，模糊搜索） |

### 用户画像 & 情感记忆

```bash
sky profile set 称呼 阿K                  # 固定事实，跨 Agent 共享
sky memories add "最近在准备考试"         # 近况心情，自动注入 prompt
sky persona set fair                       # 重写 Agent 角色
```

- 满 40 条记忆自动折叠成摘要
- 晴启动时主动提起你上次说的事

### 电脑操作（10 个跨平台工具）

| 类别 | 工具 |
|:-----|:-----|
| 应用 | `launch_app` · `open_path` · `browser_open` · `list_installed_apps` |
| 诊断 | `system_info` · `system_diagnose` |
| 进程 | `list_processes` · `kill_process` |
| 软件 | `package_manager`（winget/brew/apt/dnf/pacman 自动识别） |
| 服务 | `service_control` |

### MCP 自集成

```bash
# 运行时接入任意 MCP server，持久化
mcp_add_server name=fs command=npx args="-y @modelcontextprotocol/server-filesystem /path"

# 自动写一个新的 MCP server（零依赖 Python stub）
mcp_scaffold_server name=my_tool
```

### 内置工具

| 分类 | 工具 |
|:-----|:-----|
| 文件 | `read_file` · `write_file` · `edit_file` · `move_file` · `copy_file` · `delete_file` |
| 搜索 | `list_directory` · `tree` · `file_search` · `code_search` · `web_search` · `fetch_page` |
| Shell | `shell_exec`（argv 列表 + 黑名单，无管道/元字符） |
| HTTP | `http_get` · `http_post` |
| Git | `git_status` · `git_diff` · `git_log` · `git_add` · `git_commit` · `git_checkout` |
| 记忆 | `set_user_profile` · `remember` · `set_persona` |

### 技能系统

17 个内置技能，运行时动态激活：

```bash
sky chat frost
> /use security_auditor       # 激活安全审计
> 审计这段 Go 代码
> /deactivate                 # 回基础模式
```

---

## 语音聊天

晴（Fair）专属，豆包 TTS（火山引擎）。

```bash
# 配置（环境变量最快）
$env:DOUBAO_TTS_API_KEY = "你的火山引擎ApiKey"

# 或写入配置文件
sky config set tts.api_key 你的火山引擎ApiKey

# 启动
sky web                       # 0.0.0.0:8765，自动 HTTPS
sky web list                  # 查看可用音色
sky web select uranus         # 切换音色
```

### 同一 Wi-Fi 下手机访问

`sky web` 检测到内网 IP 会自动生成自签证书启用 HTTPS（手机麦克风需要 HTTPS）。
启动后控制台列出 `https://<内网 IP>:8765`，手机连同一 Wi-Fi 打开即可（首次会提示证书警告，点继续）。

### 跨网络（4G / 5G / 公司网）— Cloudflare Tunnel

最可靠的方式是两个终端分开起：

```bash
# 终端 A：本地起 HTTP 服务（loopback 跳过自动 HTTPS，避免回源 TLS 握手失败）
sky web --host 127.0.0.1 --port 8765

# 终端 B：起 Cloudflare Quick Tunnel
cloudflared tunnel --url http://localhost:8765
# 输出形如 https://xxx-yyy-zzz.trycloudflare.com
```

手机扫码或直接访问该 URL 即可——Cloudflare 的 HTTPS 边缘顺带解决手机麦克风权限。
任何反向代理（nginx / Caddy / ngrok）也是同样模式。

### 手机端 PWA

手机浏览器打开后「添加到主屏幕」，即得一个全屏带图标的类原生 App。

---

## CLI 参考

```bash
sky                    # 交互式对话
sky chat [agent] [msg] # 指定 Agent 对话
sky task <goal>        # 多 Agent 编排
sky web                # 语音服务器
sky web list           # TTS 音色列表
sky web select <v>     # 切换音色
sky init               # 配置向导
sky status             # Agent 状态
sky config <action>    # 配置管理
sky profile <action>   # 用户画像
sky memories <action>  # 情感记忆
sky persona <action>   # 角色设定
sky version            # 版本信息
```

交互命令：`/help` `/model` `/apikey` `/skills` `/memory` `/workspace` `/quit`

---

## 项目结构

```
src/weather_agents/
├── cli/              # 入口层：Typer 命令 + REPL（拆为 9 个内聚模块）
│   ├── main.py       4016 行   命令注册、交互循环
│   ├── wizard.py     291 行    首次启动向导
│   ├── dashboard.py            sky task 实时编排面板
│   ├── rendering.py            响应渲染（流式 + 最终面板）
│   ├── tool_display.py         工具状态显示
│   ├── pickers.py              交互选择控件
│   ├── keys.py                 跨平台键盘输入
│   ├── console.py              共享 Rich Console
│   └── mode.py                 auto/plan 模式状态
├── core/             # 核心层（与 UI 无关）
│   ├── agent.py      2789 行   BaseAgent + LLM 循环 + 并行工具执行
│   ├── memory.py     961 行    三层记忆 + SQLite
│   ├── llm.py        943 行    LiteLLM 封装 + 流式 + prompt cache
│   ├── factory.py    858 行    装配 + 编排
│   ├── mcp.py        855 行    MCP 客户端 + Manager
│   ├── config.py     810 行    配置 + 模型目录
│   └── …             tool / skill / pipelines / router / profile / bus / …
├── agents/           # 6 个 Agent 人设
├── tools/            # 内置工具：builtin / computer / mcp_tools / delegate
├── skills/           # 17 个可激活技能
└── web/              # 语音 WebSocket + Cloudflare Tunnel 工具 + TTS
```

详见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

---

## 开发

```bash
pip install -e ".[dev]"
pytest tests/ -q              # 969 tests
ruff check src/ tests/        # lint
ruff format src/ tests/       # 格式化
mypy src/                     # 类型检查
```

---

## 后续规划

### v1.5 ✅ 已完成
- [x] `core/agent.py` 解耦（2789→2248）
- [x] `tools/builtin.py` 按主题拆分（2183→1229）
- [x] mypy strict 0 错误
- [x] CI 优化（—cov 拆分，提速 ~30%）

### v2.0（进行中）
- [x] **工作流持久化** `sky task --resume`
- [x] **MCP 双向桥** `sky mcp`
- [x] **Web 记忆面板**（API + UI）
- [x] **Agent 自演进**（全部 Agent 已有 `self_evolve`）
- [x] **本地语义检索**（FTS5 + n-gram Jaccard 三层召回，CJK 原生支持）
- [ ] **多 TTS 供应商**（按需，不破坏豆包）

### 长期（探索）

- [ ] **多用户/多机器同步**：profile/memories 加密同步
- [ ] **Plugin marketplace**：用户提交的工具 + 技能社区
- [ ] **Agent A/B**：同一个目标让两套配置的 Agent 团队竞争，挑选更优结果

### 不做（明确边界）

- ❌ 桌面 GUI 应用（pywebview/Electron）— 太脆弱，PWA + 终端足够
- ❌ 闭源云端版本 — 本地优先是哲学，不是过渡形态
- ❌ Streamlit/Gradio 界面 — Skyloom 是 CLI-first 工具

---

## 项目状态

21,700+ 行 Python，969 tests，ruff + mypy 零错误，CI 全绿。

**已落地**：6 Agent · 三级路由 · DAG 编排 · Pipeline 模板 · 三层记忆 ·
电脑操作 · MCP 自集成 · 用户画像 + 情感记忆 · 角色自定义 · 并行工具执行 ·
跨平台键盘 · 配置向导 · 语音 + PWA · Web 搜索（DDG+Bing 竞速） · 费用追踪 ·
安全守卫（pkill regex/cmd injection/argv 列表）· 上下文压缩 · 100 个工具

**技术栈**：Python 3.11+ · LiteLLM · Typer · Rich · aiohttp · aiosqlite · httpx

**[MIT License](LICENSE)**

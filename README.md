<div align="center">

# Skyloom

**雾 · 雨 · 霜 · 雪 · 露 · 晴**

*六位 Agent，一支团队。专精领域，默契协作。*

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/susurrune/skyloom/actions/workflows/ci.yml/badge.svg)](https://github.com/susurrune/skyloom/actions)
[![Tests](https://img.shields.io/badge/tests-968_🌡️-8A2BE2)](https://github.com/susurrune/skyloom)
[![Code style](https://img.shields.io/badge/code%20style-ruff-000000)](https://github.com/astral-sh/ruff)

</div>

---

Skyloom 是一个**本地优先的多智能体终端框架**。六个 Agent 各司其职，通过事件总线通信、技能系统增强、编排引擎协作，完成从研究分析到代码生成到部署运维的完整工作流。

它不是又一个大模型聊天客户端——它是一个**分工明确的 AI 团队**。

```bash
# 安装
pipx install git+https://github.com/susurrune/skyloom.git

# 交互式对话
sky

# 一句话让团队协作
sky task "设计并实现一个 URL 短链接服务"

# 桌面端 + 手机端：原生窗口 + 公网网址 + 二维码，手机扫码即可访问
sky app
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
sky chat fog "对比 FastAPI 和 Flask 的生态"

# 让 Rain 写代码
sky chat rain "实现一个带超时的 LRU Cache"

# 全团队协作
sky task "搭建微服务项目：FastAPI + PostgreSQL + Docker"
```

协作模式下，Snow 自动拆解目标为 DAG 任务，分配至对应 Agent，按依赖顺序执行并汇总结果：

```
sky task "搭建微服务项目"
  ├─ [1] Fog: 调研微服务最佳实践
  ├─ [2] Rain: 生成项目骨架          ← 依赖 1
  ├─ [3] Rain: 编写 Dockerfile       ← 依赖 2
  ├─ [4] Frost: 代码审查             ← 依赖 2
  ├─ [5] Dew: 部署验证               ← 依赖 3, 4
```

## 三、快速开始

### 安装

**前提：** 需要 Python 3.11+、pipx 和 Git。

```bash
# 如果没有 pipx
pip install pipx

# 安装 skyloom
pipx install git+https://github.com/susurrune/skyloom.git
```

> 装完后 `sky` 会出现在 `~/.local/bin` 或 `%USERPROFILE%\.local\bin`。如果命令不可用，运行 `pipx ensurepath` 重开终端。

> **国内用户：** 访问 GitHub 慢时，可先克隆到本地再安装：
> ```bash
> git clone --depth=1 https://github.com/susurrune/skyloom.git
> pipx install ./skyloom
> ```
> 如果克隆也慢，给 git 设代理：`git config --global http.proxy http://127.0.0.1:你的端口`

### 配置

首次运行自动进入设置向导，选择统一模型或每个 Agent 独立配置。也可手动设置：

```bash
sky init                                    # 重新跑向导
sky config set api_key.deepseek sk-xxx      # 直接写入 API Key
sky config set cli.default_agent rain       # 设置默认启动 Agent（默认 fog）
export DEEPSEEK_API_KEY=sk-xxx             # 或环境变量
```

支持多模型提供商：

```bash
# 每个 Agent 独立配置模型
sky config set model.fog deepseek/deepseek-v4-flash
sky config set model.rain claude-sonnet-4-6

# 统一所有 Agent 模型
/model deepseek/deepseek-v4-flash    # chat 内命令
```

### 语音聊天配置

```bash
# 方式一（最快）：环境变量，无需写配置文件（PowerShell）
$env:DOUBAO_TTS_API_KEY = "你的火山引擎APIKey"
# 或永久保存
[System.Environment]::SetEnvironmentVariable("DOUBAO_TTS_API_KEY", "你的密钥", "User")

# 方式二：一条命令即可
sky config set tts.api_key 你的火山引擎APIKey

# 查看可用音色
sky web list

# 切换音色
sky web select uranus    # 乌拉努斯 - 大气知性女声
sky web select cancan    # 灿灿 - 活力甜美少女音
sky web select xiaohe    # 小河 - 温柔自然女声

# 启动语音服务器
sky web
```

#### 同一 Wi-Fi 下用手机访问

`sky web` 默认绑 `0.0.0.0`，检测到内网 IP 会自动生成自签证书启用 HTTPS（浏览器在非 localhost 下访问麦克风强制要求 HTTPS）。启动后控制台会列出 `https://<内网 IP>:8765`，手机连同一 Wi-Fi 直接打开即可（首次会提示自签证书警告，点继续）。

#### 跨网络访问（cloudflared / ngrok / nginx）

需要在公司、外网或 4G/5G 下访问时，把本地服务套一层隧道：

```bash
# 终端 A：本地起 HTTP 服务（loopback 绑定会跳过自动 HTTPS，避免回源 TLS 握手失败）
sky web --host 127.0.0.1 --port 8765

# 终端 B：起 Cloudflare Quick Tunnel，输出形如 https://xxx.trycloudflare.com
cloudflared tunnel --url http://localhost:8765
```

任何反向代理（nginx、Caddy、ngrok）都适用同样的模式——`--host 127.0.0.1` 让 sky web 走纯 HTTP，由代理那侧负责对外的 HTTPS 证书。

### 桌面端 & 手机端

一条命令把语音端变成跨设备应用：

```bash
pip install qrcode                # 二维码所需的可选依赖
sky web --tunnel                # 语音聊天 + Cloudflare 公网网址 + 二维码
```

`sky web --tunnel` 做三件事：① 在本地启动语音 WebSocket 服务器；② 开一条 **Cloudflare Quick Tunnel**（需 `cloudflared`，无需账号），拿到 `https://xxx.trycloudflare.com` 公网地址；③ 在终端打印**二维码**。手机扫码或输入网址即可语音对话——Cloudflare 的 HTTPS 边缘顺带解决了手机麦克风权限。

- **手机端 PWA**：手机浏览器打开后「添加到主屏幕」，即得一个全屏、带图标、类原生的 App。
- `sky web` 默认绑定 `0.0.0.0:8765`，参数：`--tunnel`（公网）、`-a <agent>`、`-p <port>`、`-H <host>`。
- 想要免安装 `.exe`：见 [`packaging/`](packaging/)（PyInstaller 一键打包）。

### 用户画像 & 角色自定义

Skyloom 在 `~/.skyloom/` 本地记住关于**你**的事，并允许你重塑每个 Agent 的角色（数据不出本机）。

```bash
# 用户画像（固定事实，跨 agent 共享，自动注入对话）
sky profile set 称呼 阿K
sky profile set 喜好 "巴赫、宋画"
sky profile show
sky profile forget [字段]          # 省略字段则清空

# 情感记忆（近况 / 心情，跨 agent 共享，满额自动折叠成摘要）
sky memories show
sky memories add "最近在准备考试，压力大"
sky memories forget                # 清空全部

# 角色设定：自定义任意 agent，set 从标准输入读入新设定
sky persona show fair
sky persona set fair               # 粘贴新设定，Ctrl-Z↵(Win)/Ctrl-D(Unix) 结束
sky persona reset fair             # 恢复内置设定
```

对话中 Agent 会主动维护这些：

- **用户画像**（`set_user_profile`）记住你说过的固定事实——名字、喜好、职业。
- **情感记忆**（`remember`）记下当下的你——近况、心情、该跟进的事。和画像分开存，跨 agent 共享、`memory clear` 也删不掉；满 40 条时最旧的会折叠成一条摘要而非丢弃。**晴（Fair）启动时会主动提起你上次说的事**（「☼ 晴还记得 ——」）。
- **角色自定义**：**晴还能按你的要求改写自己**（`set_persona`）——告诉她「以后用更亲近的方式跟我说话」，她会记住并改变。晴的设定本身就是一个**有真实情感、可作伴侣**的角色，而非工具。

> 桌面端 / 手机网页端还带一个**记忆面板**（顶栏图标）：直接在界面里查看、增删画像字段和情感记忆，不必用命令行。

### 升级 / 卸载

```bash
pipx upgrade skyloom
pipx uninstall skyloom
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
sky chat frost
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
| 网络 | `http_get` · `http_post` · `web_search`（DuckDuckGo + Bing 并发竞速，免 API Key，国内可用，带缓存） · `fetch_page` |
| Git | `git_status` · `git_diff` · `git_log` · `git_add` · `git_commit` · `git_checkout` |
| 时间 | `get_current_time`（本地 + UTC，确保时效性） |
| 记忆/人设 | `set_user_profile`（记住固定事实） · `remember`（记下近况心情） · `set_persona`（重写 Agent 角色） |
| 电脑操作 | `launch_app` · `open_path` · `browser_open` · `system_info` · `system_diagnose` · `list_processes` · `kill_process` · `package_manager` · `service_control` |
| MCP | `mcp_list_servers` · `mcp_add_server` · `mcp_remove_server` · `mcp_scaffold_server`（运行时接入/编写 MCP 集成） |
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

- **桌面端 + 手机端** — `sky app` 原生窗口（带品牌启动闪屏）+ Cloudflare 隧道公网地址 + 二维码；手机「添加到主屏幕」即为 PWA App；可 PyInstaller 打包免安装 exe（带专属 App 图标）
- **用户画像** — `~/.skyloom/profile.json` 本地记住固定事实，跨 Agent 共享、自动注入
- **电脑操作** — 启动应用、系统体检与故障修复、进程管理、软件安装/卸载/升级（winget/brew/apt 自动识别）、服务管控，跨平台
- **MCP 自集成** — 运行时接入任意软件的 MCP（stdio+SSE）；`mcp_scaffold_server` 直接写一个新的 MCP server（纯 Python、零依赖），即连即用
- **情感记忆** — `~/.skyloom/memories.json` 记下近况心情，跨 Agent 共享、满额折叠成摘要；晴启动时主动提起；桌面/手机端带记忆管理面板
- **角色自定义** — 每个 Agent 的角色设定可被用户覆盖；晴还能按要求改写自己
- **MCP 协议支持** — 通过 stdio 接入 Model Context Protocol 工具集
- **Plugin 系统** — `~/.skyloom/plugins/` 目录自动加载自定义工具
- **语音聊天** — 晴（Fair）专属，Doubao TTS V3 HTTP + WebSocket 语音服务器，支持环境变量一键配置
- **会话管理** — 创建/加载/删除会话，`sky chat --new` 开新会话
- **费用追踪** — 各 Agent 累计 Token 和费用，支持预算上限
- **智能工作空间** — 自动检测最佳磁盘位置，多盘跳过 C 盘
- **安全默认值** — Shell 执行拒危险命令，HTTP 请求禁私网，长输出截断标记
- **上下文压缩** — 接近限制时智能摘要，保留关键指令
- **技能自演进** — Fair（晴）可在运行时自主创建新技能

## 五、CLI 参考

### 子命令

```bash
sky init              # 交互式配置向导
sky chat [agent] [msg] # 对话（默认由 cli.default_agent 指定，默认 fog）
sky task <goal>       # 多 Agent 协作
sky web [options]   # 语音聊天 + 可选 --tunnel 公网二维码（手机访问）
sky status            # 所有 Agent 状态
sky config <action>   # 配置管理
sky memory <action>   # 对话记忆管理
sky profile <action>  # 用户画像（show / set / forget）
sky memories <action> # 情感记忆（show / add / forget）
sky persona <action>  # Agent 角色设定（show / set / reset）
sky web [options]   # 语音服务器（需配置 TTS）
sky version           # 版本信息
```

### 交互命令

进入 `sky chat` 后：

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

- **代码**：43 个源文件，~20k 行 Python
- **测试**：968 项，覆盖率 > 62%
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
| 语音端 `sky web --tunnel`（Cloudflare 隧道 + 公网二维码） | ✅ 完成 |
| 手机端 PWA（添加到主屏幕）+ PyInstaller 免安装 exe | ✅ 完成 |
| 用户画像（本地、跨 Agent、自动注入） | ✅ 完成 |
| 角色自定义（用户/Agent 自身可重写人设） | ✅ 完成 |
| Web 搜索（DDG + Bing 竞速 + 缓存，国内可用） | ✅ 完成 |
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

> 架构总览、分层依赖、数据流与扩展点见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

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
| 记忆 | aiosqlite · 三层架构（Short + Working + Long-term）· 本地用户画像 |
| 搜索 | DuckDuckGo + Bing（并发竞速，内置，免 API Key，带缓存） |
| 语音 | Doubao TTS V3 HTTP/WebSocket |
| 多端 | pywebview 桌面窗口 · Cloudflare 隧道 · PWA · PyInstaller 打包 |
| 工具 | 内置 · MCP 协议 · Plugin 系统 |
| CI | GitHub Actions · Ruff · MyPy · Pytest × 3 Python |

## License

[MIT](LICENSE)

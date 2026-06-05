# Skyloom 架构

> 一个多智能体 AI 编排框架：六个性格化 agent（雾/雨/霜/雪/露/晴）协作完成研究、
> 生成、审查、编排、运维与陪伴。本文档描述代码的分层结构、数据流与扩展点。

## 设计原则

1. **分层依赖，单向向下**：`entrypoints → core → (agents · tools · skills · mcp)`。
   上层依赖下层，下层永不反向 import 上层。`core` 不知道 `cli` / `web` 的存在。
2. **每个 agent 完全独立**：各自的 `ToolRegistry`、`SkillRegistry`、`Memory`、
   `LLMClient` 视图——没有跨 agent 的全局可变单例（MCP 的 active-manager 是唯一例外，
   且只用于运行时自集成）。
3. **能力即数据**：工具/技能/MCP server 都是可注册的对象，运行时可增删，无需改核心。
4. **安全姿势内建**：危险工具走审批门；shell 用 argv（无管道/元字符）；SSRF 守卫。
5. **性能预算**：简单问答 < 1 次 LLM 调用、< 2s 首字节；工具并行执行。

## 目录分层

```
src/weather_agents/
├── cli/              # 入口层：Typer 命令 + 交互式 REPL + 渲染（不含业务逻辑）
│   ├── main.py       #   命令注册、交互循环、流式渲染、配置向导
│   ├── console.py    #   共享 Rich Console 单例（Live 协调的唯一终端句柄）
│   ├── dashboard.py  #   sky task 的实时编排面板（TaskDashboard）
│   └── mode.py       #   auto / plan 模式状态
├── web/              # 入口层：语音/桌面/PWA
│   ├── server.py     #   aiohttp WebSocket 语音服务 + REST（画像/记忆 API）
│   ├── desktop.py    #   pywebview 原生窗口 + 启动闪屏
│   ├── tunnel.py     #   Cloudflare Quick Tunnel（手机公网访问）
│   ├── tts.py        #   Doubao TTS（HTTP + WebSocket）
│   └── certs.py      #   自签 TLS（LAN HTTPS，麦克风需要）
├── core/             # 核心层：与 UI 无关的全部业务逻辑
│   ├── agent.py      #   BaseAgent：LLM 循环、工具调用、记忆、技能激活
│   ├── factory.py    #   create_system_context() 装配 + orchestrate_task 编排
│   ├── llm.py        #   LLMClient：LiteLLM 封装、流式、prompt cache、重试
│   ├── memory.py     #   三层记忆（short/working/long-term）+ SQLite 持久化
│   ├── mcp.py        #   MCP 客户端（stdio+SSE）+ Manager + 运行时自集成
│   ├── tool.py       #   Tool / ToolRegistry / 结果缓存 / 熔断器
│   ├── tool_router.py#   每回合工具子集选择（<1ms，省 token + 提精度）
│   ├── router.py     #   goal 分类：direct / single / orchestrate
│   ├── pipelines.py  #   已知协作形态匹配（跳过 Snow 的分解 LLM 调用）
│   ├── skill.py      #   Skill / SkillRegistry（按需激活的能力包）
│   ├── profile.py    #   用户画像 + 情感记忆 + 自定义人设（本地）
│   ├── config.py     #   配置加载/合并 + 模型目录 + 配置目录迁移
│   ├── bus.py        #   事件总线（agent 间消息 + 可观测事件）
│   ├── middleware.py #   工具调用前后钩子链
│   ├── circuit_breaker.py # 工具级熔断
│   └── workspace.py  #   智能工作区检测（多盘、跳过 C 盘）
├── agents/           # 六个 agent 的人设与技能绑定（薄层，仅声明）
│   ├── fog.py rain.py frost.py snow.py dew.py fair.py
├── tools/            # 工具实现（注册进 core.ToolRegistry）
│   ├── builtin.py    #   文件/目录/shell/HTTP/git/搜索/时间/画像
│   ├── computer.py   #   电脑操作：启动应用/系统诊断/进程/包管理/服务
│   ├── mcp_tools.py  #   运行时 MCP 管理：增删/列出/脚手架生成
│   └── delegate.py   #   delegate_to（agent 间任务委派）
├── skills/           # 技能加载器（assets/builtin_skills/*/SKILL.md → Skill）
├── plugins/          # 用户插件加载（~/.skyloom/plugins/）
└── assets/           # 内置技能 SKILL.md + agent 图标
```

## 启动流程（冷启动）

```
sky chat
  └─ cli/main.py: _interactive()
       └─ core/factory.py: create_system_context()
            1. load_config()                    # 配置 + 模型目录
            2. ToolRegistry()                   # 注册 builtin + computer + mcp_tools
            3. SkillRegistry()                  # 加载 assets/builtin_skills
            4. PluginLoader                     # ~/.skyloom/plugins
            5. MCPManager（始终创建）            # config.yaml + 持久化 server 合并
            6. 每个 agent：克隆基础注册表 + 绑定 delegate_to
            7. mcp.bind_agents() + set_active_manager()
       └─ ctx.init_all()                        # MCP 并行连接 → agent.init()
```

## 请求流程

### 简单问答（`sky chat`）
```
用户输入 → agent.chat_stream()
  → tool_router 选 ~12 个相关工具
  → llm.stream_with_tools()（带 prompt cache）
  → 工具调用？→ 并行执行（见下）→ 回流给 LLM
  → 流式输出 + 三层记忆写入
```

### 复杂目标（`sky task`）
```
goal → router.classify()
  ├─ direct / single → 单 agent 快速路径（流式 + 实时状态）
  └─ orchestrate    → factory.orchestrate_task()
        → pipeline 匹配？命中则跳过 Snow 分解
        → Snow 分解为 DAG → TaskDashboard 实时显示
        → 拓扑序执行（无依赖任务并行）
        → judge（明显完成则跳过）→ 必要时 replan
```

## 关键机制

### 工具并行执行（agent.py `_llm_loop` / `chat_stream`）
LLM 一回合返回 N 个工具调用时，分四阶段：
- **A 解析**：解析参数、解析工具、发事件（串行，瞬时）
- **B 审批**：危险工具走审批门（串行，可能交互）
- **C 执行**：`asyncio.gather(return_exceptions=True)` **并行**全部工具
- **D 写回**：按原始顺序写结果（保持 `tool_call_id` 不变式）

N 个独立读操作从 ~N×T 降到 ~T。

### 三层记忆（memory.py）
| 层 | 范围 | 存储 | 不变式 |
|----|------|------|--------|
| short-term | 会话 | SQLite | `_prune_dangling_tool_calls`：tool 消息前必有匹配的 assistant.tool_calls |
| working | 任务 | 内存 | 任务执行期临时状态 |
| long-term | 持久 | SQLite KV | 检索注入：只取与当前 query 相关的事实 |

情感记忆 / 用户画像独立存于 `~/.skyloom/`（profile.py），跨 agent 共享。

### MCP 自集成（mcp.py + tools/mcp_tools.py）
运行时 `mcp_add_server` 接入任意 MCP server，工具传播进**所有** agent 的注册表
并刷新其缓存；`mcp_scaffold_server` 直接生成纯 Python MCP server 骨架。持久化到
`~/.skyloom/mcp_servers.json`，重启自动重连。

### prompt 缓存稳定性（agent.py `_rebuild_system_prompt`）
系统提示词按「基础人设 + 排序后的技能 + 运行时身份块」固定字节序拼接，使上游
prompt cache（Anthropic/DeepSeek 前缀缓存）跨回合命中，省首字节延迟。

## 扩展点

| 想加什么 | 在哪里 |
|----------|--------|
| 新工具 | `tools/` 写 handler → `register_*_tools()` 注册 |
| 新技能 | `assets/builtin_skills/<name>/SKILL.md` |
| 新 agent | `agents/<name>.py` 声明人设 + `factory.AGENT_CLASSES` |
| 接入软件 | 运行时 `mcp_add_server` 或 `~/.skyloom/plugins/` |
| 新 LLM 供应商 | `config` 的 providers.yaml（LiteLLM 透传） |

## 已知技术债 / 重构路线

按优先级（持续推进中）：

1. **`cli/main.py` 仍偏大（~4.9k 行）**。已抽出 `console` / `dashboard`；后续可继续
   抽出 `rendering`（流式/工具/欢迎渲染）、`commands`（各 Typer 命令）、`wizard`
   （配置向导）、`repl`（交互循环）。每次抽一个内聚单元，全量测试守护。
2. **`core/agent.py`（~2.8k 行）**：`chat_stream` 与 `_llm_loop` 有重叠的工具执行
   逻辑，可提取共享的 `_execute_tool_calls()` 助手。
3. **`tools/builtin.py`（~2.2k 行）**：可按主题拆为 `file_tools` / `net_tools` /
   `git_tools` / `search_tools`，与 `computer.py` / `mcp_tools.py` 并列。

重构守则：纯结构移动、不改行为、每步 `uv run pytest -x` 通过、危险红线见 `CLAUDE.md`。

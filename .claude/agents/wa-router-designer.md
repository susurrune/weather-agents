---
name: wa-router-designer
description: 设计/评审"简单问题不走完整 orchestration"的路由层。改 snow.py / router.py / factory.py 前调用。
tools: Read, Grep, Edit
model: sonnet
---

设计原则：

- **规则优先，LLM 兜底**：复杂度判定必须 < 50ms，不调 LLM。
- **三档**：`direct`（无工具单 agent 直答）/ `single`（单 agent + 工具）/ `orchestrate`（多 agent）。
- 判定信号：消息长度、动词链关键词（先/再/然后/接着/步骤）、多动词、文件路径、URL、问号数。

降级路径：
- `wa task "你好"` → `classify == "direct"` → 跳过 `snow.orchestrate`，直接挑 sunshine/rain 单 agent 回答。
- `wa task "先帮我看 X 再优化 Y"` → `classify == "orchestrate"` → 走完整 Snow 拆分。

返回：路由规则代码草案 + 边界用例清单（至少 20 条覆盖三档）。

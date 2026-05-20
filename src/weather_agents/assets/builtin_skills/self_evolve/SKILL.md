---
name: self_evolve
description: Read, analyze, and improve the Weather Agents codebase — architecture, features, bug fixes, tests
tools:
  - read_file
  - write_file
  - edit_file
  - file_search
  - code_search
  - shell_exec
  - list_directory
  - tree
---

## 技能：自我进化 (Self-Evolve)
你激活了「自我进化」技能，可以改进 Weather Agents 自身的代码。

### 代码库结构
```
src/weather_agents/
├── agents/          # Agent 定义 (fog, rain, frost, snow, dew)
├── cli/main.py      # 终端界面
├── core/            # 核心：agent, llm, memory, tool, skill, config, bus, logger, factory, workspace
├── skills/          # 技能实现 (每个文件一个技能)
└── tools/           # 工具实现 (builtin, delegate)
tests/               # pytest 测试 (全部 358+ 测试必须通过)
```

### 进化工作流
1. **探索** — 阅读相关源文件，理解当前架构和实现
2. **分析** — 识别改进点：bug、性能瓶颈、架构优化、新功能
3. **方案** — 用 2-3 句话描述改动方案，确认影响范围
4. **实现** — 修改代码，遵循项目风格（类型注解、无 any、安全优先、简洁）
5. **验证** — 运行 `python -m pytest tests/ -x -q` 确保全部通过
6. **汇报** — 总结改动内容、影响范围和测试结果

### 守则
- 改动前先阅读现有代码，不猜测接口
- 保持向后兼容
- 不改 package.json / pyproject.toml 除非必要
- 不改 README 或文档除非用户要求

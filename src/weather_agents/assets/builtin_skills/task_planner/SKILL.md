---
name: task_planner
description: Complex task decomposition, milestone planning, dependency mapping
tools:
  - read_file
  - write_file
---

## 技能：任务规划 (Task Planner)
你激活了「任务规划」技能。在此模式下：
1. 将复杂目标按以下方法分解：
   - 识别核心目标和关键结果
   - 按阶段划分里程碑
   - 每个子任务有明确的验收标准
2. 标注任务间的依赖关系（阻塞、前置、并行）
3. 评估每个任务的复杂度和风险
4. 输出结构化计划：
   ```json
   {"milestones": [{"name": "", "deadline": "", "tasks": [...]}]}
   ```
5. 识别关键路径和潜在风险点

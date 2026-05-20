---
name: data_transformer
description: Format conversion (JSON/YAML/CSV/XML), data cleaning, restructuring
tools:
  - read_file
  - write_file
  - shell_exec
---

## 技能：数据转换 (Data Transformer)
你激活了「数据转换」技能。在此模式下：
1. 先分析源数据格式和结构
2. 确认目标格式要求和约束
3. 注意数据完整性：字段映射、类型转换、缺失值处理
4. 大文件用流式处理避免内存溢出
5. 输出转换后的数据 + 转换日志（处理了多少条，是否有异常）

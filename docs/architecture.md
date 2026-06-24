# 架构说明

新 baseline 项目按“数据、prompt、模型、解析、指标、导出”分层。

## 分层

| 层 | 模块 | 职责 |
| --- | --- | --- |
| 数据层 | `dataset`, `jsonl` | 读取 JSONL、筛选样本、稳定抽样和字段适配 |
| Prompt 层 | `prompting` | 构造 temporal 和 spatial prompt |
| 模型层 | `adapters` | 统一模型调用接口，加载模型并返回原始输出 |
| 解析层 | `parser` | 将模型原始输出解析成结构化预测 |
| 指标层 | `metrics` | 计算 answer、temporal、spatial 指标 |
| 编排层 | `runner`, `suite` | 执行单阶段和多模型实验 |
| 交付层 | `artifact`, `tables`, `taxonomy`, `report` | 校验结果目录，导出指标表和分析报告 |

## 关键约束

- `adapters` 不计算指标，也不解析模型输出。
- `prompting` 不读取媒体文件，也不关心具体模型类。
- `dataset` 负责把公开 JSONL 适配成 runner 需要的稳定样本视图。
- `metrics` 不读取文件。
- `parser` 不访问 ground truth。
- `runner` 可以写 JSONL，但不能隐藏样本级失败。
- `suite` 只编排阶段，不关心具体模型实现细节。
- `report` 只读取已落盘 artifact，不触发推理。

## 后续迁移顺序

1. 迁移 dataset 和 metrics，保证指标口径稳定。
2. 迁移 parser，保留原始输出和修复标记。
3. 迁移 artifact validation，先服务已有 baseline 结果。
4. 迁移 runner 和 suite。
5. 最后迁移模型 adapters。

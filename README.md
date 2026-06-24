# EvidenceQA Baseline Refactor

这是一个从 0 开始重写的 EvidenceQA baseline 项目骨架。目标是用更清晰的模块边界、
Google Python Style 和可审计的实验输出，替代旧 baseline 中逐步堆叠出来的工程结构。

## 设计原则

- 实验可复现优先，任何结果文件都能追溯到配置、模型和数据 split。
- 模型适配器只负责推理，不负责解析和指标计算。
- Parser、metrics、artifact validation 都是纯工具模块，便于单独复用。
- 注释和 docstring 使用中文，解释实验语义和设计原因。
- 提交信息保持短、自然、能说明本次改动。

## 当前状态

第一版先建立项目骨架、artifact 校验和已有 baseline 结果分析工具，不包含模型推理适配器。

已包含：

- JSONL 数据读取和样本选择。
- Temporal / spatial 样本字段适配。
- 模型输出 JSON 解析。
- Answer / temporal / spatial 指标工具和 summary 汇总。
- Baseline artifact 结构校验。
- Metric CSV 导出。
- Grounded 阶段 answer/evidence 错误类型导出。
- Markdown 分析报告生成。
- 模型 adapter 基础接口。
- Temporal / spatial prompt 构造工具。
- CLI 入口。
- 提交规范和架构说明。

后续再逐步补：

- Qwen / LLaVA / InternVL adapter。
- Temporal runner。
- Spatial runner。
- Multi-model suite runner。
- 结果汇总页面或论文表格模板。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## CLI

查看项目版本：

```bash
evidenceqa-baseline-refactor version
```

校验 baseline artifact 目录：

```bash
evidenceqa-baseline-refactor validate-artifact /path/to/baseline-all-models
```

导出主指标 CSV：

```bash
evidenceqa-baseline-refactor export-tables /path/to/baseline-all-models outputs/analysis
```

导出 grounded 阶段错误类型：

```bash
evidenceqa-baseline-refactor export-taxonomy /path/to/baseline-all-models outputs/grounded_taxonomy.csv
```

生成完整分析报告和配套 CSV：

```bash
evidenceqa-baseline-refactor analyze-artifact /path/to/baseline-all-models outputs/analysis
```

## 目录

```text
src/evidenceqa_baseline_refactor/
  adapters/    # 模型 adapter 基础接口
  artifact.py   # 结果目录校验
  cli.py        # 命令行入口
  config.py     # 实验配置结构
  dataset.py    # JSONL 数据读取、样本选择与字段适配
  jsonl.py      # JSONL 通用读写
  metrics.py    # 指标计算和 summary 汇总
  parser.py     # 模型输出解析
  prompting.py  # Prompt 构造
  report.py     # Markdown 分析报告生成
  tables.py     # 主指标表导出
  taxonomy.py   # Grounded 错误类型导出
```

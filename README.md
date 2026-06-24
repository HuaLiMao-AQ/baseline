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

第一版只建立项目骨架和基础工具，不包含模型推理适配器。

已包含：

- JSONL 数据读取和样本选择。
- 模型输出 JSON 解析。
- Answer / temporal / spatial 指标工具。
- Baseline artifact 结构校验。
- CLI 入口。
- 提交规范和架构说明。

后续再逐步补：

- Qwen / LLaVA / InternVL adapter。
- Temporal runner。
- Spatial runner。
- Multi-model suite runner。
- 结果导出与分析脚本。

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

## 目录

```text
src/evidenceqa_baseline_refactor/
  artifact.py   # 结果目录校验
  cli.py        # 命令行入口
  config.py     # 实验配置结构
  dataset.py    # JSONL 数据读取与样本选择
  jsonl.py      # JSONL 通用读写
  metrics.py    # 指标计算
  parser.py     # 模型输出解析
```

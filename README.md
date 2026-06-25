# EvidenceQA Baseline

这是基于原始 EvidenceQA baseline 整理出的项目。实验逻辑、运行入口和输出契约按原项目保持，
代码整理只服务于可读性、目录清晰度和提交可审计性。

## 安装

推荐 Python 3.11+：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[vl]"
```

算力机镜像已经带 CUDA 版 torch 时，可先只补运行依赖：

```bash
python -m pip install -r requirements.txt
```

## 配置入口

运行参数通过命令行或环境变量指定。常用环境变量：

```bash
export EVIDENCEQA_DATASET_DIR=/path/to/public_dataset
export EVIDENCEQA_CACHE_DIR=/path/to/cache
export EVIDENCEQA_MODELS=qwen,llava,internvl
export EVIDENCEQA_SEED=20260621
export EVIDENCEQA_SAMPLE_MODE=sequential
export EVIDENCEQA_DTYPE=bfloat16
export EVIDENCEQA_MAX_FRAMES=64
export EVIDENCEQA_MAX_PIXELS=602112
export EVIDENCEQA_MAX_NEW_TOKENS=256
export EVIDENCEQA_HARDWARE_PROFILE=rtx-pro-6000-96gb-single-cuda
export EVIDENCEQA_INTERNVL_PYTHON=/path/to/internvl/python
```

运行时只指定缓存根目录，不手动改 Hugging Face 缓存结构。代码会在缓存根目录下
自动设置：

```text
HF_HOME=<cache>/huggingface
HF_HUB_CACHE=<cache>/huggingface/hub
HUGGINGFACE_HUB_CACHE=<cache>/huggingface/hub
HF_DATASETS_CACHE=<cache>/huggingface/datasets
HF_ASSETS_CACHE=<cache>/huggingface/assets
TRANSFORMERS_CACHE=<cache>/huggingface/hub
```

## 运行

默认研究入口：

```bash
python -u main.py
```

指定模型、阶段和样本数：

```bash
python -u main.py \
  --target answer_only \
  --models qwen \
  --limit 50 \
  --dataset-dir /path/to/public_dataset \
  --cache-dir /path/to/cache \
  --output-dir runs/qwen-answer-only-50 \
  --overwrite \
  --no-progress
```

AutoDL 示例：

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
EVIDENCEQA_DATASET_DIR=/root/autodl-tmp/public_dataset \
EVIDENCEQA_CACHE_DIR=/root/autodl-tmp/.cache \
PYTHONPATH=src /root/miniconda3/bin/python -u main.py \
  --target answer_only \
  --models qwen \
  --limit 50 \
  --output-dir /root/autodl-tmp/evidenceqa-baseline-runs/qwen-answer-only-50 \
  --overwrite \
  --no-progress
```

只跑 smoke：

```bash
python -u main.py --target smoke --models qwen --overwrite --no-progress
```

脚本入口：

```bash
bash scripts/run_baseline_suite.sh --models qwen --target smoke
```

## 目录

```text
main.py                         # 原项目研究入口
scripts/run_baseline_suite.sh   # 原项目运行脚本
src/evidenceqa_baseline/
  adapters/                     # Qwen、LLaVA、InternVL 适配器
  cache.py                      # 运行时缓存根目录配置
  cli.py                        # 包入口
  dataset.py                    # 数据读取、筛选和样本适配
  devices.py                    # CUDA 设备和 dtype 选择
  failure_report.py             # 失败样本报告
  log_utils.py                  # 运行日志
  media.py                      # 媒体路径解析和视频时长探测
  metrics.py                    # 指标计算和汇总
  parser.py                     # 模型输出解析
  progress.py                   # 进度显示
  prompting.py                  # Prompt 构造
  runner.py                     # Temporal runner
  spatial_runner.py             # Spatial runner
  suite.py                      # 多阶段 suite 编排
```

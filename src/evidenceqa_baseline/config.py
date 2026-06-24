"""baseline 实验配置结构。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

StageName = Literal["smoke", "answer_only", "grounded", "ref"]
PromptMode = Literal["answer_only", "grounded", "spatial"]


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """单个模型的可复现配置。

    Args:
        model_id: Hugging Face model id 或本地模型路径。
        model_slug: 用于输出目录的稳定短名。
        dtype: 推理精度。
        device: 推理设备。
    """

    model_id: str
    model_slug: str
    dtype: str = "bfloat16"
    device: str = "cuda"


@dataclass(frozen=True, slots=True)
class DataConfig:
    """数据集配置。

    Args:
        repo_id: Hugging Face dataset repo id。
        revision: 数据集 revision。
        split: 数据 split。
        local_jsonl: 可选本地 JSONL 路径。
    """

    repo_id: str
    revision: str
    split: str
    local_jsonl: Path | None = None


@dataclass(frozen=True, slots=True)
class StageConfig:
    """单个实验阶段配置。

    Args:
        name: 阶段名。
        prompt_mode: Prompt 模式。
        max_new_tokens: 最大生成 token 数。
        limit: 最大样本数；`None` 表示全量。
    """

    name: StageName
    prompt_mode: PromptMode
    max_new_tokens: int
    limit: int | None = None


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    """一次 baseline suite 的顶层配置。"""

    data: DataConfig
    models: tuple[ModelConfig, ...]
    stages: tuple[StageConfig, ...]
    output_dir: Path
    seed: int = 20260621
    sample_mode: str = "sequential"
    max_frames: int = 64
    max_pixels: int | None = 768 * 28 * 28


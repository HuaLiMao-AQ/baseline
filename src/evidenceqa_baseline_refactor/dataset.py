"""EvidenceQA JSONL 数据读取与样本选择。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .jsonl import read_jsonl

SampleMode = Literal["sequential", "random"]


@dataclass(frozen=True, slots=True)
class EvidenceSample:
    """统一样本视图。

    Args:
        sample_id: 样本 ID。
        task_type: 任务类型。
        source_dataset: 来源数据集。
        raw: 原始 JSON 记录。
    """

    sample_id: str
    task_type: str
    source_dataset: str
    raw: dict[str, Any]


def load_samples(path: Path) -> list[EvidenceSample]:
    """读取 EvidenceQA JSONL 并转换为轻量样本对象。"""

    samples: list[EvidenceSample] = []
    for row in read_jsonl(path):
        sample_id = str(row.get("id") or row.get("sample_id") or "")
        if not sample_id:
            raise ValueError("样本缺少 id")
        samples.append(
            EvidenceSample(
                sample_id=sample_id,
                task_type=str(row.get("task_type") or row.get("task") or "unknown"),
                source_dataset=str(row.get("source_dataset") or row.get("dataset") or "unknown"),
                raw=row,
            )
        )
    return samples


def filter_by_task(samples: list[EvidenceSample], task_type: str) -> list[EvidenceSample]:
    """按任务类型筛选样本。"""

    return [sample for sample in samples if sample.task_type == task_type]


def select_samples(
    samples: list[EvidenceSample],
    *,
    limit: int | None,
    seed: int,
    mode: SampleMode,
) -> list[EvidenceSample]:
    """稳定选择实验样本。"""

    if limit is None or limit >= len(samples):
        return list(samples)
    if limit < 0:
        raise ValueError("limit 不能为负数")
    if mode == "sequential":
        return list(samples[:limit])
    if mode == "random":
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(samples)), limit))
        return [samples[index] for index in indices]
    raise ValueError(f"未知 sample mode: {mode}")


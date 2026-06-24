"""模型适配器共享协议。"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from evidenceqa_baseline.dataset import DatasetSample, SpatialSample


class ModelAdapter(Protocol):
    """Runner 使用的最小模型接口。"""

    def predict(self, sample: DatasetSample, media_path: Path) -> str:
        """返回单个样本的原始模型输出。

        Args:
            sample: 已归一化的数据集样本。
            media_path: 已解析到本地的视频路径。

        Returns:
            模型生成的原始文本，不在适配器内做 JSON 解析。
        """

    def predict_spatial(
        self,
        sample: SpatialSample,
        frame_paths: list[tuple[int, Path]],
    ) -> str:
        """返回单个 spatial grounding 样本的原始模型输出。"""

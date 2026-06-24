"""模型 adapter 基础接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from evidenceqa_baseline_refactor.config import ModelConfig, PromptMode
from evidenceqa_baseline_refactor.dataset import (
    EvidenceSample,
    SpatialSample,
    TemporalSample,
)


class AdapterError(RuntimeError):
    """模型 adapter 抛出的统一异常。"""


class ModelAdapter(Protocol):
    """原 baseline runner 使用的最小模型接口。"""

    def predict(self, sample: TemporalSample, media_path: Path) -> str:
        """返回单个 temporal 样本的原始模型输出。"""

    def predict_spatial(
        self,
        sample: SpatialSample,
        frame_paths: list[tuple[int, Path]],
    ) -> str:
        """返回单个 spatial grounding 样本的原始模型输出。"""


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """一次生成调用的解码参数。

    Args:
        max_new_tokens: 最大生成 token 数。
        temperature: 采样温度；0 表示确定性解码。
        top_p: nucleus sampling 阈值；`None` 表示使用模型默认值。
        do_sample: 是否采样。
    """

    max_new_tokens: int
    temperature: float = 0.0
    top_p: float | None = None
    do_sample: bool = False


@dataclass(frozen=True, slots=True)
class PredictionRequest:
    """传给模型 adapter 的单样本请求。

    Args:
        sample: EvidenceQA 统一样本视图。
        prompt: 已渲染完成的 prompt。
        prompt_mode: 当前 prompt 模式。
        media_path: 视频或图片路径；文本任务可以为空。
        frame_paths: 空间任务的帧路径集合。
    """

    sample: EvidenceSample | TemporalSample | SpatialSample
    prompt: str
    prompt_mode: PromptMode
    media_path: Path | None = None
    frame_paths: tuple[Path, ...] = ()


@dataclass(frozen=True, slots=True)
class AdapterResponse:
    """模型 adapter 返回的原始响应。

    Args:
        raw_output: 模型原始文本输出。
        latency_seconds: 单样本推理耗时。
        metadata: adapter 侧可选元数据，例如采样帧数或模型内部参数。
    """

    raw_output: str
    latency_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseModelAdapter(ABC):
    """所有模型 adapter 的最小公共协议。"""

    def __init__(self, model_config: ModelConfig) -> None:
        self._model_config = model_config

    @property
    def model_config(self) -> ModelConfig:
        """返回当前 adapter 使用的模型配置。"""

        return self._model_config

    @abstractmethod
    def generate(
        self,
        request: PredictionRequest,
        generation_config: GenerationConfig,
    ) -> AdapterResponse:
        """执行单样本推理，返回未解析的模型输出。"""

    def close(self) -> None:
        """释放模型资源。

        具体 adapter 可以覆盖此方法来清理显存、关闭进程或断开服务连接。
        """

    def __enter__(self) -> "BaseModelAdapter":
        """进入上下文管理器。"""

        return self

    def __exit__(self, *_exc_info: object) -> None:
        """退出上下文管理器并释放资源。"""

        self.close()

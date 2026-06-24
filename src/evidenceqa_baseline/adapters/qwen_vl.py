"""基于官方 Transformers 接口的 Qwen-VL 适配器。"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evidenceqa_baseline.cache import hf_cache_kwargs
from evidenceqa_baseline.dataset import DatasetSample, SpatialSample
from evidenceqa_baseline.devices import select_device, select_dtype
from evidenceqa_baseline.prompting import (
    PROMPT_MODE_GROUNDED,
    build_qwen_messages,
    build_qwen_spatial_messages,
)

from .generation import generation_token_kwargs
from .transformers_io import decode_generated_suffix, load_pretrained_with_dtype

DEFAULT_QWEN_VL_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


class QwenVLAdapterError(RuntimeError):
    """Qwen-VL 加载或推理失败时抛出。"""


@dataclass(frozen=True, slots=True)
class QwenVLConfig:
    """Qwen-VL 推理配置。

    Attributes:
        model_id: Hugging Face 模型 ID 或本地缓存路径。
        device: 运行设备；PRO 6000 单卡默认使用 ``cuda``。
        dtype: 推理精度，默认 ``bfloat16``。
        max_frames: 视频最大采样帧数。
        fps: 可选视频采样 FPS。
        max_pixels: 可选视觉输入最大像素数。
        max_new_tokens: 最大生成 token 数。
        prompt_mode: ``answer_only`` 或 ``grounded``。
        do_sample: 是否启用采样，baseline 默认关闭。
    """

    model_id: str = DEFAULT_QWEN_VL_MODEL_ID
    model_cache_dir: Path | None = None
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_frames: int = 64
    fps: float | None = None
    max_pixels: int | None = 768 * 28 * 28
    max_new_tokens: int = 128
    prompt_mode: str = PROMPT_MODE_GROUNDED
    do_sample: bool = False


class QwenVLAdapter:
    """单样本、懒加载的 Qwen-VL 适配器。"""

    def __init__(self, config: QwenVLConfig) -> None:
        self.config = config
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None
        self._device: str | None = None

    def predict(self, sample: DatasetSample, media_path: Path) -> str:
        """执行 Qwen-VL 推理并返回解码文本。

        Args:
            sample: 已归一化的数据集样本。
            media_path: 已解析到本地的视频路径。

        Returns:
            模型生成的原始文本。

        Raises:
            QwenVLAdapterError: 样本缺少时长、依赖缺失或模型推理失败时抛出。
        """

        model, processor, torch, device = self._ensure_loaded()
        duration = sample.duration_seconds
        if duration is None:
            raise QwenVLAdapterError("样本缺少视频时长，无法构造时间问答 prompt")

        messages = build_qwen_messages(
            question=sample.question,
            duration_seconds=duration,
            media_path=media_path,
            fps=self.config.fps,
            max_frames=self.config.max_frames,
            max_pixels=self.config.max_pixels,
            prompt_mode=self.config.prompt_mode,
        )
        text = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=self.config.do_sample,
                max_new_tokens=self.config.max_new_tokens,
                **generation_token_kwargs(processor, model),
            )
        return decode_generated_suffix(processor, inputs, generated)

    def predict_spatial(
        self,
        sample: SpatialSample,
        frame_paths: list[tuple[int, Path]],
    ) -> str:
        """执行 Qwen-VL 空间定位推理并返回解码文本。"""

        model, processor, torch, device = self._ensure_loaded()
        messages = build_qwen_spatial_messages(
            question=sample.question,
            frame_paths=frame_paths,
            max_pixels=self.config.max_pixels,
        )
        text = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=self.config.do_sample,
                max_new_tokens=self.config.max_new_tokens,
                **generation_token_kwargs(processor, model),
            )
        return decode_generated_suffix(processor, inputs, generated)

    def _ensure_loaded(self) -> tuple[Any, Any, Any, str]:
        if self._model is not None and self._processor is not None:
            assert self._torch is not None
            assert self._device is not None
            return self._model, self._processor, self._torch, self._device

        try:
            import torch

            with contextlib.redirect_stdout(io.StringIO()):
                from transformers import AutoProcessor
        except ImportError as exc:
            raise QwenVLAdapterError(
                "Qwen-VL 推理缺少可选依赖；请安装 `.[vl]`。"
            ) from exc

        model_class = _resolve_qwen_vl_model_class(self.config.model_id)
        device = select_device(torch, self.config.device)
        dtype = select_dtype(torch, self.config.dtype, device)
        cache_kwargs = hf_cache_kwargs(self.config.model_cache_dir)

        model = load_pretrained_with_dtype(
            model_class,
            self.config.model_id,
            dtype=dtype,
            cache_kwargs=cache_kwargs,
        )
        processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            **cache_kwargs,
        )
        model.to(device)
        model.eval()

        self._model = model
        self._processor = processor
        self._torch = torch
        self._device = device
        return model, processor, torch, device

    def _process_vision_info(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[Any, Any]:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise QwenVLAdapterError(
                "Qwen-VL 视频推理缺少 qwen-vl-utils；请安装 `.[vl]`。"
            ) from exc
        return process_vision_info(messages)


def _resolve_qwen_vl_model_class(model_id: str) -> Any:
    normalized = model_id.lower().replace("_", "-")
    if "qwen2.5-vl" not in normalized and "qwen2-5-vl" not in normalized:
        raise QwenVLAdapterError(
            "当前 Qwen 适配器只支持 Qwen2.5-VL；"
            f"不支持的 model_id={model_id!r}。"
        )

    try:
        import transformers
    except ImportError as exc:
        raise QwenVLAdapterError(
            "Qwen-VL 推理缺少 transformers；请安装 `.[vl]`。"
        ) from exc

    model_class = getattr(transformers, "Qwen2_5_VLForConditionalGeneration", None)
    if model_class is not None:
        return model_class
    raise QwenVLAdapterError(
        "当前 transformers 版本不支持 Qwen2.5-VL，请更新 transformers。"
    )

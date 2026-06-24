"""LLaVA-OneVision adapter using Hugging Face Transformers."""

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
    build_frame_temporal_prompt,
    build_spatial_text_prompt,
)

from .frame_utils import frame_context, load_spatial_frames, sample_video_frames
from .generation import generation_token_kwargs

DEFAULT_LLAVA_ONEVISION_MODEL_ID = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
LLAVA_ONEVISION_MAX_IMAGE_FRAMES = 16


class LlavaOneVisionAdapterError(RuntimeError):
    """LLaVA-OneVision loading or inference failure."""


@dataclass(frozen=True, slots=True)
class LlavaOneVisionConfig:
    model_id: str = DEFAULT_LLAVA_ONEVISION_MODEL_ID
    model_cache_dir: Path | None = None
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_frames: int = 64
    max_pixels: int | None = 768 * 28 * 28
    max_new_tokens: int = 128
    prompt_mode: str = PROMPT_MODE_GROUNDED
    do_sample: bool = False


class LlavaOneVisionAdapter:
    """Lazy LLaVA-OneVision adapter for temporal and spatial baselines."""

    def __init__(self, config: LlavaOneVisionConfig) -> None:
        self.config = config
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None
        self._device: str | None = None

    def predict(self, sample: DatasetSample, media_path: Path) -> str:
        model, processor, torch, device = self._ensure_loaded()
        duration = sample.duration_seconds
        if duration is None:
            raise LlavaOneVisionAdapterError(
                "sample duration is required before prediction"
            )
        frames = sample_video_frames(
            media_path,
            duration_seconds=duration,
            max_frames=_bounded_image_frame_count(self.config.max_frames),
        )
        prompt = build_frame_temporal_prompt(
            question=sample.question,
            duration_seconds=duration,
            frame_context=frame_context(frames),
            prompt_mode=self.config.prompt_mode,
        )
        return self._generate_from_images(
            prompt,
            [frame.image for frame in frames],
            model=model,
            processor=processor,
            torch=torch,
            device=device,
        )

    def predict_spatial(
        self,
        sample: SpatialSample,
        frame_paths: list[tuple[int, Path]],
    ) -> str:
        model, processor, torch, device = self._ensure_loaded()
        frames = load_spatial_frames(
            _uniformly_limit_items(
                frame_paths,
                _bounded_image_frame_count(self.config.max_frames),
            )
        )
        prompt = build_spatial_text_prompt(
            question=sample.question,
            frame_context=frame_context(frames),
        )
        return self._generate_from_images(
            prompt,
            [frame.image for frame in frames],
            model=model,
            processor=processor,
            torch=torch,
            device=device,
        )

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
            raise LlavaOneVisionAdapterError(
                "LLaVA-OneVision inference requires transformers, torch, pillow, "
                'and decord. Install with: python -m pip install -e ".[vl]"'
            ) from exc

        model_class = _resolve_llava_onevision_model_class()
        device = select_device(torch, self.config.device)
        dtype = select_dtype(torch, self.config.dtype, device)
        cache_kwargs = hf_cache_kwargs(self.config.model_cache_dir)
        try:
            model = model_class.from_pretrained(
                self.config.model_id,
                dtype=dtype,
                low_cpu_mem_usage=True,
                **cache_kwargs,
            )
        except TypeError:
            model = model_class.from_pretrained(
                self.config.model_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                **cache_kwargs,
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

    def _generate_from_images(
        self,
        prompt: str,
        images: list[Any],
        *,
        model: Any,
        processor: Any,
        torch: Any,
        device: str,
    ) -> str:
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = processor(
            text=[text],
            images=images,
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
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated)
        ]
        decoded = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0] if decoded else ""


def _resolve_llava_onevision_model_class() -> Any:
    try:
        import transformers
    except ImportError as exc:
        raise LlavaOneVisionAdapterError(
            "LLaVA-OneVision inference requires transformers. "
            'Install with: python -m pip install -e ".[vl]"'
        ) from exc

    for name in (
        "LlavaOnevisionForConditionalGeneration",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
    ):
        model_class = getattr(transformers, name, None)
        if model_class is not None:
            return model_class
    raise LlavaOneVisionAdapterError(
        "当前 transformers 版本不支持 LLaVA-OneVision，请更新 transformers。"
    )


def _bounded_image_frame_count(max_frames: int) -> int:
    if max_frames <= 0:
        return LLAVA_ONEVISION_MAX_IMAGE_FRAMES
    return min(max_frames, LLAVA_ONEVISION_MAX_IMAGE_FRAMES)


def _uniformly_limit_items(items: list[Any], limit: int) -> list[Any]:
    if limit <= 0 or len(items) <= limit:
        return items
    if limit == 1:
        return [items[0]]
    step = (len(items) - 1) / float(limit - 1)
    return [items[min(len(items) - 1, round(index * step))] for index in range(limit)]

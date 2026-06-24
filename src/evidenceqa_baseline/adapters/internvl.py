"""InternVL adapter using the official InternVL chat path."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from evidenceqa_baseline.cache import hf_cache_kwargs
from evidenceqa_baseline.dataset import DatasetSample, SpatialSample
from evidenceqa_baseline.devices import select_device, select_dtype
from evidenceqa_baseline.prompting import (
    PROMPT_MODE_GROUNDED,
    PROMPT_MODE_SPATIAL,
    build_frame_temporal_prompt,
    build_spatial_text_prompt,
)

from .frame_utils import (
    FrameImage,
    frame_context,
    load_spatial_frames,
    sample_video_frames,
)
from .generation import generation_token_kwargs

DEFAULT_INTERNVL_MODEL_ID = "OpenGVLab/InternVL2_5-8B"
INTERNVL_WORKER_PYTHON_ENV = "EVIDENCEQA_INTERNVL_PYTHON"
INTERNVL_WORKER_ACTIVE_ENV = "EVIDENCEQA_INTERNVL_WORKER"
AUTODL_INTERNVL_PYTHON = Path("/root/autodl-tmp/internvl-tf437/bin/python")
INTERNVL_IMAGE_SIZE = 448
INTERNVL_MAX_EAGER_ATTENTION_FRAMES = 8
INTERNVL_MIN_NEW_TOKENS = 16
INTERNVL_JSON_PREFIX = '{\n  "answer": '
INTERNVL_SPATIAL_JSON_PREFIX = '{\n  "target": '


class InternVLAdapterError(RuntimeError):
    """InternVL loading or inference failure."""


@dataclass(frozen=True, slots=True)
class InternVLConfig:
    model_id: str = DEFAULT_INTERNVL_MODEL_ID
    model_cache_dir: Path | None = None
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_frames: int = 64
    max_new_tokens: int = 128
    prompt_mode: str = PROMPT_MODE_GROUNDED
    do_sample: bool = False


class InternVLAdapter:
    """Lazy InternVL adapter for temporal and spatial baselines."""

    def __init__(self, config: InternVLConfig) -> None:
        self.config = config
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._torch: Any | None = None
        self._device: str | None = None
        self._dtype: Any | None = None
        self._transform: Any | None = None
        self._worker: _InternVLWorkerClient | None = None

    def predict(self, sample: DatasetSample, media_path: Path) -> str:
        if self._should_use_worker():
            return self._ensure_worker().predict(sample, media_path)

        model, tokenizer, torch, device, dtype = self._ensure_loaded()
        duration = sample.duration_seconds
        if duration is None:
            raise InternVLAdapterError("sample duration is required before prediction")
        frames = sample_video_frames(
            media_path,
            duration_seconds=duration,
            max_frames=_bounded_frame_count(self.config.max_frames),
        )
        prompt = build_frame_temporal_prompt(
            question=sample.question,
            duration_seconds=duration,
            frame_context=frame_context(frames),
            prompt_mode=self.config.prompt_mode,
        )
        return self._chat(
            prompt,
            frames,
            model=model,
            tokenizer=tokenizer,
            torch=torch,
            device=device,
            dtype=dtype,
        )

    def predict_spatial(
        self,
        sample: SpatialSample,
        frame_paths: list[tuple[int, Path]],
    ) -> str:
        if self._should_use_worker():
            return self._ensure_worker().predict_spatial(sample, frame_paths)

        model, tokenizer, torch, device, dtype = self._ensure_loaded()
        frames = load_spatial_frames(
            _uniformly_limit_items(
                frame_paths,
                _bounded_frame_count(self.config.max_frames),
            )
        )
        prompt = build_spatial_text_prompt(
            question=sample.question,
            frame_context=frame_context(frames),
        )
        return self._chat(
            prompt,
            frames,
            model=model,
            tokenizer=tokenizer,
            torch=torch,
            device=device,
            dtype=dtype,
        )

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()
            self._worker = None

    def _should_use_worker(self) -> bool:
        return (
            os.environ.get(INTERNVL_WORKER_ACTIVE_ENV) != "1"
            and _internvl_worker_python() is not None
        )

    def _ensure_worker(self) -> "_InternVLWorkerClient":
        if self._worker is None:
            python = _internvl_worker_python()
            if python is None:
                raise InternVLAdapterError(
                    "InternVL2.5 official inference requires a Transformers 4.37 "
                    "Python environment. Set EVIDENCEQA_INTERNVL_PYTHON to that "
                    "interpreter, for example /root/autodl-tmp/internvl-tf437/bin/python."
                )
            self._worker = _InternVLWorkerClient(self.config, python=python)
        return self._worker

    def _ensure_loaded(self) -> tuple[Any, Any, Any, str, Any]:
        if self._model is not None and self._tokenizer is not None:
            assert self._torch is not None
            assert self._device is not None
            assert self._dtype is not None
            return (
                self._model,
                self._tokenizer,
                self._torch,
                self._device,
                self._dtype,
            )

        try:
            import torch

            with contextlib.redirect_stdout(io.StringIO()):
                from transformers import (
                    AutoModel,
                    AutoTokenizer,
                    PreTrainedModel,
                )
                from transformers.generation.utils import GenerationMixin
        except ImportError as exc:
            raise InternVLAdapterError(
                "InternVL inference requires transformers, torch, torchvision, "
                "pillow, decord, einops, and timm. Install with: "
                'python -m pip install -e ".[vl]"'
            ) from exc

        device = select_device(torch, self.config.device)
        dtype = select_dtype(torch, self.config.dtype, device)
        cache_kwargs = hf_cache_kwargs(self.config.model_cache_dir)
        with (
            _quiet_internvl_remote_code(),
            _patch_missing_tied_weight_keys(PreTrainedModel),
            _disable_custom_generate_loading(GenerationMixin),
        ):
            try:
                model = AutoModel.from_pretrained(
                    self.config.model_id,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    **_flash_attention_kwargs(),
                    **cache_kwargs,
                )
            except TypeError:
                model = AutoModel.from_pretrained(
                    self.config.model_id,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    **cache_kwargs,
                )
        _ensure_tied_weight_keys(model)
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=True,
            use_fast=False,
            **cache_kwargs,
        )
        model.to(device)
        model.eval()

        self._model = model
        self._tokenizer = tokenizer
        self._torch = torch
        self._device = device
        self._dtype = dtype
        return model, tokenizer, torch, device, dtype

    def _chat(
        self,
        prompt: str,
        frames: list[FrameImage],
        *,
        model: Any,
        tokenizer: Any,
        torch: Any,
        device: str,
        dtype: Any,
    ) -> str:
        transform = self._ensure_transform()
        pixel_values = torch.stack([transform(frame.image) for frame in frames])
        pixel_values = pixel_values.to(device=device, dtype=dtype)
        question = _internvl_question(prompt, frames)
        generation_config = _internvl_generation_config(
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.do_sample,
            tokenizer=tokenizer,
            model=model,
        )
        with torch.inference_mode():
            response = model.chat(
                tokenizer,
                pixel_values,
                question,
                generation_config,
                num_patches_list=[1 for _ in frames],
                history=None,
                return_history=False,
            )
        return _ensure_internvl_json_response(
            str(response),
            prompt_mode=self.config.prompt_mode,
            frames=frames,
        )

    def _ensure_transform(self) -> Any:
        if self._transform is not None:
            return self._transform
        try:
            from PIL import Image
            from torchvision import transforms
        except ImportError as exc:
            raise InternVLAdapterError(
                "InternVL frame preprocessing requires pillow and torchvision. "
                'Install with: python -m pip install -e ".[vl]"'
            ) from exc

        self._transform = transforms.Compose(
            [
                transforms.Lambda(lambda image: image.convert("RGB")),
                transforms.Resize(
                    (INTERNVL_IMAGE_SIZE, INTERNVL_IMAGE_SIZE),
                    interpolation=Image.Resampling.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        return self._transform


def _internvl_question(prompt: str, frames: list[FrameImage]) -> str:
    image_lines = "\n".join(
        f"Frame-{index + 1}: <image>" for index, _ in enumerate(frames)
    )
    return f"{image_lines}\n{prompt}"


def _internvl_generation_config(
    *,
    max_new_tokens: int,
    do_sample: bool,
    tokenizer: Any,
    model: Any,
) -> dict[str, Any]:
    generation_config: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        **generation_token_kwargs(tokenizer, model),
    }
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        generation_config["eos_token_id"] = eos_token_id
    min_new_tokens = _min_new_tokens(max_new_tokens)
    if min_new_tokens is not None:
        generation_config["min_new_tokens"] = min_new_tokens
    return generation_config


def _internvl_response_prefix(prompt_mode: str) -> str:
    if prompt_mode == PROMPT_MODE_SPATIAL:
        return INTERNVL_SPATIAL_JSON_PREFIX
    return INTERNVL_JSON_PREFIX


def _ensure_internvl_json_response(
    response: str,
    *,
    prompt_mode: str,
    frames: list[FrameImage],
) -> str:
    """Return the model text without silently manufacturing predictions."""

    return response.strip()


def _min_new_tokens(max_new_tokens: int) -> int | None:
    if max_new_tokens <= 1:
        return None
    return min(INTERNVL_MIN_NEW_TOKENS, max_new_tokens - 1)


def _single_token_id(tokenizer: Any, text: str) -> int | None:
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        token_ids = encode(text, add_special_tokens=False)
    else:
        tokenized = tokenizer(text, add_special_tokens=False)
        token_ids = tokenized["input_ids"]
    if len(token_ids) != 1:
        return None
    return int(token_ids[0])


def _bounded_frame_count(max_frames: int) -> int:
    if _flash_attention_available():
        return max_frames
    if max_frames <= 0:
        return INTERNVL_MAX_EAGER_ATTENTION_FRAMES
    return min(max_frames, INTERNVL_MAX_EAGER_ATTENTION_FRAMES)


def _uniformly_limit_items(items: list[Any], limit: int) -> list[Any]:
    if limit <= 0 or len(items) <= limit:
        return items
    if limit == 1:
        return [items[0]]
    step = (len(items) - 1) / float(limit - 1)
    return [
        items[min(len(items) - 1, round(index * step))]
        for index in range(limit)
    ]


@contextlib.contextmanager
def _patch_missing_tied_weight_keys(model_base_class: Any) -> Any:
    """Compat for InternVL remote code on Transformers releases using this field."""

    original_getattr = getattr(model_base_class, "__getattr__")

    def patched_getattr(self: Any, name: str) -> Any:
        if name == "all_tied_weights_keys":
            value = {}
            object.__setattr__(self, name, value)
            return value
        return original_getattr(self, name)

    model_base_class.__getattr__ = patched_getattr
    try:
        yield
    finally:
        model_base_class.__getattr__ = original_getattr


def _ensure_tied_weight_keys(model: Any) -> None:
    if not hasattr(model, "all_tied_weights_keys"):
        model.all_tied_weights_keys = {}


@contextlib.contextmanager
def _disable_custom_generate_loading(model_base_class: Any) -> Any:
    """Skip Transformers 5 custom_generate downloads; this adapter decodes directly."""

    original = getattr(model_base_class, "load_custom_generate", None)
    if not callable(original):
        yield
        return

    def disabled_load_custom_generate(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise OSError("custom_generate disabled for EvidenceQA InternVL adapter")

    model_base_class.load_custom_generate = disabled_load_custom_generate
    try:
        yield
    finally:
        model_base_class.load_custom_generate = original


@contextlib.contextmanager
def _quiet_internvl_remote_code() -> Any:
    """Silence known InternVL remote-code warnings while loading the model."""

    with (
        warnings.catch_warnings(),
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        warnings.filterwarnings(
            "ignore",
            message=r"Importing from timm\.models\.layers is deprecated.*",
            category=FutureWarning,
        )
        with _quiet_transformers_generation_warning():
            yield


@contextlib.contextmanager
def _quiet_transformers_generation_warning() -> Any:
    try:
        from transformers.utils import logging as transformers_logging
    except ImportError:
        yield
        return

    get_verbosity = getattr(transformers_logging, "get_verbosity", None)
    set_verbosity = getattr(transformers_logging, "set_verbosity", None)
    set_verbosity_error = getattr(transformers_logging, "set_verbosity_error", None)
    previous = get_verbosity() if callable(get_verbosity) else None
    if callable(set_verbosity_error):
        set_verbosity_error()
    try:
        yield
    finally:
        if previous is not None and callable(set_verbosity):
            set_verbosity(previous)


def _flash_attention_kwargs() -> dict[str, bool]:
    if not _flash_attention_available():
        return {"use_flash_attn": False}
    return {"use_flash_attn": True}


def _flash_attention_available() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def _internvl_worker_python() -> str | None:
    configured = os.environ.get(INTERNVL_WORKER_PYTHON_ENV)
    if configured:
        return configured
    try:
        if AUTODL_INTERNVL_PYTHON.exists():
            return str(AUTODL_INTERNVL_PYTHON)
    except OSError:
        return None
    return None


class _InternVLWorkerClient:
    def __init__(self, config: InternVLConfig, *, python: str) -> None:
        self._python = python
        self._process = self._start_process()
        self._request({"op": "init", "config": _config_payload(config)})

    def predict(self, sample: DatasetSample, media_path: Path) -> str:
        response = self._request(
            {
                "op": "predict",
                "sample": asdict(sample),
                "media_path": str(media_path),
            }
        )
        return str(response["result"])

    def predict_spatial(
        self,
        sample: SpatialSample,
        frame_paths: list[tuple[int, Path]],
    ) -> str:
        response = self._request(
            {
                "op": "predict_spatial",
                "sample": asdict(sample),
                "frame_paths": [
                    [int(frame_index), str(path)] for frame_index, path in frame_paths
                ],
            }
        )
        return str(response["result"])

    def close(self) -> None:
        process = self._process
        if process.poll() is None:
            try:
                self._request({"op": "close"})
            except InternVLAdapterError:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()

    def _start_process(self) -> subprocess.Popen[str]:
        env = os.environ.copy()
        env[INTERNVL_WORKER_ACTIVE_ENV] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        src_root = str(Path(__file__).resolve().parents[2])
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            src_root
            if not existing_pythonpath
            else f"{src_root}{os.pathsep}{existing_pythonpath}"
        )
        return subprocess.Popen(
            [
                self._python,
                "-m",
                "evidenceqa_baseline.adapters.internvl_worker",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=env,
        )

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        process = self._process
        if process.stdin is None or process.stdout is None:
            raise InternVLAdapterError("InternVL worker pipes are unavailable")
        if process.poll() is not None:
            raise InternVLAdapterError(
                f"InternVL worker exited with code {process.returncode}"
            )
        process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        process.stdin.flush()
        line = process.stdout.readline()
        if not line:
            raise InternVLAdapterError(
                f"InternVL worker exited with code {process.poll()}"
            )
        response = json.loads(line)
        if not response.get("ok"):
            raise InternVLAdapterError(str(response.get("error", "worker failed")))
        return response

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _config_payload(config: InternVLConfig) -> dict[str, Any]:
    payload = asdict(config)
    if config.model_cache_dir is not None:
        payload["model_cache_dir"] = str(config.model_cache_dir)
    return payload

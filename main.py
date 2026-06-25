"""论文 baseline 研究入口。

常用运行只需要：

```
python -u main.py
python -u main.py --target ref
python -u main.py --models llava,internvl
python -u main.py --dataset-dir /path/to/public_dataset
python -u main.py --output-dir runs/my-baseline-suite
python -u main.py --limit 20 --overwrite --no-resume
```

数据目录、缓存目录、模型列表、硬件参数和抽样策略可以通过命令行或环境变量
覆盖；默认值只用于快速复现 baseline 配置。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))


def _optional_int(raw_value: str | None, default: int | None = None) -> int | None:
    if raw_value is None or raw_value.strip() == "":
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"none", "null"}:
        return None
    return int(normalized)


def _optional_float(raw_value: str | None) -> float | None:
    if raw_value is None or raw_value.strip() == "":
        return None
    normalized = raw_value.strip().lower()
    if normalized in {"none", "null"}:
        return None
    return float(normalized)

DATASET_DIR = Path(os.environ.get("EVIDENCEQA_DATASET_DIR", "public_dataset"))
CACHE_DIR = Path(os.environ.get("EVIDENCEQA_CACHE_DIR", ".cache/evidenceqa-baseline"))

from evidenceqa_baseline.cache import configure_runtime_cache  # noqa: E402

DEFAULT_MODELS = [
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    "OpenGVLab/InternVL2_5-8B",
]
MODEL_ALIASES = {
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2.5": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2.5-vl": "Qwen/Qwen2.5-VL-7B-Instruct",
    "llava": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    "llava-onevision": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    "internvl": "OpenGVLab/InternVL2_5-8B",
    "internvl2.5": "OpenGVLab/InternVL2_5-8B",
}
SEED = int(os.environ.get("EVIDENCEQA_SEED", "20260621"))
SAMPLE_MODE = os.environ.get("EVIDENCEQA_SAMPLE_MODE", "sequential")
LIMIT = _optional_int(os.environ.get("EVIDENCEQA_LIMIT"))
DTYPE = os.environ.get("EVIDENCEQA_DTYPE", "bfloat16")
MAX_FRAMES = int(os.environ.get("EVIDENCEQA_MAX_FRAMES", "64"))
FPS = _optional_float(os.environ.get("EVIDENCEQA_FPS"))
MAX_PIXELS = _optional_int(os.environ.get("EVIDENCEQA_MAX_PIXELS"), 768 * 28 * 28)
MAX_NEW_TOKENS = int(os.environ.get("EVIDENCEQA_MAX_NEW_TOKENS", "256"))
MEDIA_SYNC = os.environ.get("EVIDENCEQA_MEDIA_SYNC", "eager")
HARDWARE_PROFILE = os.environ.get(
    "EVIDENCEQA_HARDWARE_PROFILE",
    "rtx-pro-6000-96gb-single-cuda",
)


def main(argv: list[str] | None = None) -> int:
    from evidenceqa_baseline.runner import RunConfig
    from evidenceqa_baseline.suite import TARGETS, TARGET_SUITE, run_suite

    parser = argparse.ArgumentParser(
        description="Run the fixed EvidenceQA paper baseline suite."
    )
    parser.add_argument("--target", choices=TARGETS, default=TARGET_SUITE)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=LIMIT,
        help="最大样本数；0 或省略表示跑完整自动选择的 split。",
    )
    parser.add_argument(
        "--models",
        "--model",
        action="append",
        default=None,
        help=(
            "只运行指定模型；支持 qwen、llava、internvl 别名，"
            "也支持完整 HF model id。多个模型可用逗号分隔或重复传参。"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--sample-mode",
        choices=["random", "sequential"],
        default=SAMPLE_MODE,
    )
    parser.add_argument("--dtype", default=DTYPE)
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--fps", type=float, default=FPS)
    parser.add_argument("--max-pixels", type=int, default=MAX_PIXELS)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--hardware-profile", default=HARDWARE_PROFILE)
    parser.add_argument("--media-sync", choices=["eager", "lazy"], default=MEDIA_SYNC)
    parser.add_argument(
        "--resume",
        dest="resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative; use --limit 0 for the full split")
    if args.max_frames < 0:
        parser.error("--max-frames must be non-negative")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.max_pixels is not None and args.max_pixels < 0:
        parser.error("--max-pixels must be non-negative")
    limit = None if args.limit in (None, 0) else args.limit
    selected_models = _select_models(args.models)
    dataset_dir = args.dataset_dir.expanduser()
    cache_dir = args.cache_dir.expanduser()
    configure_runtime_cache(cache_dir)

    config = RunConfig(
        limit=limit,
        seed=args.seed,
        sample_mode=args.sample_mode,
        output_dir=args.output_dir,
        cache_dir=cache_dir,
        dry_run=args.dry_run,
        resume=args.resume,
        overwrite=args.overwrite,
        dtype=args.dtype,
        max_frames=args.max_frames,
        fps=args.fps,
        max_pixels=None if args.max_pixels == 0 else args.max_pixels,
        max_new_tokens=args.max_new_tokens,
        media_sync=args.media_sync,
        progress=args.progress,
        hardware_profile=args.hardware_profile,
    )
    result = run_suite(
        config,
        target=args.target,
        dataset_dir=dataset_dir,
        models=selected_models,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "suite_summary": str(result.summary_path),
                "summary_payload": result.summary,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 3 if result.summary.get("status") == "smoke_failed" else 0


def _select_models(raw_values: list[str] | None) -> list[str]:
    if not raw_values:
        env_models = os.environ.get("EVIDENCEQA_MODELS")
        raw_values = [env_models] if env_models else list(DEFAULT_MODELS)

    selected: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        for item in raw_value.split(","):
            model = _resolve_model_name(item)
            if model and model not in seen:
                selected.append(model)
                seen.add(model)
    if not selected:
        raise SystemExit("--models must include at least one model")
    return selected


def _resolve_model_name(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return ""
    return MODEL_ALIASES.get(cleaned.lower(), cleaned)


if __name__ == "__main__":
    raise SystemExit(main())

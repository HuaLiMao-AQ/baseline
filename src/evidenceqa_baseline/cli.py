"""独立 baseline 的命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .dataset import (
    DEFAULT_LIMIT,
    DEFAULT_REPO_ID,
    DEFAULT_REVISION,
    DEFAULT_SAMPLE_MODE,
    DEFAULT_SEED,
    DEFAULT_TASK_TYPE,
    DatasetError,
)
from .runner import DEFAULT_MAX_NEW_TOKENS, DEFAULT_QWEN_VL_MODEL_ID, RunConfig
from .suite import DEFAULT_DATASET_DIR, TARGETS, TARGET_SUITE, run_suite


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。

    Returns:
        配置好所有 baseline 参数的 ``ArgumentParser``。
    """

    parser = argparse.ArgumentParser(
        prog="evidenceqa-baseline",
        description="Run a minimal Video-LMM baseline on EvidenceQA Core temporal QA.",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--task-type", default=DEFAULT_TASK_TYPE)
    parser.add_argument("--model", default=DEFAULT_QWEN_VL_MODEL_ID)
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="最大样本数；0 表示跑完整自动选择的 split。",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--sample-mode",
        choices=["random", "sequential"],
        default=DEFAULT_SAMPLE_MODE,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help=(
            "本地 EvidenceQA 数据集目录；默认读取 EVIDENCEQA_DATASET_DIR，"
            f"或存在时使用 {DEFAULT_DATASET_DIR}。"
        ),
    )
    parser.add_argument(
        "--target",
        choices=TARGETS,
        default=TARGET_SUITE,
        help=(
            "suite 自动跑 validation；train 自动跑 train；其他值只跑 "
            "validation 上的指定阶段。"
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="显式启用 smoke-gated suite；等价于 --target suite。",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/evidenceqa-baseline"),
    )
    parser.add_argument("--local-jsonl", type=Path, default=None)
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="干净运行日志路径；默认写入 output_dir/run.log。",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        dest="resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max-pixels", type=int, default=768 * 28 * 28)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--hardware-profile",
        default="rtx-pro-6000-96gb-single-cuda",
        help="写入结果记录的硬件画像标签。",
    )
    parser.add_argument(
        "--media-sync",
        choices=["eager", "lazy"],
        default="eager",
        help="eager 会在模型推理前解析/下载本次样本视频；lazy 按样本边跑边下载。",
    )
    parser.add_argument(
        "--progress",
        dest="progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否使用 Rich 在 stderr 显示进度。",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """执行命令行入口。

    Args:
        argv: 可选命令行参数列表；为 ``None`` 时读取进程参数。

    Returns:
        进程退出码。
    """

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error(
            "--limit must be non-negative; use --limit 0 for the full selected split"
        )
    if args.smoke and args.target != TARGET_SUITE:
        parser.error("--smoke cannot be combined with --target other than suite")
    limit = None if args.limit == 0 else args.limit
    config = RunConfig(
        repo_id=args.repo_id,
        revision=args.revision,
        task_type=args.task_type,
        model=args.model,
        limit=limit,
        seed=args.seed,
        sample_mode=args.sample_mode,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir.expanduser(),
        dry_run=args.dry_run,
        resume=args.resume,
        overwrite=args.overwrite,
        dtype=args.dtype,
        max_frames=args.max_frames,
        fps=args.fps,
        max_pixels=args.max_pixels,
        max_new_tokens=args.max_new_tokens,
        local_jsonl=args.local_jsonl,
        progress=args.progress,
        log_file=args.log_file,
        media_sync=args.media_sync,
        hardware_profile=args.hardware_profile,
    )
    try:
        result = run_suite(
            config,
            target=args.target,
            dataset_dir=args.dataset_dir.expanduser() if args.dataset_dir else None,
        )
    except (DatasetError, FileNotFoundError, ValueError) as exc:
        print(f"dataset error: {exc}", file=sys.stderr)
        return 2
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
if __name__ == "__main__":
    raise SystemExit(main())

"""命令行入口。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from . import __version__
from .artifact import validate_artifact
from .dataset import (
    DEFAULT_LIMIT,
    DEFAULT_REPO_ID,
    DEFAULT_REVISION,
    DEFAULT_SAMPLE_MODE,
    DEFAULT_SEED,
    DEFAULT_TASK_TYPE,
)
from .report import write_analysis_report
from .runner import DEFAULT_MAX_NEW_TOKENS, RunConfig
from .suite import DEFAULT_DATASET_DIR, TARGETS, TARGET_SUITE, run_suite
from .tables import export_metric_tables
from .taxonomy import export_grounded_taxonomy

MODEL_ALIASES = {
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2.5": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2.5-vl": "Qwen/Qwen2.5-VL-7B-Instruct",
    "llava": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    "llava-onevision": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    "internvl": "OpenGVLab/InternVL2_5-8B",
    "internvl2.5": "OpenGVLab/InternVL2_5-8B",
}


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI parser。"""

    parser = argparse.ArgumentParser(
        prog="evidenceqa-baseline",
        description="EvidenceQA baseline 的baseline 入口。",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("version", help="显示项目版本")

    _add_run_parser(subparsers)

    validate = subparsers.add_parser("validate-artifact", help="校验 baseline 结果目录")
    validate.add_argument("root", type=Path)
    validate.add_argument("--json", action="store_true", help="以 JSON 输出校验结果")

    export = subparsers.add_parser("export-tables", help="导出 baseline 主指标 CSV")
    export.add_argument("root", type=Path)
    export.add_argument("output_dir", type=Path)

    taxonomy = subparsers.add_parser(
        "export-taxonomy",
        help="导出 grounded 阶段 A/E 错误类型 CSV",
    )
    taxonomy.add_argument("root", type=Path)
    taxonomy.add_argument("output_path", type=Path)

    analyze = subparsers.add_parser("analyze-artifact", help="生成 baseline 分析报告")
    analyze.add_argument("root", type=Path)
    analyze.add_argument("output_dir", type=Path)
    return parser


def _add_run_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    run = subparsers.add_parser("run", help="运行 baseline suite 或单个阶段")
    run.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    run.add_argument("--revision", default=DEFAULT_REVISION)
    run.add_argument("--task-type", default=DEFAULT_TASK_TYPE)
    run.add_argument("--target", choices=TARGETS, default=TARGET_SUITE)
    run.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="本地 EvidenceQA 数据集目录。",
    )
    run.add_argument(
        "--models",
        "--model",
        action="append",
        default=None,
        help="模型别名或完整模型 ID；多个模型可用逗号分隔，也可重复传参。",
    )
    run.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="最大样本数；0 表示跑完整 split。",
    )
    run.add_argument("--seed", type=int, default=DEFAULT_SEED)
    run.add_argument(
        "--sample-mode",
        choices=["random", "sequential"],
        default=DEFAULT_SAMPLE_MODE,
    )
    run.add_argument("--output-dir", type=Path, default=None)
    run.add_argument("--cache-dir", type=Path, default=Path(".cache/evidenceqa-baseline"))
    run.add_argument("--local-jsonl", type=Path, default=None)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--overwrite", action="store_true")
    run.add_argument(
        "--resume",
        dest="resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    run.add_argument("--dtype", default="bfloat16")
    run.add_argument("--max-frames", type=int, default=64)
    run.add_argument("--fps", type=float, default=None)
    run.add_argument("--max-pixels", type=int, default=768 * 28 * 28)
    run.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    run.add_argument("--media-sync", choices=["eager", "lazy"], default="eager")
    run.add_argument(
        "--progress",
        dest="progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """执行 CLI。"""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "version":
        print(__version__)
        return 0
    if args.command == "run":
        return _run_command(args, parser)
    if args.command == "validate-artifact":
        issues = validate_artifact(args.root)
        if args.json:
            print(
                json.dumps(
                    [asdict(issue) for issue in issues],
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            if not issues:
                print("artifact 校验通过")
            for issue in issues:
                print(f"{issue.severity}: {issue.path}: {issue.message}")
        return 1 if any(issue.severity == "error" for issue in issues) else 0
    if args.command == "export-tables":
        paths = export_metric_tables(args.root, args.output_dir)
        for path in paths:
            print(path)
        return 0
    if args.command == "export-taxonomy":
        print(export_grounded_taxonomy(args.root, args.output_path))
        return 0
    if args.command == "analyze-artifact":
        print(write_analysis_report(args.root, args.output_dir))
        return 0
    parser.error(f"未知命令: {args.command}")
    return 2


def _run_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.limit < 0:
        parser.error("--limit 不能为负数；使用 --limit 0 跑完整 split")
    limit = None if args.limit == 0 else args.limit
    config = RunConfig(
        repo_id=args.repo_id,
        revision=args.revision,
        task_type=args.task_type,
        limit=limit,
        seed=args.seed,
        sample_mode=args.sample_mode,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
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
        media_sync=args.media_sync,
    )
    result = run_suite(
        config,
        target=args.target,
        dataset_dir=args.dataset_dir,
        models=_select_models(args.models),
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


def _select_models(raw_values: list[str] | None) -> list[str] | None:
    if raw_values is None:
        return None
    selected: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        for item in raw_value.split(","):
            model = _resolve_model_name(item)
            if model and model not in seen:
                selected.append(model)
                seen.add(model)
    if not selected:
        raise SystemExit("--models 至少需要包含一个模型")
    return selected


def _resolve_model_name(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return ""
    return MODEL_ALIASES.get(cleaned.lower(), cleaned)


if __name__ == "__main__":
    raise SystemExit(main())

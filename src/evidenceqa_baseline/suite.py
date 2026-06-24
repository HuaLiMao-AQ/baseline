"""本地基线实验编排。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from . import runner
from . import spatial_runner
from .prompting import PROMPT_MODE_ANSWER_ONLY, PROMPT_MODE_GROUNDED, PROMPT_MODE_SPATIAL
from .runner import RunConfig, RunResult

DEFAULT_DATASET_DIR = Path(
    os.environ.get("EVIDENCEQA_DATASET_DIR", "/root/autodl-tmp/public_dataset")
)
SMOKE_LIMIT = 3
TARGET_SUITE = "suite"
TARGET_SMOKE = "smoke"
TARGET_ANSWER_ONLY = "answer_only"
TARGET_GROUNDED = "grounded"
TARGET_TRAIN = "train"
TARGET_REF = "ref"
TARGET_SPATIAL = "spatial"
TARGETS = (
    TARGET_SUITE,
    TARGET_SMOKE,
    TARGET_ANSWER_ONLY,
    TARGET_GROUNDED,
    TARGET_TRAIN,
    TARGET_REF,
    TARGET_SPATIAL,
)

Target = Literal[
    "suite",
    "smoke",
    "answer_only",
    "grounded",
    "train",
    "ref",
    "spatial",
]


@dataclass(frozen=True, slots=True)
class SuiteResult:
    """一次基线 suite 的输出。"""

    output_dir: Path
    summary_path: Path
    stage_results: dict[str, RunResult]
    summary: dict[str, Any]


def run_suite(
    config: RunConfig,
    *,
    target: Target = TARGET_SUITE,
    dataset_dir: Path | None = None,
    models: list[str] | None = None,
) -> SuiteResult:
    """运行带 smoke gate 的基线 suite。

    ``target=suite`` 或 ``target=train`` 时先跑 smoke；smoke 的推理、解析和导出
    都通过后，继续跑 answer-only、grounded 与 ref 三个正式基线。
    其他 target 只运行 validation split 上的指定阶段。
    """

    if target not in TARGETS:
        raise ValueError(f"target 必须是 {TARGETS!r} 之一")

    resolved_dataset_dir = _effective_dataset_dir(dataset_dir)
    model_ids = _normalize_models(config.model, models)
    output_dir = config.output_dir or _default_suite_output_dir(
        config,
        target,
        model_count=len(model_ids),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "suite_summary.json"
    stage_results: dict[str, RunResult] = {}
    model_runs: dict[str, dict[str, Any]] = {}
    overall_status = "complete"

    for model_id in model_ids:
        model_slug = _model_slug(model_id)
        model_output_dir = output_dir if len(model_ids) == 1 else output_dir / model_slug
        model_config = replace(config, model=model_id, output_dir=model_output_dir)
        stages: dict[str, dict[str, Any]] = {}
        model_status = "complete"

        for stage in _stages_for_target(target):
            stage_split = _split_for_target(target)
            stage_local_jsonl = _stage_local_jsonl(
                model_config,
                resolved_dataset_dir,
                split=stage_split,
            )
            stage_config = _stage_config(
                model_config,
                stage=stage,
                split=stage_split,
                output_dir=model_output_dir,
                local_jsonl=stage_local_jsonl,
            )
            result = _run_stage(stage, stage_config)
            result_key = stage if len(model_ids) == 1 else f"{model_slug}/{stage}"
            stage_results[result_key] = result
            stages[stage] = _stage_summary(stage, stage_config, result)

            if stage == TARGET_SMOKE and target in {TARGET_SUITE, TARGET_TRAIN}:
                smoke = _smoke_gate(result.summary)
                stages[stage]["smoke_gate"] = smoke
                if not smoke["passed"]:
                    model_status = "smoke_failed"
                    overall_status = "smoke_failed"
                    break

        model_runs[model_slug] = {
            "model": model_id,
            "status": model_status,
            "output_dir": str(model_output_dir),
            "stages": stages,
        }

    summary = _suite_summary(
        config=config,
        target=target,
        status=overall_status,
        output_dir=output_dir,
        dataset_dir=resolved_dataset_dir,
        model_runs=model_runs,
    )
    _write_json(summary_path, summary)
    return SuiteResult(
        output_dir=output_dir,
        summary_path=summary_path,
        stage_results=stage_results,
        summary=summary,
    )


def dataset_jsonl_for_split(dataset_dir: Path, split: str) -> Path:
    """返回本地数据集目录中的 split JSONL。"""

    candidates = (
        dataset_dir / f"{split}.jsonl",
        dataset_dir / "splits" / f"{split}.jsonl",
        dataset_dir / "data" / f"{split}.jsonl",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"在 dataset_dir={dataset_dir} 下找不到 {split}.jsonl"
    )


def _effective_dataset_dir(dataset_dir: Path | None) -> Path | None:
    if dataset_dir is not None:
        return dataset_dir
    if _path_exists(DEFAULT_DATASET_DIR):
        return DEFAULT_DATASET_DIR
    return None


def _stage_local_jsonl(
    config: RunConfig,
    dataset_dir: Path | None,
    *,
    split: str,
) -> Path | None:
    if config.local_jsonl is not None:
        return config.local_jsonl
    if dataset_dir is None:
        return None
    return dataset_jsonl_for_split(dataset_dir, split)


def _stages_for_target(target: Target) -> list[str]:
    if target in {TARGET_SUITE, TARGET_TRAIN}:
        return [TARGET_SMOKE, TARGET_ANSWER_ONLY, TARGET_GROUNDED, TARGET_REF]
    if target == TARGET_SPATIAL:
        return [TARGET_REF]
    return [target]


def _split_for_target(target: Target) -> str:
    if target == TARGET_TRAIN:
        return "train"
    return "validation"


def _stage_config(
    config: RunConfig,
    *,
    stage: str,
    split: str,
    output_dir: Path,
    local_jsonl: Path | None,
) -> RunConfig:
    stage_dir = output_dir / stage
    prompt_mode = (
        PROMPT_MODE_SPATIAL
        if stage == TARGET_REF
        else
        PROMPT_MODE_ANSWER_ONLY
        if stage == TARGET_ANSWER_ONLY
        else PROMPT_MODE_GROUNDED
    )
    return replace(
        config,
        split=split,
        output_dir=stage_dir,
        log_file=stage_dir / "run.log",
        local_jsonl=local_jsonl,
        limit=SMOKE_LIMIT if stage == TARGET_SMOKE else config.limit,
        sample_mode="sequential" if stage == TARGET_SMOKE else config.sample_mode,
        prompt_mode=prompt_mode,
        experiment_stage=stage,
        max_new_tokens=64 if stage == TARGET_ANSWER_ONLY else config.max_new_tokens,
    )


def _run_stage(stage: str, config: RunConfig) -> RunResult:
    if stage == TARGET_REF:
        return spatial_runner.run_spatial_baseline(config)
    return runner.run_baseline(config)


def _smoke_gate(summary: dict[str, Any]) -> dict[str, Any]:
    selected = int(summary.get("selected_samples") or 0)
    if summary.get("dry_run") is True:
        return {
            "passed": selected > 0,
            "reasons": [] if selected > 0 else ["没有选中样本"],
            "selected_samples": selected,
            "inference_success_count": None,
            "parse_success_count": None,
            "incomplete_records": None,
            "dry_run": True,
        }
    inference_success = int(summary.get("inference_success_count") or 0)
    parse_success = int(summary.get("parse_success_count") or 0)
    export = summary.get("export_completeness") or {}
    incomplete = int(export.get("incomplete_records") or 0)
    reasons: list[str] = []
    if selected <= 0:
        reasons.append("没有选中样本")
    if inference_success != selected:
        reasons.append("不是所有样本都完成推理")
    if parse_success != selected:
        reasons.append("不是所有样本都解析成功")
    if incomplete:
        reasons.append("导出记录不完整")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "selected_samples": selected,
        "inference_success_count": inference_success,
        "parse_success_count": parse_success,
        "incomplete_records": incomplete,
    }


def _stage_summary(
    stage: str,
    config: RunConfig,
    result: RunResult,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "prompt_mode": config.prompt_mode,
        "split": config.split,
        "limit": "all" if config.limit is None else config.limit,
        "sample_mode": config.sample_mode,
        "seed": config.seed,
        "output_dir": str(result.output_dir),
        "run_config": str(result.run_config_path),
        "predictions": str(result.predictions_path),
        "failed_samples": str(result.failed_samples_path),
        "log": str(result.log_path),
        "summary": str(result.summary_path),
        "summary_payload": result.summary,
        "paper_metrics": result.summary.get("paper_metrics", {}),
    }


def _suite_summary(
    *,
    config: RunConfig,
    target: Target,
    status: str,
    output_dir: Path,
    dataset_dir: Path | None,
    model_runs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    all_stages = [
        stage
        for model_run in model_runs.values()
        for stage in (model_run.get("stages") or {}).values()
    ]
    splits = sorted({str(stage.get("split")) for stage in all_stages})
    paper_metrics_table = [
        {
            "model": model_run.get("model"),
            "stage": stage,
            "prompt_mode": payload.get("prompt_mode"),
            **(payload.get("paper_metrics") or {}),
        }
        for model_run in model_runs.values()
        for stage, payload in (model_run.get("stages") or {}).items()
    ]
    models = [str(model_run.get("model")) for model_run in model_runs.values()]
    payload: dict[str, Any] = {
        "status": status,
        "target": target,
        "output_dir": str(output_dir),
        "dataset_dir": str(dataset_dir) if dataset_dir else None,
        "splits": splits,
        "model": models[0] if len(models) == 1 else None,
        "models": models,
        "hardware_profile": config.hardware_profile,
        "max_frames": config.max_frames,
        "max_pixels": config.max_pixels,
        "dtype": config.dtype,
        "paper_metrics_table": paper_metrics_table,
        "model_runs": model_runs,
    }
    if len(model_runs) == 1:
        only_run = next(iter(model_runs.values()))
        payload["stages"] = only_run.get("stages") or {}
    return payload


def _default_suite_output_dir(
    config: RunConfig,
    target: Target,
    *,
    model_count: int = 1,
) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    model_slug = "multi-model" if model_count > 1 else _model_slug(config.model)
    limit = "all" if config.limit is None else str(config.limit)
    split = _split_for_target(target)
    return Path("runs") / f"{timestamp}-{model_slug}-{split}-{target}-{limit}"


def _normalize_models(default_model: str, models: list[str] | None) -> list[str]:
    raw = models if models else [default_model]
    result: list[str] = []
    seen: set[str] = set()
    for model in raw:
        cleaned = model.strip()
        if not cleaned or cleaned in seen:
            continue
        result.append(cleaned)
        seen.add(cleaned)
    if not result:
        raise ValueError("至少需要配置一个模型")
    return result


def _model_slug(model_id: str) -> str:
    return (
        model_id.rsplit("/", 1)[-1]
        .replace("_", "-")
        .replace(" ", "-")
        .lower()
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False

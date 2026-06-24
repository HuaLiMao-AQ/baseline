"""Spatial grounding baseline 执行器。"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

from .adapters.base import ModelAdapter
from .cache import configure_runtime_cache
from .dataset import (
    SPATIAL_TASK_TYPE,
    FrameRef,
    SpatialDatasetLoadResult,
    SpatialSample,
    filter_spatial_grounding,
    load_spatial_samples,
    select_samples,
)
from .media import resolve_or_download_media
from .metrics import summarize_spatial_predictions
from .parser import parse_spatial_model_output
from .prompting import PROMPT_MODE_SPATIAL
from .progress import iter_with_progress, rich_status
from .runner import (
    PROGRESS_LOG_EVERY,
    RunConfig,
    RunResult,
    _PredictionStats,
    _append_jsonl,
    _build_adapter,
    _cuda_peak_memory_bytes,
    _default_output_dir,
    _format_media_sync_log,
    _jsonable_config,
    _log_media_sync_progress,
    _log_prediction_progress,
    _log_stage_start,
    _media_sync_summary,
    _reset_cuda_peak_memory,
    _short_error,
    _write_json,
    collect_runtime_info,
    read_completed_ids,
    read_prediction_rows,
)
from .log_utils import configure_run_logging

FrameResolver = Callable[[FrameRef, RunConfig], Path]


@dataclass(frozen=True, slots=True)
class SyncedFrames:
    """Pre-resolved frame paths for a selected spatial sample."""

    frame_paths: list[tuple[int, Path]] | None
    error: Exception | None = None


def run_spatial_baseline(
    config: RunConfig,
    *,
    samples: list[SpatialSample] | None = None,
    frame_resolver: FrameResolver | None = None,
) -> RunResult:
    """执行 Ref-YouTube-VOS spatial grounding baseline。"""

    output_dir = config.output_dir or _default_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    failed_samples_path = output_dir / "failed_samples.jsonl"
    run_config_path = output_dir / "run_config.json"
    summary_path = output_dir / "summary.json"
    log_path = config.log_file or output_dir / "run.log"
    logger = configure_run_logging(log_path, overwrite=config.overwrite)
    logger.info(
        "run_start repo_id=%s revision=%s split=%s limit=%s model=%s "
        "media_sync=%s prompt_mode=%s experiment_stage=%s output_dir=%s",
        config.repo_id,
        config.revision,
        config.split,
        "all" if config.limit is None else config.limit,
        config.model,
        config.media_sync,
        PROMPT_MODE_SPATIAL,
        config.experiment_stage,
        output_dir,
    )
    runtime_cache = configure_runtime_cache(config.cache_dir)
    logger.info(
        "runtime_cache_configured cache_dir=%s hf_cache=%s tmpdir=%s",
        config.cache_dir,
        runtime_cache.get("HF_HUB_CACHE"),
        runtime_cache.get("TMPDIR"),
    )

    if config.overwrite:
        for path in (predictions_path, failed_samples_path, summary_path):
            path.unlink(missing_ok=True)

    _log_stage_start(logger, "load_dataset")
    with rich_status(config.progress, "读取并筛选 spatial grounding split..."):
        dataset_result = _load_spatial_samples(config, samples=samples)
    selected = dataset_result.selected_samples
    logger.info(
        "stage_complete name=load_dataset split_path=%s total_rows=%d spatial_rows=%d "
        "selected_samples=%d",
        dataset_result.split_path,
        dataset_result.total_rows,
        dataset_result.spatial_rows,
        len(selected),
    )

    _log_stage_start(logger, "collect_runtime")
    with rich_status(config.progress, "收集运行环境信息..."):
        runtime = collect_runtime_info()
    logger.info(
        "stage_complete name=collect_runtime cuda_available=%s device_name=%s",
        runtime.get("cuda", {}).get("available"),
        runtime.get("cuda", {}).get("device_name"),
    )
    run_config_payload = {
        **_jsonable_config(config, output_dir),
        "task_type": SPATIAL_TASK_TYPE,
        "prompt_mode": PROMPT_MODE_SPATIAL,
        "runtime": runtime,
        "runtime_cache": runtime_cache,
        "platform": config.platform,
        "hardware_profile": config.hardware_profile,
        "log_file": str(log_path),
        "split_path": str(dataset_result.split_path),
        "total_rows": dataset_result.total_rows,
        "spatial_rows": dataset_result.spatial_rows,
        "selected_sample_ids": [sample.id for sample in selected],
        "selected_samples": [_sample_preview(sample) for sample in selected],
    }
    _write_json(run_config_path, run_config_payload)

    if config.dry_run:
        summary = {
            "dry_run": True,
            "total_rows": dataset_result.total_rows,
            "spatial_rows": dataset_result.spatial_rows,
            "selected_samples": len(selected),
            "sample_preview": [_sample_preview(sample) for sample in selected[:5]],
        }
        _write_json(summary_path, summary)
        logger.info(
            "run_complete dry_run=true total_rows=%d spatial_rows=%d "
            "selected_samples=%d",
            dataset_result.total_rows,
            dataset_result.spatial_rows,
            len(selected),
        )
        return RunResult(
            output_dir=output_dir,
            run_config_path=run_config_path,
            predictions_path=predictions_path,
            failed_samples_path=failed_samples_path,
            summary_path=summary_path,
            log_path=log_path,
            summary=summary,
        )

    resolver = frame_resolver or _default_frame_resolver
    completed_ids = read_completed_ids(predictions_path) if config.resume else set()
    pending_samples = [sample for sample in selected if sample.id not in completed_ids]
    logger.info(
        "resume_state completed_samples=%d pending_samples=%d",
        len(completed_ids),
        len(pending_samples),
    )

    synced_frames: dict[str, SyncedFrames] = {}
    if config.media_sync == "eager":
        logger.info(
            "stage_start name=media_sync pending_samples=%d",
            len(pending_samples),
        )
        with rich_status(config.progress, "同步本次样本帧..."):
            synced_frames = _sync_selected_frames(
                pending_samples,
                config=config,
                frame_resolver=resolver,
                logger=logger,
            )
        logger.info(
            "stage_complete name=media_sync %s",
            _format_media_sync_log(synced_frames),  # type: ignore[arg-type]
        )
    elif config.media_sync != "lazy":
        raise ValueError("media_sync must be 'eager' or 'lazy'")

    model_adapter: ModelAdapter | None = None

    def get_model_adapter() -> ModelAdapter:
        nonlocal model_adapter
        if model_adapter is None:
            logger.info(
                "model_load_start model=%s device=%s dtype=%s",
                config.model,
                config.device,
                config.dtype,
            )
            model_adapter = _build_adapter(replace(config, prompt_mode=PROMPT_MODE_SPATIAL))
            _reset_cuda_peak_memory()
            logger.info("model_load_complete model=%s", config.model)
        return model_adapter

    logger.info(
        "stage_start name=prediction selected_samples=%d pending_samples=%d "
        "completed_samples=%d",
        len(selected),
        len(pending_samples),
        len(completed_ids),
    )
    prediction_stats = _PredictionStats()
    for sample in iter_with_progress(
        pending_samples,
        total=len(pending_samples),
        enabled=config.progress,
        description="运行 Ref grounding baseline",
        item_label=lambda item: item.id,
    ):
        record = _run_one_spatial_sample(
            sample=sample,
            config=config,
            model_getter=get_model_adapter,
            frame_resolver=resolver,
            synced_frames=synced_frames.get(sample.id),
        )
        _append_jsonl(predictions_path, record)
        prediction_stats.record(record)
        if record.get("error") or not record.get("parse_success"):
            synced = synced_frames.get(sample.id)
            if synced is None or synced.error is None:
                logger.warning(
                    "sample_failed id=%s error=%s parse_error=%s",
                    sample.id,
                    _short_error(record.get("error")),
                    _short_error(record.get("parse_error")),
                )
            _append_jsonl(failed_samples_path, record)
        _log_prediction_progress(
            logger,
            stats=prediction_stats,
            total=len(pending_samples),
            current_id=sample.id,
            every=PROGRESS_LOG_EVERY,
        )
    logger.info(
        "stage_complete name=prediction attempted=%d success=%d failed=%d "
        "parse_failed=%d",
        prediction_stats.processed,
        prediction_stats.success,
        prediction_stats.failed,
        prediction_stats.parse_failed,
    )

    selected_ids = {sample.id for sample in selected}
    all_records = [
        record
        for record in read_prediction_rows(predictions_path)
        if record.get("id") in selected_ids
    ]
    summary = summarize_spatial_predictions(
        all_records,
        cuda_peak_memory_bytes=_cuda_peak_memory_bytes(),
    )
    summary["dry_run"] = False
    summary["prompt_mode"] = PROMPT_MODE_SPATIAL
    summary["experiment_stage"] = config.experiment_stage
    summary["selected_samples"] = len(selected)
    summary["media_sync"] = _media_sync_summary(
        config.media_sync,
        synced_frames,  # type: ignore[arg-type]
    )
    summary["export_completeness"] = _export_completeness_summary(all_records)
    summary["paper_metrics"] = _paper_metrics(summary)
    _write_json(summary_path, summary)
    logger.info(
        "run_complete selected_samples=%d valid_predictions=%s parse_failures=%s "
        "failed_samples_path=%s summary_path=%s",
        len(selected),
        summary.get("valid_prediction_count"),
        summary.get("parse_failure_count"),
        failed_samples_path,
        summary_path,
    )

    return RunResult(
        output_dir=output_dir,
        run_config_path=run_config_path,
        predictions_path=predictions_path,
        failed_samples_path=failed_samples_path,
        summary_path=summary_path,
        log_path=log_path,
        summary=summary,
    )


def _load_spatial_samples(
    config: RunConfig,
    *,
    samples: list[SpatialSample] | None,
) -> SpatialDatasetLoadResult:
    if samples is None:
        return load_spatial_samples(
            repo_id=config.repo_id,
            revision=config.revision,
            split=config.split,
            task_type=SPATIAL_TASK_TYPE,
            limit=config.limit,
            seed=config.seed,
            sample_mode=config.sample_mode,
            cache_dir=config.cache_dir,
            local_jsonl=config.local_jsonl,
        )

    rows = [sample.raw for sample in samples]
    spatial_samples = filter_spatial_grounding(rows, task_type=SPATIAL_TASK_TYPE)
    selected = select_samples(
        spatial_samples,
        limit=config.limit,
        seed=config.seed,
        sample_mode=config.sample_mode,
    )
    return SpatialDatasetLoadResult(
        split_path=config.local_jsonl or Path("<in-memory>"),
        total_rows=len(rows),
        spatial_rows=len(spatial_samples),
        selected_samples=selected,
    )


def _run_one_spatial_sample(
    *,
    sample: SpatialSample,
    config: RunConfig,
    model_getter: Callable[[], ModelAdapter],
    frame_resolver: FrameResolver,
    synced_frames: SyncedFrames | None = None,
) -> dict[str, Any]:
    frame_refs = _selected_frames(sample, max_frames=config.max_frames)
    base_record: dict[str, Any] = {
        "id": sample.id,
        "sample_id": sample.id,
        "video_id": sample.video_id,
        "source_dataset": sample.source_dataset,
        "source_split": sample.source_split,
        "task_type": SPATIAL_TASK_TYPE,
        "question": sample.question,
        "target_ref": sample.target_ref,
        "frame_refs": [asdict(frame) for frame in frame_refs],
        "resolved_frame_paths": [],
        "gt_box_track": [asdict(item) for item in sample.gt_box_track],
        "gt_point_track": [asdict(item) for item in sample.gt_point_track],
        "reference_mask_path": sample.reference_mask_path,
        "hf_dataset_repo": config.repo_id,
        "hf_revision": config.revision,
        "hf_split": config.split,
        "model": config.model,
        "prompt_mode": PROMPT_MODE_SPATIAL,
        "experiment_stage": config.experiment_stage,
        "platform": config.platform,
        "hardware_profile": config.hardware_profile,
        "source_record": sample.raw,
        "pred_target": None,
        "pred_frame_index": None,
        "pred_point": None,
        "pred_box": None,
        "parse_success": False,
        "parse_error": None,
        "was_repaired": False,
        "raw_output": "",
        "error": None,
        "latency_seconds": 0.0,
    }

    started = time.perf_counter()
    try:
        if synced_frames is not None:
            if synced_frames.error is not None:
                raise synced_frames.error
            if synced_frames.frame_paths is None:
                raise RuntimeError("media sync did not produce local frames")
            frame_paths = synced_frames.frame_paths
        else:
            frame_paths = [
                (frame.frame_index, frame_resolver(frame, config))
                for frame in frame_refs
            ]
        base_record["resolved_frame_paths"] = [
            {"frame_index": frame_index, "path": str(path)}
            for frame_index, path in frame_paths
        ]
        raw_output = model_getter().predict_spatial(sample, frame_paths)
        latency = time.perf_counter() - started
        parsed = parse_spatial_model_output(
            raw_output,
            valid_frame_indices={frame_index for frame_index, _ in frame_paths},
        )
        base_record.update(
            {
                "pred_target": parsed.target,
                "pred_frame_index": parsed.frame_index,
                "pred_point": parsed.point,
                "pred_box": parsed.box,
                "parse_success": parsed.parse_success,
                "parse_error": parsed.parse_error,
                "was_repaired": parsed.was_repaired,
                "raw_output": parsed.raw_output,
                "latency_seconds": latency,
            }
        )
        _attach_export_completeness(base_record)
        return base_record
    except Exception as exc:  # noqa: BLE001 - 样本级隔离。
        latency = time.perf_counter() - started
        base_record.update(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "latency_seconds": latency,
            }
        )
        _attach_export_completeness(base_record)
        return base_record


def _sync_selected_frames(
    samples: list[SpatialSample],
    *,
    config: RunConfig,
    frame_resolver: FrameResolver,
    logger: logging.Logger,
) -> dict[str, SyncedFrames]:
    synced: dict[str, SyncedFrames] = {}
    resolved = 0
    failed = 0
    for sample in iter_with_progress(
        samples,
        total=len(samples),
        enabled=config.progress,
        description="同步帧媒体",
        item_label=lambda item: item.id,
    ):
        try:
            frame_paths = [
                (frame.frame_index, frame_resolver(frame, config))
                for frame in _selected_frames(sample, max_frames=config.max_frames)
            ]
            if not frame_paths:
                raise RuntimeError("spatial sample has no selected frames")
        except Exception as exc:  # noqa: BLE001 - keep per-sample isolation.
            synced[sample.id] = SyncedFrames(frame_paths=None, error=exc)
            failed += 1
        else:
            synced[sample.id] = SyncedFrames(frame_paths=frame_paths)
            resolved += 1
        _log_media_sync_progress(
            logger,
            processed=len(synced),
            total=len(samples),
            resolved=resolved,
            failed=failed,
            current_id=sample.id,
            every=PROGRESS_LOG_EVERY,
        )
    return synced


def _default_frame_resolver(frame: FrameRef, config: RunConfig) -> Path:
    return resolve_or_download_media(
        media_ref=frame.path,
        repo_id=config.repo_id,
        revision=config.revision,
        cache_dir=config.cache_dir,
    )


def _selected_frames(sample: SpatialSample, *, max_frames: int) -> list[FrameRef]:
    if max_frames <= 0 or len(sample.frames) <= max_frames:
        return list(sample.frames)
    if max_frames == 1:
        return [sample.frames[0]]
    step = (len(sample.frames) - 1) / float(max_frames - 1)
    selected_indices = sorted(
        {min(len(sample.frames) - 1, round(index * step)) for index in range(max_frames)}
    )
    return [sample.frames[index] for index in selected_indices]


def _sample_preview(sample: SpatialSample) -> dict[str, Any]:
    return {
        "id": sample.id,
        "video_id": sample.video_id,
        "source_dataset": sample.source_dataset,
        "source_split": sample.source_split,
        "task_type": sample.task_type,
        "question": sample.question,
        "target_ref": sample.target_ref,
        "frame_count": len(sample.frames),
        "box_track_count": len(sample.gt_box_track),
        "point_track_count": len(sample.gt_point_track),
        "reference_mask_path": sample.reference_mask_path,
    }


REQUIRED_EXPORT_FIELDS = (
    "id",
    "sample_id",
    "video_id",
    "source_dataset",
    "source_split",
    "task_type",
    "question",
    "frame_refs",
    "resolved_frame_paths",
    "gt_box_track",
    "gt_point_track",
    "hf_dataset_repo",
    "hf_revision",
    "hf_split",
    "model",
    "prompt_mode",
    "experiment_stage",
    "platform",
    "hardware_profile",
    "source_record",
)


def _attach_export_completeness(record: dict[str, Any]) -> None:
    missing = [
        field
        for field in REQUIRED_EXPORT_FIELDS
        if record.get(field) in (None, "", [])
    ]
    record["export_required_fields"] = list(REQUIRED_EXPORT_FIELDS)
    record["export_missing_fields"] = missing
    record["export_complete"] = not missing


def _export_completeness_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    incomplete = [
        record for record in records if not bool(record.get("export_complete"))
    ]
    missing_counts: dict[str, int] = {}
    for record in incomplete:
        for field in record.get("export_missing_fields") or []:
            missing_counts[field] = missing_counts.get(field, 0) + 1
    return {
        "required_fields": list(REQUIRED_EXPORT_FIELDS),
        "record_count": len(records),
        "complete_records": len(records) - len(incomplete),
        "incomplete_records": len(incomplete),
        "missing_field_counts": dict(sorted(missing_counts.items())),
    }


def _paper_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_samples": summary.get("total_samples"),
        "valid_prediction_count": summary.get("valid_prediction_count"),
        "inference_success_rate": summary.get("inference_success_rate"),
        "parse_success_rate": summary.get("parse_success_rate"),
        "spatial_metrics_enabled": summary.get("spatial_metrics_enabled"),
        "evidence_accuracy": summary.get("evidence_accuracy"),
        "pointing_accuracy": summary.get("pointing_accuracy"),
        "spatial_box_iou": summary.get("spatial_box_iou"),
        "recall_at_box_iou_0_3": summary.get("recall_at_box_iou_0_3"),
        "recall_at_box_iou_0_5": summary.get("recall_at_box_iou_0_5"),
        "gt_box_count": summary.get("gt_box_count"),
        "pred_point_count": summary.get("pred_point_count"),
        "pred_box_count": summary.get("pred_box_count"),
        "average_latency_seconds": summary.get("average_latency_seconds"),
        "cuda_peak_memory_bytes": summary.get("cuda_peak_memory_bytes"),
    }

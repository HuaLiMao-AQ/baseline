"""端到端 baseline 执行器。"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import platform
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from .adapters.base import ModelAdapter
from .adapters.internvl import InternVLAdapter, InternVLConfig
from .adapters.llava_onevision import LlavaOneVisionAdapter, LlavaOneVisionConfig
from .adapters.qwen_vl import (
    DEFAULT_QWEN_VL_MODEL_ID,
    QwenVLAdapter,
    QwenVLConfig,
)
from .cache import configure_runtime_cache, model_cache_dir
from .dataset import (
    DEFAULT_LIMIT,
    DEFAULT_PLATFORM,
    DEFAULT_REPO_ID,
    DEFAULT_REVISION,
    DEFAULT_SAMPLE_MODE,
    DEFAULT_SEED,
    DEFAULT_SPLIT,
    DEFAULT_TASK_TYPE,
    DatasetLoadResult,
    DatasetSample,
    filter_temporal_qa,
    load_temporal_samples,
    select_samples,
)
from .devices import collect_torch_accelerator_info
from .log_utils import configure_run_logging
from .media import probe_video_duration, resolve_or_download_media
from .metrics import summarize_predictions
from .parser import parse_model_output
from .prompting import PROMPT_MODE_GROUNDED, PROMPT_MODES
from .progress import iter_with_progress, rich_status


DEFAULT_MAX_NEW_TOKENS = 256


@dataclass(frozen=True, slots=True)
class RunConfig:
    """一次 baseline 运行的配置。

    Attributes:
        repo_id: Hugging Face dataset repo ID。
        revision: 固定 revision 或 tag。
        split: 数据集 split。
        task_type: 任务类型，当前默认 ``temporal_qa``。
        model: 模型 ID。
        limit: 最大样本数。
        seed: 随机抽样 seed。
        sample_mode: 抽样模式。
        output_dir: 输出目录；为空时自动生成。
        cache_dir: split 与媒体缓存目录。
        dry_run: 是否只做数据读取和配置输出。
        resume: 是否跳过已有预测 ID。
        overwrite: 是否覆盖已有输出。
        device: 推理设备；PRO 6000 单卡固定为 ``cuda``。
        dtype: 推理精度。
        max_frames: 视频最大采样帧数。
        fps: 可选采样 FPS。
        max_pixels: 可选视觉输入最大像素数。
        max_new_tokens: 最大生成 token 数。
        local_jsonl: 可选本地 JSONL，用于测试和 smoke run。
        progress: 是否使用 Rich 显示运行进度。
        log_file: 干净运行日志文件；默认写入输出目录下的 ``run.log``。
        media_sync: 媒体同步模式，``eager`` 会在模型推理前解析/下载本次样本视频。
        prompt_mode: ``answer_only`` 或 ``grounded``。
        experiment_stage: 当前实验阶段名，用于论文绘图和结果回溯。
        platform: 数据与模型平台，当前固定为 Hugging Face Hub。
        hardware_profile: 记录本次运行面向的硬件 profile。
    """

    repo_id: str = DEFAULT_REPO_ID
    revision: str = DEFAULT_REVISION
    split: str = DEFAULT_SPLIT
    task_type: str = DEFAULT_TASK_TYPE
    model: str = DEFAULT_QWEN_VL_MODEL_ID
    limit: int | None = DEFAULT_LIMIT
    seed: int = DEFAULT_SEED
    sample_mode: str = DEFAULT_SAMPLE_MODE
    output_dir: Path | None = None
    cache_dir: Path = Path(".cache/evidenceqa-baseline")
    dry_run: bool = False
    resume: bool = True
    overwrite: bool = False
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_frames: int = 64
    fps: float | None = None
    max_pixels: int | None = 768 * 28 * 28
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    local_jsonl: Path | None = None
    progress: bool = True
    log_file: Path | None = None
    media_sync: str = "eager"
    prompt_mode: str = PROMPT_MODE_GROUNDED
    experiment_stage: str = "single"
    platform: str = DEFAULT_PLATFORM
    hardware_profile: str = "rtx-pro-6000-96gb-single-cuda"


@dataclass(frozen=True, slots=True)
class RunResult:
    """一次运行产生的路径和汇总结果。

    Attributes:
        output_dir: 本次运行输出目录。
        run_config_path: ``run_config.json`` 路径。
        predictions_path: ``predictions.jsonl`` 路径。
        failed_samples_path: ``failed_samples.jsonl`` 路径。
        summary_path: ``summary.json`` 路径。
        log_path: 干净运行日志路径。
        summary: 汇总指标内容。
    """

    output_dir: Path
    run_config_path: Path
    predictions_path: Path
    failed_samples_path: Path
    summary_path: Path
    log_path: Path
    summary: dict[str, Any]


MediaResolver = Callable[[DatasetSample, RunConfig], Path]
DurationProbe = Callable[[Path], float | None]
PROGRESS_LOG_EVERY = 25


@dataclass(frozen=True, slots=True)
class SyncedMedia:
    """已提前解析的样本媒体路径和视频时长。"""

    path: Path | None
    duration_seconds: float | None
    error: Exception | None = None


def run_baseline(
    config: RunConfig,
    *,
    samples: list[DatasetSample] | None = None,
    media_resolver: MediaResolver | None = None,
    duration_probe: DurationProbe = probe_video_duration,
) -> RunResult:
    """执行 baseline 验证流程。

    Args:
        config: 本次运行配置。
        samples: 可选内存样本列表；测试时使用。
        media_resolver: 可选媒体解析函数；测试时可注入。
        duration_probe: 视频时长探测函数。

    Returns:
        输出路径和 summary 内容。
    """

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
        config.prompt_mode,
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

    if config.prompt_mode not in PROMPT_MODES:
        raise ValueError(f"prompt_mode 必须是 {PROMPT_MODES!r} 之一")

    if config.overwrite:
        for path in (predictions_path, failed_samples_path, summary_path):
            path.unlink(missing_ok=True)

    _log_stage_start(logger, "load_dataset")
    with rich_status(config.progress, "读取并筛选数据集 split..."):
        dataset_result = _load_samples(config, samples=samples)
    selected = dataset_result.selected_samples
    logger.info(
        "stage_complete name=load_dataset split_path=%s total_rows=%d temporal_rows=%d "
        "selected_samples=%d",
        dataset_result.split_path,
        dataset_result.total_rows,
        dataset_result.temporal_rows,
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
        "started_at": datetime.now(UTC).isoformat(),
        "runtime": runtime,
        "runtime_cache": runtime_cache,
        "platform": config.platform,
        "hardware_profile": config.hardware_profile,
        "log_file": str(log_path),
        "split_path": str(dataset_result.split_path),
        "total_rows": dataset_result.total_rows,
        "temporal_rows": dataset_result.temporal_rows,
        "selected_sample_ids": [sample.id for sample in selected],
        "selected_samples": [_sample_preview(sample) for sample in selected],
    }
    _write_json(run_config_path, run_config_payload)

    if config.dry_run:
        summary = {
            "dry_run": True,
            "total_rows": dataset_result.total_rows,
            "temporal_rows": dataset_result.temporal_rows,
            "selected_samples": len(selected),
            "sample_preview": [_sample_preview(sample) for sample in selected[:5]],
        }
        _write_json(summary_path, summary)
        logger.info(
            "run_complete dry_run=true total_rows=%d temporal_rows=%d selected_samples=%d",
            dataset_result.total_rows,
            dataset_result.temporal_rows,
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

    resolver = media_resolver or _default_media_resolver
    completed_ids = read_completed_ids(predictions_path) if config.resume else set()
    pending_samples = [sample for sample in selected if sample.id not in completed_ids]
    logger.info(
        "resume_state completed_samples=%d pending_samples=%d",
        len(completed_ids),
        len(pending_samples),
    )
    synced_media: dict[str, SyncedMedia] = {}
    if config.media_sync == "eager":
        logger.info(
            "stage_start name=media_sync pending_samples=%d",
            len(pending_samples),
        )
        with rich_status(config.progress, "同步本次样本媒体..."):
            synced_media = _sync_selected_media(
                pending_samples,
                config=config,
                media_resolver=resolver,
                duration_probe=duration_probe,
                logger=logger,
            )
        logger.info(
            "stage_complete name=media_sync %s",
            _format_media_sync_log(synced_media),
        )
    elif config.media_sync != "lazy":
        raise ValueError("media_sync 必须是 'eager' 或 'lazy'")

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
            model_adapter = _build_adapter(config)
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
        description="运行 Video-LMM baseline",
        item_label=lambda item: item.id,
    ):
        record = _run_one_sample(
            sample=sample,
            config=config,
            model_getter=get_model_adapter,
            media_resolver=resolver,
            duration_probe=duration_probe,
            synced_media=synced_media.get(sample.id),
        )
        _append_jsonl(predictions_path, record)
        prediction_stats.record(record)
        if record.get("error") or not record.get("parse_success"):
            synced = synced_media.get(sample.id)
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
    summary = summarize_predictions(
        all_records,
        cuda_peak_memory_bytes=_cuda_peak_memory_bytes(),
        include_temporal_metrics=config.prompt_mode == PROMPT_MODE_GROUNDED,
    )
    summary["dry_run"] = False
    summary["prompt_mode"] = config.prompt_mode
    summary["experiment_stage"] = config.experiment_stage
    summary["selected_samples"] = len(selected)
    summary["media_sync"] = _media_sync_summary(config.media_sync, synced_media)
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


def read_completed_ids(predictions_path: Path) -> set[str]:
    """读取已完成预测的样本 ID。

    Args:
        predictions_path: 已有 ``predictions.jsonl`` 路径。

    Returns:
        文件中已出现的样本 ID 集合。
    """

    return {
        str(record["id"])
        for record in read_prediction_rows(predictions_path)
        if "id" in record
    }


def read_prediction_rows(predictions_path: Path) -> list[dict[str, Any]]:
    """读取预测 JSONL。

    Args:
        predictions_path: ``predictions.jsonl`` 路径。

    Returns:
        预测记录列表；文件不存在时返回空列表。
    """

    if not predictions_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def collect_runtime_info() -> dict[str, Any]:
    """收集可复现性运行环境信息。

    Returns:
        Python、平台、PyTorch、Transformers 和 CUDA 元数据。
    """

    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {},
        "cuda": {"available": False, "device_name": None},
    }
    for package in ("huggingface_hub", "torch", "transformers"):
        try:
            info["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            info["packages"][package] = None

    try:
        import torch
    except ImportError:
        return info

    info.update(collect_torch_accelerator_info(torch))
    return info


def _load_samples(
    config: RunConfig,
    *,
    samples: list[DatasetSample] | None,
) -> DatasetLoadResult:
    if samples is None:
        return load_temporal_samples(
            repo_id=config.repo_id,
            revision=config.revision,
            split=config.split,
            task_type=config.task_type,
            limit=config.limit,
            seed=config.seed,
            sample_mode=config.sample_mode,
            cache_dir=config.cache_dir,
            local_jsonl=config.local_jsonl,
        )

    rows = [sample.raw for sample in samples]
    temporal_samples = filter_temporal_qa(rows, task_type=config.task_type)
    selected = select_samples(
        temporal_samples,
        limit=config.limit,
        seed=config.seed,
        sample_mode=config.sample_mode,
    )
    return DatasetLoadResult(
        split_path=config.local_jsonl or Path("<in-memory>"),
        total_rows=len(rows),
        temporal_rows=len(temporal_samples),
        selected_samples=selected,
    )


def _build_adapter(config: RunConfig) -> ModelAdapter:
    normalized_model = config.model.lower().replace("_", "-")
    hf_model_cache_dir = model_cache_dir(config.cache_dir)
    if "llava-onevision" in normalized_model:
        return LlavaOneVisionAdapter(
            LlavaOneVisionConfig(
                model_id=config.model,
                model_cache_dir=hf_model_cache_dir,
                device=config.device,
                dtype=config.dtype,
                max_frames=config.max_frames,
                max_pixels=config.max_pixels,
                max_new_tokens=config.max_new_tokens,
                prompt_mode=config.prompt_mode,
            )
        )
    if "internvl" in normalized_model:
        return InternVLAdapter(
            InternVLConfig(
                model_id=config.model,
                model_cache_dir=hf_model_cache_dir,
                device=config.device,
                dtype=config.dtype,
                max_frames=config.max_frames,
                max_new_tokens=config.max_new_tokens,
                prompt_mode=config.prompt_mode,
            )
        )
    return QwenVLAdapter(
        QwenVLConfig(
            model_id=config.model,
            model_cache_dir=hf_model_cache_dir,
            device=config.device,
            dtype=config.dtype,
            max_frames=config.max_frames,
            fps=config.fps,
            max_pixels=config.max_pixels,
            max_new_tokens=config.max_new_tokens,
            prompt_mode=config.prompt_mode,
        )
    )


def _run_one_sample(
    *,
    sample: DatasetSample,
    config: RunConfig,
    model_getter: Callable[[], ModelAdapter],
    media_resolver: MediaResolver,
    duration_probe: DurationProbe,
    synced_media: SyncedMedia | None = None,
) -> dict[str, Any]:
    base_record: dict[str, Any] = {
        "id": sample.id,
        "sample_id": sample.id,
        "video_id": sample.video_id,
        "source_dataset": sample.source_dataset,
        "source_split": sample.source_split,
        "task_type": sample.task_type,
        "question": sample.question,
        "gt_answer": sample.gt_answer,
        "gt_temporal_evidence": sample.gt_temporal_evidence,
        "media_ref": sample.media_path,
        "resolved_media_path": None,
        "duration_seconds": sample.duration_seconds,
        "hf_dataset_repo": config.repo_id,
        "hf_revision": config.revision,
        "hf_split": config.split,
        "model": config.model,
        "prompt_mode": config.prompt_mode,
        "experiment_stage": config.experiment_stage,
        "platform": config.platform,
        "hardware_profile": config.hardware_profile,
        "source_record": sample.raw,
        "pred_answer": None,
        "pred_temporal_evidence": [],
        "parse_success": False,
        "parse_error": None,
        "was_repaired": False,
        "raw_output": "",
        "error": None,
        "latency_seconds": 0.0,
    }

    started = time.perf_counter()
    try:
        if synced_media is not None:
            if synced_media.error is not None:
                raise synced_media.error
            if synced_media.path is None:
                raise RuntimeError("媒体同步没有产出本地路径")
            media_path = synced_media.path
            duration = synced_media.duration_seconds
        else:
            media_path = media_resolver(sample, config)
            duration = sample.duration_seconds or duration_probe(media_path)
        if duration is None or duration <= 0:
            raise RuntimeError("视频时长不可用")
        base_record["resolved_media_path"] = str(media_path)
        base_record["duration_seconds"] = duration
        sample_for_prompt = DatasetSample(
            sample_id=sample.id,
            video_id=sample.video_id,
            source_dataset=sample.source_dataset,
            source_split=sample.source_split,
            task_type=sample.task_type,
            question=sample.question,
            gt_answer=sample.gt_answer,
            gt_temporal_evidence=sample.gt_temporal_evidence,
            media_path=sample.media_path,
            duration_seconds=duration,
            raw=sample.raw,
        )
        raw_output = model_getter().predict(sample_for_prompt, media_path)
        latency = time.perf_counter() - started
        parsed = parse_model_output(
            raw_output,
            duration_seconds=duration,
            require_temporal_evidence=config.prompt_mode == PROMPT_MODE_GROUNDED,
        )
        base_record.update(
            {
                "pred_answer": parsed.answer,
                "pred_temporal_evidence": (
                    parsed.temporal_evidence
                    if config.prompt_mode == PROMPT_MODE_GROUNDED
                    else []
                ),
                "parse_success": parsed.parse_success,
                "parse_error": parsed.parse_error,
                "was_repaired": parsed.was_repaired,
                "raw_output": parsed.raw_output,
                "latency_seconds": latency,
            }
        )
        _attach_export_completeness(base_record)
        return base_record
    except Exception as exc:  # noqa: BLE001 - 这里需要样本级隔离。
        _release_cuda_after_exception()
        latency = time.perf_counter() - started
        base_record.update(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "latency_seconds": latency,
            }
        )
        _attach_export_completeness(base_record)
        return base_record


def _release_cuda_after_exception() -> None:
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except RuntimeError:
        pass


def _default_media_resolver(sample: DatasetSample, config: RunConfig) -> Path:
    return resolve_or_download_media(
        media_ref=sample.media_path,
        repo_id=config.repo_id,
        revision=config.revision,
        cache_dir=config.cache_dir,
    )


def _sync_selected_media(
    samples: list[DatasetSample],
    *,
    config: RunConfig,
    media_resolver: MediaResolver,
    duration_probe: DurationProbe,
    logger: logging.Logger,
) -> dict[str, SyncedMedia]:
    synced: dict[str, SyncedMedia] = {}
    resolved = 0
    failed = 0
    for sample in iter_with_progress(
        samples,
        total=len(samples),
        enabled=config.progress,
        description="同步视频媒体",
        item_label=lambda item: item.id,
    ):
        try:
            media_path = media_resolver(sample, config)
            duration = sample.duration_seconds or duration_probe(media_path)
            if duration is None or duration <= 0:
                raise RuntimeError("视频时长不可用")
        except Exception as exc:  # noqa: BLE001 - 保持样本级失败隔离。
            synced[sample.id] = SyncedMedia(
                path=None,
                duration_seconds=None,
                error=exc,
            )
            failed += 1
        else:
            synced[sample.id] = SyncedMedia(
                path=media_path,
                duration_seconds=duration,
            )
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


@dataclass(slots=True)
class _PredictionStats:
    processed: int = 0
    success: int = 0
    failed: int = 0
    parse_failed: int = 0

    def record(self, record: dict[str, Any]) -> None:
        self.processed += 1
        if record.get("error"):
            self.failed += 1
            return
        if record.get("parse_success") is True:
            self.success += 1
        else:
            self.parse_failed += 1


def _log_stage_start(logger: logging.Logger, name: str) -> None:
    logger.info("stage_start name=%s", name)


def _should_log_progress(processed: int, total: int, every: int) -> bool:
    if total <= 0:
        return False
    if processed == total:
        return True
    return every > 0 and processed % every == 0


def _log_media_sync_progress(
    logger: logging.Logger,
    *,
    processed: int,
    total: int,
    resolved: int,
    failed: int,
    current_id: str,
    every: int,
) -> None:
    if not _should_log_progress(processed, total, every):
        return
    logger.info(
        "media_sync_progress processed=%d total=%d resolved=%d failed=%d current_id=%s",
        processed,
        total,
        resolved,
        failed,
        current_id,
    )


def _log_prediction_progress(
    logger: logging.Logger,
    *,
    stats: _PredictionStats,
    total: int,
    current_id: str,
    every: int,
) -> None:
    if not _should_log_progress(stats.processed, total, every):
        return
    logger.info(
        "prediction_progress processed=%d total=%d success=%d failed=%d "
        "parse_failed=%d current_id=%s",
        stats.processed,
        total,
        stats.success,
        stats.failed,
        stats.parse_failed,
        current_id,
    )


def _media_sync_summary(
    mode: str,
    synced_media: dict[str, SyncedMedia],
) -> dict[str, Any]:
    failed = sum(1 for item in synced_media.values() if item.error is not None)
    return {
        "mode": mode,
        "attempted": len(synced_media),
        "resolved": len(synced_media) - failed,
        "failed": failed,
    }


def _format_media_sync_log(synced_media: dict[str, SyncedMedia]) -> str:
    summary = _media_sync_summary("eager", synced_media)
    parts = [
        f"attempted={summary['attempted']}",
        f"resolved={summary['resolved']}",
        f"failed={summary['failed']}",
    ]
    if summary["failed"]:
        parts.append(f"failure_counts={_media_sync_failure_counts(synced_media)}")
        parts.append(f"failure_examples={_media_sync_failure_examples(synced_media)}")
    return " ".join(parts)


def _media_sync_failure_counts(
    synced_media: dict[str, SyncedMedia],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in synced_media.values():
        if item.error is None:
            continue
        signature = _error_signature(item.error)
        counts[signature] = counts.get(signature, 0) + 1
    return dict(sorted(counts.items()))


def _media_sync_failure_examples(
    synced_media: dict[str, SyncedMedia],
    *,
    limit: int = 5,
) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    for sample_id, item in synced_media.items():
        if item.error is None:
            continue
        examples.append(
            {
                "id": sample_id,
                "error": _short_error(item.error) or "unknown error",
            }
        )
        if len(examples) >= limit:
            break
    return examples


def _error_signature(error: Exception) -> str:
    message = str(error)
    if "Network is unreachable" in message:
        return "Network is unreachable"
    if "Name or service not known" in message:
        return "DNS unavailable"
    if "Connection timed out" in message:
        return "Connection timed out"
    if "Read timed out" in message:
        return "Read timed out"
    if "HTTP 401" in message or "access denied" in message.lower():
        return "HF access denied"
    if "HTTP 403" in message:
        return "HF forbidden"
    if "HTTP 404" in message:
        return "HF file not found"
    return _short_error(message, max_length=120) or "unknown error"


def _short_error(error: object, *, max_length: int = 300) -> str | None:
    if error in (None, ""):
        return None
    text = str(error).replace("\n", " ").strip()
    if len(text) <= max_length:
        return text
    return f"{text[: max_length - 3]}..."


def _default_output_dir(config: RunConfig) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    model_slug = config.model.rsplit("/", 1)[-1].replace("_", "-").lower()
    limit = "all" if config.limit is None else str(config.limit)
    return Path("runs") / f"{timestamp}-{model_slug}-{config.split}-{limit}"


def _jsonable_config(config: RunConfig, output_dir: Path) -> dict[str, Any]:
    payload = asdict(config)
    payload["output_dir"] = str(output_dir)
    payload["cache_dir"] = str(config.cache_dir)
    payload["local_jsonl"] = str(config.local_jsonl) if config.local_jsonl else None
    payload["log_file"] = str(config.log_file) if config.log_file else None
    return payload


def _sample_preview(sample: DatasetSample) -> dict[str, Any]:
    return {
        "id": sample.id,
        "video_id": sample.video_id,
        "source_dataset": sample.source_dataset,
        "source_split": sample.source_split,
        "task_type": sample.task_type,
        "question": sample.question,
        "media_path": sample.media_path,
        "duration_seconds": sample.duration_seconds,
        "gt_answer": sample.gt_answer,
        "gt_temporal_evidence": sample.gt_temporal_evidence,
    }


REQUIRED_EXPORT_FIELDS = (
    "id",
    "sample_id",
    "video_id",
    "source_dataset",
    "source_split",
    "task_type",
    "question",
    "gt_answer",
    "gt_temporal_evidence",
    "media_ref",
    "duration_seconds",
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
        "answer_accuracy": summary.get("answer_accuracy"),
        "answer_token_f1": summary.get("answer_token_f1"),
        "temporal_metrics_enabled": summary.get("temporal_metrics_enabled"),
        "temporal_evidence_iou": summary.get("temporal_evidence_iou"),
        "recall_at_iou_0_3": summary.get("recall_at_iou_0_3"),
        "recall_at_iou_0_5": summary.get("recall_at_iou_0_5"),
        "acc_correct_evidence_iou_0_3": summary.get(
            "acc_correct_evidence_iou_0_3"
        ),
        "acc_correct_evidence_iou_0_5": summary.get(
            "acc_correct_evidence_iou_0_5"
        ),
        "answer_evidence_gap_iou_0_5": summary.get(
            "answer_evidence_gap_iou_0_5"
        ),
        "average_latency_seconds": summary.get("average_latency_seconds"),
        "cuda_peak_memory_bytes": summary.get("cuda_peak_memory_bytes"),
    }


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _reset_cuda_peak_memory() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _cuda_peak_memory_bytes() -> int | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated())

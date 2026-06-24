"""预测记录构造。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from evidenceqa_baseline_refactor.adapters import AdapterResponse
from evidenceqa_baseline_refactor.config import DataConfig, ModelConfig, StageConfig
from evidenceqa_baseline_refactor.dataset import SpatialSample, TemporalSample
from evidenceqa_baseline_refactor.parser import (
    ParsedSpatialPrediction,
    ParsedTemporalPrediction,
)

TEMPORAL_REQUIRED_FIELDS = [
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
]

SPATIAL_REQUIRED_FIELDS = [
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
]


def build_temporal_record(
    *,
    sample: TemporalSample,
    parsed: ParsedTemporalPrediction,
    response: AdapterResponse,
    data_config: DataConfig,
    model_config: ModelConfig,
    stage_config: StageConfig,
    platform: str,
    hardware_profile: str,
    resolved_media_path: Path | str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """构造 temporal `predictions.jsonl` 单行记录。"""

    record: dict[str, Any] = {
        "id": sample.sample_id,
        "sample_id": sample.sample_id,
        "video_id": sample.video_id,
        "source_dataset": sample.source_dataset,
        "source_split": sample.source_split,
        "task_type": sample.task_type,
        "question": sample.question,
        "gt_answer": sample.gt_answer,
        "gt_temporal_evidence": sample.gt_temporal_evidence,
        "media_ref": sample.media_path,
        "duration_seconds": sample.duration_seconds,
        "hf_dataset_repo": data_config.repo_id,
        "hf_revision": data_config.revision,
        "hf_split": data_config.split,
        "model": model_config.model_id,
        "prompt_mode": stage_config.prompt_mode,
        "experiment_stage": stage_config.name,
        "platform": platform,
        "hardware_profile": hardware_profile,
        "source_record": sample.raw,
        "raw_output": response.raw_output,
        "latency_seconds": response.latency_seconds,
        "parse_success": parsed.parse_success,
        "parse_error": parsed.parse_error,
        "was_repaired": parsed.was_repaired,
        "pred_answer": parsed.answer if parsed.parse_success else None,
        "pred_temporal_evidence": (
            parsed.temporal_evidence if parsed.parse_success else []
        ),
        "adapter_metadata": response.metadata,
        "error": error,
    }
    if resolved_media_path is not None:
        record["resolved_media_path"] = str(resolved_media_path)
    return add_export_status(record, TEMPORAL_REQUIRED_FIELDS)


def build_spatial_record(
    *,
    sample: SpatialSample,
    parsed: ParsedSpatialPrediction,
    response: AdapterResponse,
    data_config: DataConfig,
    model_config: ModelConfig,
    stage_config: StageConfig,
    platform: str,
    hardware_profile: str,
    resolved_frame_paths: list[dict[str, Any]] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """构造 spatial `predictions.jsonl` 单行记录。"""

    record: dict[str, Any] = {
        "id": sample.sample_id,
        "sample_id": sample.sample_id,
        "video_id": sample.video_id,
        "source_dataset": sample.source_dataset,
        "source_split": sample.source_split,
        "task_type": sample.task_type,
        "question": sample.question,
        "target_ref": sample.target_ref,
        "frame_refs": [asdict(frame) for frame in sample.frames],
        "resolved_frame_paths": resolved_frame_paths
        if resolved_frame_paths is not None
        else [],
        "gt_box_track": [asdict(item) for item in sample.gt_box_track],
        "gt_point_track": [asdict(item) for item in sample.gt_point_track],
        "reference_mask_path": sample.reference_mask_path,
        "hf_dataset_repo": data_config.repo_id,
        "hf_revision": data_config.revision,
        "hf_split": data_config.split,
        "model": model_config.model_id,
        "prompt_mode": stage_config.prompt_mode,
        "experiment_stage": stage_config.name,
        "platform": platform,
        "hardware_profile": hardware_profile,
        "source_record": sample.raw,
        "raw_output": response.raw_output,
        "latency_seconds": response.latency_seconds,
        "parse_success": parsed.parse_success,
        "parse_error": parsed.parse_error,
        "was_repaired": parsed.was_repaired,
        "pred_target": parsed.target if parsed.parse_success else None,
        "pred_frame_index": parsed.frame_index if parsed.parse_success else None,
        "pred_point": parsed.point if parsed.parse_success else None,
        "pred_box": parsed.box if parsed.parse_success else None,
        "adapter_metadata": response.metadata,
        "error": error,
    }
    return add_export_status(record, SPATIAL_REQUIRED_FIELDS)


def add_export_status(
    record: dict[str, Any],
    required_fields: list[str],
) -> dict[str, Any]:
    """追加导出完整性字段。"""

    missing = [
        field
        for field in required_fields
        if field not in record or record[field] is None
    ]
    record["export_required_fields"] = list(required_fields)
    record["export_missing_fields"] = missing
    record["export_complete"] = not missing
    return record

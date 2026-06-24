"""单阶段 baseline runner。"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any

from evidenceqa_baseline_refactor.adapters import (
    AdapterResponse,
    BaseModelAdapter,
    GenerationConfig,
    PredictionRequest,
)
from evidenceqa_baseline_refactor.config import DataConfig, ModelConfig, StageConfig
from evidenceqa_baseline_refactor.dataset import SpatialSample, TemporalSample
from evidenceqa_baseline_refactor.jsonl import write_json, write_jsonl
from evidenceqa_baseline_refactor.metrics import (
    summarize_spatial_predictions,
    summarize_temporal_predictions,
)
from evidenceqa_baseline_refactor.parser import (
    ParsedSpatialPrediction,
    ParsedTemporalPrediction,
    parse_spatial_prediction,
    parse_temporal_prediction,
)
from evidenceqa_baseline_refactor.prompting import (
    build_spatial_prompt,
    build_temporal_prompt,
)
from evidenceqa_baseline_refactor.records import (
    build_spatial_record,
    build_temporal_record,
)


@dataclass(frozen=True, slots=True)
class StageResult:
    """单阶段运行结果。"""

    output_dir: Path
    predictions_path: Path
    summary_path: Path
    run_config_path: Path
    summary: dict[str, Any]


def run_temporal_stage(
    *,
    adapter: BaseModelAdapter,
    samples: list[TemporalSample],
    data_config: DataConfig,
    model_config: ModelConfig,
    stage_config: StageConfig,
    output_dir: Path,
    platform: str,
    hardware_profile: str,
) -> StageResult:
    """运行 answer-only 或 grounded temporal 阶段。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    records = [
        _run_one_temporal_sample(
            adapter=adapter,
            sample=sample,
            data_config=data_config,
            model_config=model_config,
            stage_config=stage_config,
            platform=platform,
            hardware_profile=hardware_profile,
        )
        for sample in samples
    ]
    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"
    run_config_path = output_dir / "run_config.json"
    write_jsonl(predictions_path, records)
    summary = summarize_temporal_predictions(
        records,
        include_temporal_metrics=stage_config.prompt_mode == "grounded",
    )
    summary.update(
        {
            "selected_samples": len(samples),
            "prompt_mode": stage_config.prompt_mode,
            "experiment_stage": stage_config.name,
        }
    )
    write_json(summary_path, summary)
    write_json(
        run_config_path,
        _run_config_payload(
            data_config=data_config,
            model_config=model_config,
            stage_config=stage_config,
            platform=platform,
            hardware_profile=hardware_profile,
        ),
    )
    return StageResult(output_dir, predictions_path, summary_path, run_config_path, summary)


def run_spatial_stage(
    *,
    adapter: BaseModelAdapter,
    samples: list[SpatialSample],
    data_config: DataConfig,
    model_config: ModelConfig,
    stage_config: StageConfig,
    output_dir: Path,
    platform: str,
    hardware_profile: str,
) -> StageResult:
    """运行 spatial/ref 阶段。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    records = [
        _run_one_spatial_sample(
            adapter=adapter,
            sample=sample,
            data_config=data_config,
            model_config=model_config,
            stage_config=stage_config,
            platform=platform,
            hardware_profile=hardware_profile,
        )
        for sample in samples
    ]
    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"
    run_config_path = output_dir / "run_config.json"
    write_jsonl(predictions_path, records)
    summary = summarize_spatial_predictions(records)
    summary.update(
        {
            "selected_samples": len(samples),
            "prompt_mode": stage_config.prompt_mode,
            "experiment_stage": stage_config.name,
        }
    )
    write_json(summary_path, summary)
    write_json(
        run_config_path,
        _run_config_payload(
            data_config=data_config,
            model_config=model_config,
            stage_config=stage_config,
            platform=platform,
            hardware_profile=hardware_profile,
        ),
    )
    return StageResult(output_dir, predictions_path, summary_path, run_config_path, summary)


def _run_one_temporal_sample(
    *,
    adapter: BaseModelAdapter,
    sample: TemporalSample,
    data_config: DataConfig,
    model_config: ModelConfig,
    stage_config: StageConfig,
    platform: str,
    hardware_profile: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        prompt = build_temporal_prompt(
            question=sample.question,
            duration_seconds=sample.duration_seconds or 0.0,
            prompt_mode=stage_config.prompt_mode,
        )
        request = PredictionRequest(
            sample=sample,
            prompt=prompt.as_text(),
            prompt_mode=stage_config.prompt_mode,
            media_path=Path(sample.media_path) if sample.media_path else None,
        )
        response = adapter.generate(
            request,
            GenerationConfig(max_new_tokens=stage_config.max_new_tokens),
        )
        response = _ensure_latency(response, started)
        parsed = parse_temporal_prediction(
            response.raw_output,
            require_temporal_evidence=stage_config.prompt_mode == "grounded",
            duration_seconds=sample.duration_seconds,
        )
        return build_temporal_record(
            sample=sample,
            parsed=parsed,
            response=response,
            data_config=data_config,
            model_config=model_config,
            stage_config=stage_config,
            platform=platform,
            hardware_profile=hardware_profile,
            resolved_media_path=sample.media_path,
        )
    except Exception as exc:  # noqa: BLE001
        response = AdapterResponse(
            raw_output="",
            latency_seconds=time.perf_counter() - started,
        )
        parsed = ParsedTemporalPrediction(
            answer=None,
            temporal_evidence=[],
            parse_success=False,
            parse_error=str(exc),
            raw_output="",
        )
        return build_temporal_record(
            sample=sample,
            parsed=parsed,
            response=response,
            data_config=data_config,
            model_config=model_config,
            stage_config=stage_config,
            platform=platform,
            hardware_profile=hardware_profile,
            resolved_media_path=sample.media_path,
            error=str(exc),
        )


def _run_one_spatial_sample(
    *,
    adapter: BaseModelAdapter,
    sample: SpatialSample,
    data_config: DataConfig,
    model_config: ModelConfig,
    stage_config: StageConfig,
    platform: str,
    hardware_profile: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    frame_indices = [frame.frame_index for frame in sample.frames]
    resolved_frame_paths = [
        {"frame_index": frame.frame_index, "path": frame.path}
        for frame in sample.frames
    ]
    try:
        prompt = build_spatial_prompt(
            question=sample.question,
            frame_indices=frame_indices,
        )
        request = PredictionRequest(
            sample=sample,
            prompt=prompt.as_text(),
            prompt_mode=stage_config.prompt_mode,
            frame_paths=tuple(Path(frame.path) for frame in sample.frames),
        )
        response = adapter.generate(
            request,
            GenerationConfig(max_new_tokens=stage_config.max_new_tokens),
        )
        response = _ensure_latency(response, started)
        parsed = parse_spatial_prediction(
            response.raw_output,
            allowed_frame_indices=set(frame_indices),
        )
        return build_spatial_record(
            sample=sample,
            parsed=parsed,
            response=response,
            data_config=data_config,
            model_config=model_config,
            stage_config=stage_config,
            platform=platform,
            hardware_profile=hardware_profile,
            resolved_frame_paths=resolved_frame_paths,
        )
    except Exception as exc:  # noqa: BLE001
        response = AdapterResponse(
            raw_output="",
            latency_seconds=time.perf_counter() - started,
        )
        parsed = ParsedSpatialPrediction(
            target=None,
            frame_index=None,
            point=None,
            box=None,
            parse_success=False,
            parse_error=str(exc),
            raw_output="",
        )
        return build_spatial_record(
            sample=sample,
            parsed=parsed,
            response=response,
            data_config=data_config,
            model_config=model_config,
            stage_config=stage_config,
            platform=platform,
            hardware_profile=hardware_profile,
            resolved_frame_paths=resolved_frame_paths,
            error=str(exc),
        )


def _ensure_latency(response: AdapterResponse, started: float) -> AdapterResponse:
    if response.latency_seconds is not None:
        return response
    return replace(response, latency_seconds=time.perf_counter() - started)


def _run_config_payload(
    *,
    data_config: DataConfig,
    model_config: ModelConfig,
    stage_config: StageConfig,
    platform: str,
    hardware_profile: str,
) -> dict[str, Any]:
    return {
        "data": _jsonable(data_config),
        "model": _jsonable(model_config),
        "stage": _jsonable(stage_config),
        "platform": platform,
        "hardware_profile": hardware_profile,
    }


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value

"""EvidenceQA JSONL 数据读取与样本选择。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .jsonl import read_jsonl

SampleMode = Literal["sequential", "random"]
TEMPORAL_TASK_TYPE = "temporal_qa"
SPATIAL_TASK_TYPE = "spatial_grounding"


class DatasetError(ValueError):
    """数据样本字段不满足 baseline 契约。"""


@dataclass(frozen=True, slots=True)
class EvidenceSample:
    """统一样本视图。

    Args:
        sample_id: 样本 ID。
        task_type: 任务类型。
        source_dataset: 来源数据集。
        raw: 原始 JSON 记录。
    """

    sample_id: str
    task_type: str
    source_dataset: str
    raw: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TemporalSample:
    """Runner 使用的 temporal QA 样本。"""

    sample_id: str
    video_id: str
    source_dataset: str
    source_split: str
    task_type: str
    question: str
    gt_answer: str
    gt_temporal_evidence: list[list[float]]
    media_path: str | None
    duration_seconds: float | None
    raw: dict[str, Any]


@dataclass(frozen=True, slots=True)
class FrameRef:
    """Runner 使用的 frame sequence 引用。"""

    frame_id: str
    frame_index: int
    path: str
    video_id: str


@dataclass(frozen=True, slots=True)
class BoxTrackItem:
    """单帧 normalized box 标注。"""

    frame_id: str
    frame_index: int
    video_id: str
    box: list[float]
    coordinate_space: str = "normalized_0_1"


@dataclass(frozen=True, slots=True)
class PointTrackItem:
    """单帧 normalized point 标注。"""

    frame_id: str
    frame_index: int
    video_id: str
    point: list[float]
    coordinate_space: str = "normalized_0_1"


@dataclass(frozen=True, slots=True)
class SpatialSample:
    """Runner 使用的 spatial grounding 样本。"""

    sample_id: str
    video_id: str
    source_dataset: str
    source_split: str
    task_type: str
    question: str
    frames: list[FrameRef]
    gt_box_track: list[BoxTrackItem]
    gt_point_track: list[PointTrackItem]
    reference_mask_path: str | None
    target_ref: str | None
    raw: dict[str, Any]


def load_samples(path: Path) -> list[EvidenceSample]:
    """读取 EvidenceQA JSONL 并转换为轻量样本对象。"""

    samples: list[EvidenceSample] = []
    for row in read_jsonl(path):
        sample_id = str(row.get("id") or row.get("sample_id") or "")
        if not sample_id:
            raise ValueError("样本缺少 id")
        samples.append(
            EvidenceSample(
                sample_id=sample_id,
                task_type=str(row.get("task_type") or row.get("task") or "unknown"),
                source_dataset=str(row.get("source_dataset") or row.get("dataset") or "unknown"),
                raw=row,
            )
        )
    return samples


def adapt_temporal_sample(sample: EvidenceSample | dict[str, Any]) -> TemporalSample:
    """把原始样本适配成 temporal QA 契约。"""

    row = _sample_raw(sample)
    sample_id = _sample_id(row)
    question = _question(row, sample_id=sample_id)
    media_path, duration_seconds = _media(row)
    return TemporalSample(
        sample_id=sample_id,
        video_id=_video_id(row, media_path=media_path),
        source_dataset=_source_dataset(row),
        source_split=_source_split(row),
        task_type=str(row.get("task_type") or row.get("task") or TEMPORAL_TASK_TYPE),
        question=question,
        gt_answer=_answer(row),
        gt_temporal_evidence=_temporal_evidence(row),
        media_path=media_path,
        duration_seconds=duration_seconds,
        raw=row,
    )


def adapt_spatial_sample(sample: EvidenceSample | dict[str, Any]) -> SpatialSample:
    """把原始样本适配成 spatial grounding 契约。"""

    row = _sample_raw(sample)
    sample_id = _sample_id(row)
    question = _question(row, sample_id=sample_id)
    media = row.get("media")
    if not isinstance(media, dict):
        raise DatasetError(f"{sample_id}: 缺少 media object")
    video_id = _video_id(row, media_path=None)
    target = row.get("target")
    if not isinstance(target, dict):
        raise DatasetError(f"{sample_id}: 缺少 spatial target")
    return SpatialSample(
        sample_id=sample_id,
        video_id=video_id,
        source_dataset=_source_dataset(row),
        source_split=_source_split(row),
        task_type=str(row.get("task_type") or row.get("task") or SPATIAL_TASK_TYPE),
        question=question,
        frames=_frames(media, sample_id=sample_id, video_id=video_id),
        gt_box_track=_box_track(target, sample_id=sample_id, video_id=video_id),
        gt_point_track=_point_track(target, sample_id=sample_id, video_id=video_id),
        reference_mask_path=_optional_string(target.get("reference_mask_path")),
        target_ref=_target_ref(row),
        raw=row,
    )


def filter_temporal_samples(samples: list[EvidenceSample]) -> list[TemporalSample]:
    """筛选并适配 temporal QA 样本。"""

    return [
        adapt_temporal_sample(sample)
        for sample in samples
        if sample.task_type == TEMPORAL_TASK_TYPE
    ]


def filter_spatial_samples(samples: list[EvidenceSample]) -> list[SpatialSample]:
    """筛选并适配 spatial grounding 样本。"""

    return [
        adapt_spatial_sample(sample)
        for sample in samples
        if sample.task_type == SPATIAL_TASK_TYPE
    ]


def filter_by_task(samples: list[EvidenceSample], task_type: str) -> list[EvidenceSample]:
    """按任务类型筛选样本。"""

    return [sample for sample in samples if sample.task_type == task_type]


def select_samples(
    samples: list[EvidenceSample],
    *,
    limit: int | None,
    seed: int,
    mode: SampleMode,
) -> list[EvidenceSample]:
    """稳定选择实验样本。"""

    if limit is None or limit >= len(samples):
        return list(samples)
    if limit < 0:
        raise ValueError("limit 不能为负数")
    if mode == "sequential":
        return list(samples[:limit])
    if mode == "random":
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(samples)), limit))
        return [samples[index] for index in indices]
    raise ValueError(f"未知 sample mode: {mode}")


def _sample_raw(sample: EvidenceSample | dict[str, Any]) -> dict[str, Any]:
    if isinstance(sample, EvidenceSample):
        return sample.raw
    return sample


def _sample_id(row: dict[str, Any]) -> str:
    sample_id = str(row.get("id") or row.get("sample_id") or row.get("qa_id") or "")
    if not sample_id:
        raise DatasetError("样本缺少 id")
    return sample_id


def _question(row: dict[str, Any], *, sample_id: str) -> str:
    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        raise DatasetError(f"{sample_id}: 缺少 question")
    return question


def _source_dataset(row: dict[str, Any]) -> str:
    value = row.get("source_dataset") or row.get("dataset")
    if isinstance(value, str) and value.strip():
        return value
    raise DatasetError("样本缺少 source_dataset")


def _source_split(row: dict[str, Any]) -> str:
    value = row.get("source_split") or row.get("split")
    return value if isinstance(value, str) and value.strip() else "unknown"


def _video_id(row: dict[str, Any], *, media_path: str | None) -> str:
    value = row.get("video_id")
    media = row.get("media")
    if value is None and isinstance(media, dict):
        value = media.get("video_id")
    if value is None and media_path:
        value = Path(media_path).stem
    if isinstance(value, str) and value.strip():
        return value
    raise DatasetError(f"{_sample_id(row)}: 缺少 video_id")


def _answer(row: dict[str, Any]) -> str:
    answer = row.get("gt_answer") or row.get("answer")
    if isinstance(answer, str):
        return answer
    if isinstance(answer, dict):
        for key in ("text", "canonical", "value"):
            value = answer.get(key)
            if isinstance(value, str) and value.strip():
                return value
    raise DatasetError(f"{_sample_id(row)}: 缺少 answer")


def _temporal_evidence(row: dict[str, Any]) -> list[list[float]]:
    for key in ("gt_temporal_evidence", "temporal_evidence"):
        value = row.get(key)
        if value is not None:
            return _intervals(value)
    evidence = row.get("evidence")
    if isinstance(evidence, dict) and evidence.get("segments") is not None:
        return _intervals(evidence["segments"])
    raise DatasetError(f"{_sample_id(row)}: 缺少 temporal evidence")


def _intervals(value: Any) -> list[list[float]]:
    if not isinstance(value, list):
        raise DatasetError("temporal evidence 必须是列表")
    intervals: list[list[float]] = []
    for item in value:
        if isinstance(item, dict):
            start = item.get("start_seconds", item.get("start"))
            end = item.get("end_seconds", item.get("end"))
        elif isinstance(item, list | tuple) and len(item) == 2:
            start, end = item
        else:
            raise DatasetError("temporal evidence 每项必须是区间")
        if not isinstance(start, int | float) or not isinstance(end, int | float):
            raise DatasetError("temporal evidence 边界必须是数值")
        intervals.append([float(start), float(end)])
    return intervals


def _media(row: dict[str, Any]) -> tuple[str | None, float | None]:
    media = row.get("media")
    media_path: str | None = None
    duration_seconds: float | None = None
    if isinstance(media, dict):
        value = media.get("path") or media.get("url")
        if isinstance(value, str) and value.strip():
            media_path = value
        duration = media.get("duration_seconds") or media.get("duration")
        if isinstance(duration, int | float):
            duration_seconds = float(duration)
    return media_path, duration_seconds


def _frames(
    media: dict[str, Any],
    *,
    sample_id: str,
    video_id: str,
) -> list[FrameRef]:
    frames = media.get("frames")
    if not isinstance(frames, list) or not frames:
        raise DatasetError(f"{sample_id}: spatial sample 没有 frames")
    result: list[FrameRef] = []
    for position, item in enumerate(frames):
        if not isinstance(item, dict):
            raise DatasetError(f"{sample_id}: frame 必须是 object")
        path = item.get("path")
        if not isinstance(path, str) or not path.strip():
            raise DatasetError(f"{sample_id}: frame 缺少 path")
        frame_index = item.get("frame_index", position)
        if not isinstance(frame_index, int):
            raise DatasetError(f"{sample_id}: frame_index 必须是整数")
        result.append(
            FrameRef(
                frame_id=str(item.get("frame_id") or frame_index),
                frame_index=frame_index,
                path=path,
                video_id=str(item.get("video_id") or video_id),
            )
        )
    return result


def _box_track(
    target: dict[str, Any],
    *,
    sample_id: str,
    video_id: str,
) -> list[BoxTrackItem]:
    value = target.get("box_track")
    if not isinstance(value, list) or not value:
        raise DatasetError(f"{sample_id}: 缺少 target.box_track")
    result: list[BoxTrackItem] = []
    for item in value:
        if not isinstance(item, dict):
            raise DatasetError(f"{sample_id}: box_track item 必须是 object")
        result.append(
            BoxTrackItem(
                frame_id=str(item.get("frame_id") or item.get("frame_index")),
                frame_index=_frame_index(item, sample_id=sample_id),
                video_id=str(item.get("video_id") or video_id),
                box=_numbers(item.get("box"), expected=4, label="box"),
                coordinate_space=str(item.get("coordinate_space") or "normalized_0_1"),
            )
        )
    return result


def _point_track(
    target: dict[str, Any],
    *,
    sample_id: str,
    video_id: str,
) -> list[PointTrackItem]:
    value = target.get("point_track")
    if not isinstance(value, list) or not value:
        raise DatasetError(f"{sample_id}: 缺少 target.point_track")
    result: list[PointTrackItem] = []
    for item in value:
        if not isinstance(item, dict):
            raise DatasetError(f"{sample_id}: point_track item 必须是 object")
        result.append(
            PointTrackItem(
                frame_id=str(item.get("frame_id") or item.get("frame_index")),
                frame_index=_frame_index(item, sample_id=sample_id),
                video_id=str(item.get("video_id") or video_id),
                point=_numbers(item.get("point"), expected=2, label="point"),
                coordinate_space=str(item.get("coordinate_space") or "normalized_0_1"),
            )
        )
    return result


def _frame_index(item: dict[str, Any], *, sample_id: str) -> int:
    frame_index = item.get("frame_index")
    if not isinstance(frame_index, int):
        raise DatasetError(f"{sample_id}: target frame_index 必须是整数")
    return frame_index


def _numbers(value: Any, *, expected: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != expected:
        raise DatasetError(f"{label} 必须包含 {expected} 个数值")
    result: list[float] = []
    for item in value:
        if not isinstance(item, int | float):
            raise DatasetError(f"{label} 坐标必须是数值")
        result.append(float(item))
    return result


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _target_ref(row: dict[str, Any]) -> str | None:
    target = row.get("target")
    if isinstance(target, dict):
        for key in ("text", "ref", "expression", "target_ref"):
            value = target.get(key)
            if isinstance(value, str) and value.strip():
                return value
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        semantic = metadata.get("semantic")
        if isinstance(semantic, dict):
            answer = semantic.get("answer")
            if isinstance(answer, dict):
                for key in ("canonical", "text", "value"):
                    value = answer.get(key)
                    if isinstance(value, str) and value.strip():
                        return value
    return None

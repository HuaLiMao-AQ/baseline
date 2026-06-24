"""模型输出解析。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ParsedOutput:
    """解析后的模型输出。"""

    data: dict[str, Any] | None
    parse_success: bool
    parse_error: str | None
    raw_output: str
    was_repaired: bool = False


@dataclass(frozen=True, slots=True)
class ParsedTemporalPrediction:
    """解析后的 temporal QA 预测。"""

    answer: str | None
    temporal_evidence: list[list[float]]
    parse_success: bool
    parse_error: str | None
    raw_output: str
    was_repaired: bool = False


@dataclass(frozen=True, slots=True)
class ParsedSpatialPrediction:
    """解析后的 spatial grounding 预测。"""

    target: str | None
    frame_index: int | None
    point: list[float] | None
    box: list[float] | None
    parse_success: bool
    parse_error: str | None
    raw_output: str
    was_repaired: bool = False


def parse_json_object(raw_output: str) -> ParsedOutput:
    """从模型输出中解析第一个 JSON object。

    Args:
        raw_output: 模型原始输出文本。

    Returns:
        解析结果；解析失败时保留错误原因和原文。
    """

    text = raw_output.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as first_error:
        candidate = _extract_first_json_object(text)
        if candidate is None:
            return ParsedOutput(
                data=None,
                parse_success=False,
                parse_error=f"json parse failed: {first_error.msg}",
                raw_output=raw_output,
            )
        try:
            repaired = json.loads(candidate)
        except json.JSONDecodeError as repair_error:
            return ParsedOutput(
                data=None,
                parse_success=False,
                parse_error=f"json repair failed: {repair_error.msg}",
                raw_output=raw_output,
            )
        if not isinstance(repaired, dict):
            return ParsedOutput(
                data=None,
                parse_success=False,
                parse_error="parsed JSON is not an object",
                raw_output=raw_output,
            )
        return ParsedOutput(
            data=repaired,
            parse_success=True,
            parse_error=None,
            raw_output=raw_output,
            was_repaired=True,
        )

    if not isinstance(data, dict):
        return ParsedOutput(
            data=None,
            parse_success=False,
            parse_error="parsed JSON is not an object",
            raw_output=raw_output,
        )
    return ParsedOutput(data=data, parse_success=True, parse_error=None, raw_output=raw_output)


def parse_temporal_prediction(
    raw_output: str,
    *,
    require_temporal_evidence: bool,
    duration_seconds: float | None = None,
) -> ParsedTemporalPrediction:
    """解析 temporal QA 模型输出。

    Args:
        raw_output: 模型原始输出。
        require_temporal_evidence: 是否要求 `temporal_evidence` 字段。
        duration_seconds: 可选视频时长，用于校验证据区间上界。

    Returns:
        结构化 temporal 预测；失败时保留原始输出和错误原因。
    """

    parsed = parse_json_object(raw_output)
    if not parsed.parse_success or parsed.data is None:
        return ParsedTemporalPrediction(
            answer=None,
            temporal_evidence=[],
            parse_success=False,
            parse_error=parsed.parse_error,
            raw_output=raw_output,
            was_repaired=parsed.was_repaired,
        )

    try:
        answer = _parse_answer(parsed.data)
        evidence = _parse_temporal_evidence(
            parsed.data,
            require_temporal_evidence=require_temporal_evidence,
            duration_seconds=duration_seconds,
        )
    except ValueError as exc:
        return ParsedTemporalPrediction(
            answer=None,
            temporal_evidence=[],
            parse_success=False,
            parse_error=str(exc),
            raw_output=raw_output,
            was_repaired=parsed.was_repaired,
        )

    return ParsedTemporalPrediction(
        answer=answer,
        temporal_evidence=evidence,
        parse_success=True,
        parse_error=None,
        raw_output=raw_output,
        was_repaired=parsed.was_repaired,
    )


def parse_spatial_prediction(
    raw_output: str,
    *,
    allowed_frame_indices: set[int] | None = None,
) -> ParsedSpatialPrediction:
    """解析 spatial grounding 模型输出。"""

    parsed = parse_json_object(raw_output)
    if not parsed.parse_success or parsed.data is None:
        return ParsedSpatialPrediction(
            target=None,
            frame_index=None,
            point=None,
            box=None,
            parse_success=False,
            parse_error=parsed.parse_error,
            raw_output=raw_output,
            was_repaired=parsed.was_repaired,
        )

    try:
        target, frame_index, point, box = _parse_spatial_payload(
            parsed.data,
            allowed_frame_indices=allowed_frame_indices,
        )
    except ValueError as exc:
        return ParsedSpatialPrediction(
            target=None,
            frame_index=None,
            point=None,
            box=None,
            parse_success=False,
            parse_error=str(exc),
            raw_output=raw_output,
            was_repaired=parsed.was_repaired,
        )

    return ParsedSpatialPrediction(
        target=target,
        frame_index=frame_index,
        point=point,
        box=box,
        parse_success=True,
        parse_error=None,
        raw_output=raw_output,
        was_repaired=parsed.was_repaired,
    )


def _extract_first_json_object(text: str) -> str | None:
    """提取第一个花括号平衡的 JSON object 文本。"""

    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _parse_answer(data: dict[str, Any]) -> str:
    value = data.get("answer")
    if value is None:
        raise ValueError("缺少 answer")
    answer = str(value).strip()
    if not answer:
        raise ValueError("answer 不能为空")
    return answer


def _parse_temporal_evidence(
    data: dict[str, Any],
    *,
    require_temporal_evidence: bool,
    duration_seconds: float | None,
) -> list[list[float]]:
    value = data.get("temporal_evidence")
    if value is None and not require_temporal_evidence:
        return []
    if not isinstance(value, list):
        raise ValueError("temporal_evidence 必须是列表")

    intervals: list[list[float]] = []
    for item in value:
        if not isinstance(item, list | tuple) or len(item) != 2:
            raise ValueError("temporal_evidence 每项必须是 [start, end]")
        start = float(item[0])
        end = float(item[1])
        if start < 0:
            raise ValueError("evidence start 不能为负")
        if end < start:
            raise ValueError("evidence end 不能小于 start")
        if duration_seconds is not None and end > duration_seconds:
            raise ValueError("evidence end 超过视频时长")
        intervals.append([start, end])
    return intervals


def _parse_spatial_payload(
    data: dict[str, Any],
    *,
    allowed_frame_indices: set[int] | None,
) -> tuple[str | None, int, list[float] | None, list[float] | None]:
    target_value = data.get("target") or data.get("answer")
    target = str(target_value).strip() if target_value is not None else None
    frame_index = _parse_frame_index(data.get("frame_index"))
    if allowed_frame_indices is not None and frame_index not in allowed_frame_indices:
        raise ValueError("frame_index 不在已提供帧中")

    point = _parse_coord_list(data.get("point"), expected=2, label="point")
    box_value = data.get("box")
    region = data.get("region")
    if box_value is None and isinstance(region, dict):
        if region.get("type") == "box":
            box_value = region.get("value")
    box = _parse_coord_list(box_value, expected=4, label="box")
    if point is None and box is None:
        raise ValueError("point 和 box 不能同时为空")
    if box is not None and (box[0] > box[2] or box[1] > box[3]):
        raise ValueError("box 必须满足 x1<=x2 且 y1<=y2")
    return target, frame_index, point, box


def _parse_frame_index(value: Any) -> int:
    if value is None:
        raise ValueError("缺少 frame_index")
    frame_index = int(value)
    if frame_index < 0:
        raise ValueError("frame_index 不能为负")
    return frame_index


def _parse_coord_list(value: Any, *, expected: int, label: str) -> list[float] | None:
    if value in (None, []):
        return None
    if not isinstance(value, list | tuple) or len(value) != expected:
        raise ValueError(f"{label} 必须包含 {expected} 个数值")
    coords = [float(item) for item in value]
    if any(coord < 0.0 or coord > 1.0 for coord in coords):
        raise ValueError(f"{label} 坐标必须归一化到 [0, 1]")
    return coords

"""模型 JSON 输出的鲁棒解析、修复与校验。"""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from typing import Any

CLIP_EPSILON_SECONDS = 1e-3
CLIP_EPSILON_COORD = 1e-4


@dataclass(frozen=True, slots=True)
class ParsedPrediction:
    """解析后的模型输出。

    Attributes:
        answer: 解析得到的答案；解析失败时为 ``None``。
        temporal_evidence: 解析并校验后的时间证据区间。
        parse_success: JSON 解析和字段校验是否成功。
        parse_error: 解析失败原因；成功时为 ``None``。
        was_repaired: 是否对模型输出做过安全修复或裁剪。
        raw_output: 原始模型输出，始终保留。
    """

    answer: str | None
    temporal_evidence: list[list[float]]
    parse_success: bool
    parse_error: str | None
    was_repaired: bool
    raw_output: str


# 新项目早期骨架使用的名称，保留为兼容别名。
ParsedTemporalPrediction = ParsedPrediction


@dataclass(frozen=True, slots=True)
class ParsedSpatialPrediction:
    """解析后的 spatial grounding 模型输出。"""

    target: str | None
    frame_index: int | None
    point: list[float] | None
    box: list[float] | None
    parse_success: bool
    parse_error: str | None
    was_repaired: bool
    raw_output: str


def parse_model_output(
    raw_output: str,
    *,
    duration_seconds: float | None,
    require_temporal_evidence: bool = True,
) -> ParsedPrediction:
    """把模型原始文本解析成答案和时间证据。

    Args:
        raw_output: 模型生成的原始文本。
        duration_seconds: 视频时长，用于校验证据区间边界。
        require_temporal_evidence: 是否要求输出 ``temporal_evidence`` 字段。

    Returns:
        包含解析结果、错误信息和原始文本的结构化对象。
    """

    last_error = "empty model output"
    validation_error: str | None = None
    for candidate, candidate_repaired in _candidate_texts(raw_output):
        for data, load_repaired, error in _load_jsonish(candidate):
            if error is not None:
                if validation_error is None:
                    last_error = error
                continue
            try:
                answer, intervals, validation_repaired = _validate_payload(
                    data,
                    duration_seconds=duration_seconds,
                    require_temporal_evidence=require_temporal_evidence,
                )
            except ValueError as exc:
                validation_error = str(exc)
                last_error = validation_error
                continue
            return ParsedPrediction(
                answer=answer,
                temporal_evidence=intervals,
                parse_success=True,
                parse_error=None,
                was_repaired=(
                    candidate_repaired or load_repaired or validation_repaired
                ),
                raw_output=raw_output,
            )

    return ParsedPrediction(
        answer=None,
        temporal_evidence=[],
        parse_success=False,
        parse_error=validation_error or last_error,
        was_repaired=False,
        raw_output=raw_output,
    )


def parse_spatial_model_output(
    raw_output: str,
    *,
    valid_frame_indices: set[int] | None = None,
) -> ParsedSpatialPrediction:
    """把模型原始文本解析成 frame index、point 和 box。"""

    last_error = "empty model output"
    validation_error: str | None = None
    for candidate, candidate_repaired in _candidate_texts(raw_output):
        for data, load_repaired, error in _load_jsonish(candidate):
            if error is not None:
                if validation_error is None:
                    last_error = error
                continue
            try:
                target, frame_index, point, box, validation_repaired = (
                    _validate_spatial_payload(
                        data,
                        valid_frame_indices=valid_frame_indices,
                    )
                )
            except ValueError as exc:
                validation_error = str(exc)
                last_error = validation_error
                continue
            return ParsedSpatialPrediction(
                target=target,
                frame_index=frame_index,
                point=point,
                box=box,
                parse_success=True,
                parse_error=None,
                was_repaired=(
                    candidate_repaired or load_repaired or validation_repaired
                ),
                raw_output=raw_output,
            )

    return ParsedSpatialPrediction(
        target=None,
        frame_index=None,
        point=None,
        box=None,
        parse_success=False,
        parse_error=validation_error or last_error,
        was_repaired=False,
        raw_output=raw_output,
    )


def parse_temporal_prediction(
    raw_output: str,
    *,
    require_temporal_evidence: bool,
    duration_seconds: float | None = None,
) -> ParsedPrediction:
    """兼容早期新项目骨架的 temporal parser 名称。"""

    return parse_model_output(
        raw_output,
        duration_seconds=duration_seconds,
        require_temporal_evidence=require_temporal_evidence,
    )


def parse_spatial_prediction(
    raw_output: str,
    *,
    allowed_frame_indices: set[int] | None = None,
) -> ParsedSpatialPrediction:
    """兼容早期新项目骨架的 spatial parser 名称。"""

    return parse_spatial_model_output(
        raw_output,
        valid_frame_indices=allowed_frame_indices,
    )


def _candidate_texts(raw_output: str) -> list[tuple[str, bool]]:
    raw = raw_output.strip()
    candidates: list[tuple[str, bool]] = []
    if raw:
        candidates.append((raw, False))

    for block in re.findall(r"```(?:json)?\s*(.*?)```", raw_output, flags=re.I | re.S):
        stripped = block.strip()
        if stripped:
            candidates.insert(0, (stripped, True))

    extracted = _extract_first_json_object(raw_output)
    if extracted and extracted != raw:
        candidates.append((extracted, True))

    deduped: list[tuple[str, bool]] = []
    seen: set[str] = set()
    for text, repaired in candidates:
        if text not in seen:
            deduped.append((text, repaired))
            seen.add(text)
    return deduped


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    quote_char = ""
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote_char:
                in_string = False
            continue
        if char in {'"', "'"}:
            in_string = True
            quote_char = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _load_jsonish(candidate: str) -> list[tuple[Any | None, bool, str | None]]:
    attempts: list[tuple[str, bool]] = [(candidate, False)]
    repaired = _remove_trailing_commas(candidate)
    if repaired != candidate:
        attempts.append((repaired, True))
    for base_text, _ in list(attempts):
        repaired_jsonish = _repair_common_jsonish_object(base_text)
        if repaired_jsonish != base_text:
            attempts.append((repaired_jsonish, True))
            repaired_jsonish = _remove_trailing_commas(repaired_jsonish)
            if repaired_jsonish != base_text:
                attempts.append((repaired_jsonish, True))

    results: list[tuple[Any | None, bool, str | None]] = []
    seen: set[str] = set()
    for text, was_repaired in attempts:
        if text in seen:
            continue
        seen.add(text)
        try:
            results.append((json.loads(text), was_repaired, None))
            continue
        except json.JSONDecodeError as exc:
            results.append((None, was_repaired, f"JSON decode failed: {exc.msg}"))
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError) as exc:
            results.append((None, True, f"literal parse failed: {exc}"))
            continue
        results.append((value, True, None))
    return results


def _remove_trailing_commas(text: str) -> str:
    """移除 JSON object/array 末尾多余逗号。"""

    return re.sub(r",\s*([}\]])", r"\1", text)


def _repair_common_jsonish_object(text: str) -> str:
    """Repair conservative pseudo-JSON often emitted by chat VLMs."""

    repaired = re.sub(
        r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:',
        r'\1"\2":',
        text,
    )
    return re.sub(
        r'("+(?:answer|target|ref)"+\s*:\s*)([^,\n}\]]+)',
        _quote_bare_string_value,
        repaired,
        flags=re.I,
    )


def _quote_bare_string_value(match: re.Match[str]) -> str:
    prefix = match.group(1)
    value = match.group(2).strip()
    if not value:
        return match.group(0)
    if value[0] in {'"', "'", "{", "["}:
        return match.group(0)
    if value.lower() in {"true", "false", "null", "none"}:
        return match.group(0)
    try:
        float(value)
    except ValueError:
        return f"{prefix}{json.dumps(value)}"
    return match.group(0)


def _validate_payload(
    data: Any,
    *,
    duration_seconds: float | None,
    require_temporal_evidence: bool,
) -> tuple[str, list[list[float]], bool]:
    if not isinstance(data, dict):
        raise ValueError("parsed output is not an object")
    answer = data.get("answer")
    if not isinstance(answer, str):
        raise ValueError("answer must be a string")
    evidence = data.get("temporal_evidence")
    if evidence is None and not require_temporal_evidence:
        return answer, [], False
    if not isinstance(evidence, list):
        raise ValueError("temporal_evidence must be a list")
    if duration_seconds is not None and duration_seconds < 0:
        raise ValueError("duration_seconds must be non-negative")

    intervals: list[list[float]] = []
    was_repaired = False
    for item in evidence:
        if not isinstance(item, list | tuple) or len(item) != 2:
            raise ValueError("each evidence interval must contain two numbers")
        start, end = item
        if not isinstance(start, int | float) or not isinstance(end, int | float):
            raise ValueError("each evidence interval must contain two numbers")
        start_f = float(start)
        end_f = float(end)
        if not math.isfinite(start_f) or not math.isfinite(end_f):
            raise ValueError("evidence interval bounds must be finite")

        clipped_start = _clip_small_bound(start_f, 0.0)
        if clipped_start != start_f:
            was_repaired = True
            start_f = clipped_start
        if duration_seconds is not None:
            clipped_end = _clip_small_bound(end_f, duration_seconds)
            if clipped_end != end_f:
                was_repaired = True
                end_f = clipped_end

        if start_f < 0:
            raise ValueError("evidence start must be >= 0")
        if duration_seconds is not None and end_f > duration_seconds:
            raise ValueError("evidence end exceeds video duration")
        if start_f > end_f:
            raise ValueError("evidence start must be <= end")
        intervals.append([start_f, end_f])
    return answer, intervals, was_repaired


def _validate_spatial_payload(
    data: Any,
    *,
    valid_frame_indices: set[int] | None,
) -> tuple[str | None, int, list[float] | None, list[float] | None, bool]:
    if not isinstance(data, dict):
        raise ValueError("parsed output is not an object")

    target = data.get("target") or data.get("answer") or data.get("ref")
    if target is not None and not isinstance(target, str):
        raise ValueError("target must be a string when present")

    frame_value = (
        data.get("frame_index")
        if "frame_index" in data
        else data.get("frame")
        if "frame" in data
        else data.get("image_index")
    )
    if not isinstance(frame_value, int):
        raise ValueError("frame_index must be an integer")
    if valid_frame_indices is not None and frame_value not in valid_frame_indices:
        raise ValueError("frame_index is not one of the supplied frames")

    point_value = data.get("point")
    box_value = data.get("box")
    region = data.get("region")
    if isinstance(region, dict):
        region_type = str(region.get("type") or "").lower()
        region_value = region.get("value")
        if region_type == "point" and point_value is None:
            point_value = region_value
        elif region_type == "box" and box_value is None:
            box_value = region_value

    point, point_repaired = _validate_coord_list(point_value, expected=2, label="point")
    box, box_repaired = _validate_coord_list(box_value, expected=4, label="box")
    was_repaired = point_repaired or box_repaired
    if point is None and box is None:
        raise ValueError("at least one of point or box is required")
    if box is not None:
        x1, y1, x2, y2 = box
        if x1 > x2 or y1 > y2:
            raise ValueError("box must be [x1, y1, x2, y2] with x1<=x2 and y1<=y2")
        if point is None:
            point = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
            was_repaired = True
    return target, frame_value, point, box, was_repaired


def _validate_coord_list(
    value: Any,
    *,
    expected: int,
    label: str,
) -> tuple[list[float] | None, bool]:
    if value is None:
        return None, False
    if not isinstance(value, list | tuple) or len(value) != expected:
        raise ValueError(f"{label} must contain {expected} numbers")
    result: list[float] = []
    was_repaired = False
    for item in value:
        if not isinstance(item, int | float):
            raise ValueError(f"{label} coordinates must be numeric")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{label} coordinates must be finite")
        clipped = _clip_small_coord(number)
        if clipped != number:
            was_repaired = True
            number = clipped
        if number < 0.0 or number > 1.0:
            raise ValueError(f"{label} coordinates must be normalized to [0, 1]")
        result.append(number)
    return result, was_repaired


def _clip_small_bound(value: float, boundary: float) -> float:
    if boundary == 0.0 and -CLIP_EPSILON_SECONDS <= value < 0:
        return 0.0
    if boundary > 0 and boundary < value <= boundary + CLIP_EPSILON_SECONDS:
        return boundary
    return value


def _clip_small_coord(value: float) -> float:
    if -CLIP_EPSILON_COORD <= value < 0.0:
        return 0.0
    if 1.0 < value <= 1.0 + CLIP_EPSILON_COORD:
        return 1.0
    return value

"""答案、时间证据和空间证据基线指标。"""

from __future__ import annotations

import re
import string
from collections import defaultdict
from typing import Any

Interval = list[float] | tuple[float, float]


def normalize_answer(text: str | None, *, remove_articles: bool = True) -> str:
    """归一化答案文本。

    Args:
        text: 原始答案文本。
        remove_articles: 是否移除英文冠词 ``a``、``an``、``the``。

    Returns:
        用于 EM 和 token F1 的归一化字符串。
    """

    if text is None:
        return ""
    value = text.lower().strip()
    value = re.sub(f"[{re.escape(string.punctuation)}]", " ", value)
    if remove_articles:
        value = re.sub(r"\b(a|an|the)\b", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def exact_match(prediction: str | None, ground_truth: str | None) -> float:
    """计算归一化 Exact Match。

    Args:
        prediction: 预测答案。
        ground_truth: 标准答案。

    Returns:
        完全匹配返回 ``1.0``，否则返回 ``0.0``。
    """

    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def token_f1(prediction: str | None, ground_truth: str | None) -> float:
    """计算归一化答案 token F1。

    Args:
        prediction: 预测答案。
        ground_truth: 标准答案。

    Returns:
        token 级 F1 分数。
    """

    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    gt_counts: dict[str, int] = {}
    for token in gt_tokens:
        gt_counts[token] = gt_counts.get(token, 0) + 1

    overlap = 0
    for token in pred_tokens:
        count = gt_counts.get(token, 0)
        if count > 0:
            overlap += 1
            gt_counts[token] = count - 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def temporal_iou(predicted: list[Interval], ground_truth: list[Interval]) -> float:
    """计算多区间集合 IoU。

    Args:
        predicted: 预测时间区间。
        ground_truth: 标准时间区间。

    Returns:
        先合并区间集合后得到的交并比。
    """

    pred_union = merge_intervals(predicted)
    gt_union = merge_intervals(ground_truth)
    pred_duration = intervals_duration(pred_union)
    gt_duration = intervals_duration(gt_union)
    if pred_duration == 0.0 and gt_duration == 0.0:
        return 1.0
    if pred_duration == 0.0 or gt_duration == 0.0:
        return 0.0

    intersection = intersection_duration(pred_union, gt_union)
    union = pred_duration + gt_duration - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def merge_intervals(intervals: list[Interval]) -> list[list[float]]:
    """合并重叠时间区间。

    Args:
        intervals: 原始时间区间列表。

    Returns:
        按起点排序且互不重叠的区间列表。
    """

    cleaned = sorted(
        (float(start), float(end))
        for start, end in intervals
        if float(end) >= float(start)
    )
    merged: list[list[float]] = []
    for start, end in cleaned:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def intervals_duration(intervals: list[Interval]) -> float:
    """计算区间总时长。

    Args:
        intervals: 时间区间列表。

    Returns:
        所有区间长度之和。
    """

    return sum(max(0.0, float(end) - float(start)) for start, end in intervals)


def intersection_duration(a: list[Interval], b: list[Interval]) -> float:
    """计算两个区间集合的交集时长。

    Args:
        a: 第一个区间集合。
        b: 第二个区间集合。

    Returns:
        两个集合重叠部分的总时长。
    """

    left = merge_intervals(a)
    right = merge_intervals(b)
    total = 0.0
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if end > start:
            total += end - start
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


def summarize_predictions(
    records: list[dict[str, Any]],
    *,
    cuda_peak_memory_bytes: int | None = None,
    include_groups: bool = True,
    include_temporal_metrics: bool = True,
) -> dict[str, Any]:
    """从逐条预测聚合基线指标。

    Args:
        records: ``predictions.jsonl`` 中的逐条预测记录。
        cuda_peak_memory_bytes: CUDA 峰值显存字节数。
        include_groups: 是否额外计算 ``source_dataset`` 分组指标。
        include_temporal_metrics: 纯回答阶段设为 ``False``。

    Returns:
        可写入 ``summary.json`` 的指标字典。
    """

    total = len(records)
    inference_success = [
        record for record in records if record.get("error") in (None, "")
    ]
    valid = [
        record
        for record in records
        if record.get("error") in (None, "") and record.get("parse_success") is True
    ]

    em_values = [
        exact_match(record.get("pred_answer"), record.get("gt_answer"))
        for record in valid
    ]
    f1_values = [
        token_f1(record.get("pred_answer"), record.get("gt_answer"))
        for record in valid
    ]
    iou_values = [
        temporal_iou(
            record.get("pred_temporal_evidence") or [],
            record.get("gt_temporal_evidence") or [],
        )
        for record in valid
    ] if include_temporal_metrics else []
    latencies = [
        float(record["latency_seconds"])
        for record in inference_success
        if isinstance(record.get("latency_seconds"), int | float)
    ]
    answer_accuracy = _mean(em_values)
    recall_at_iou_0_3 = (
        _mean([float(value >= 0.3) for value in iou_values])
        if include_temporal_metrics
        else None
    )
    recall_at_iou_0_5 = (
        _mean([float(value >= 0.5) for value in iou_values])
        if include_temporal_metrics
        else None
    )
    acc_correct_evidence_iou_0_3 = (
        _mean(
            [
                float(answer_ok == 1.0 and iou >= 0.3)
                for answer_ok, iou in zip(em_values, iou_values)
            ]
        )
        if include_temporal_metrics
        else None
    )
    acc_correct_evidence_iou_0_5 = (
        _mean(
            [
                float(answer_ok == 1.0 and iou >= 0.5)
                for answer_ok, iou in zip(em_values, iou_values)
            ]
        )
        if include_temporal_metrics
        else None
    )
    answer_evidence_gap_iou_0_5 = (
        answer_accuracy - acc_correct_evidence_iou_0_5
        if answer_accuracy is not None and acc_correct_evidence_iou_0_5 is not None
        else None
    )

    summary: dict[str, Any] = {
        "total_samples": total,
        "inference_success_count": len(inference_success),
        "inference_success_rate": _rate(len(inference_success), total),
        "parse_success_count": sum(1 for record in records if record.get("parse_success")),
        "parse_success_rate": _rate(
            sum(1 for record in records if record.get("parse_success")),
            total,
        ),
        "valid_prediction_count": len(valid),
        "answer_accuracy": answer_accuracy,
        "normalized_exact_match": answer_accuracy,
        "answer_token_f1": _mean(f1_values),
        "temporal_metrics_enabled": include_temporal_metrics,
        "temporal_evidence_iou": (
            _mean(iou_values) if include_temporal_metrics else None
        ),
        "recall_at_iou_0_3": recall_at_iou_0_3,
        "recall_at_iou_0_5": recall_at_iou_0_5,
        "acc_correct_evidence_iou_0_3": acc_correct_evidence_iou_0_3,
        "acc_correct_evidence_iou_0_5": acc_correct_evidence_iou_0_5,
        "answer_evidence_gap_iou_0_5": answer_evidence_gap_iou_0_5,
        "average_latency_seconds": _mean(latencies),
        "cuda_peak_memory_bytes": cuda_peak_memory_bytes,
    }

    if include_groups:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[str(record.get("source_dataset") or "unknown")].append(record)
        summary["by_source_dataset"] = {
            key: summarize_predictions(
                value,
                include_groups=False,
                include_temporal_metrics=include_temporal_metrics,
            )
            for key, value in sorted(grouped.items())
        }
    return summary


def summarize_temporal_predictions(
    records: list[dict[str, Any]],
    *,
    include_temporal_metrics: bool,
    cuda_peak_memory_bytes: int | None = None,
    include_groups: bool = True,
) -> dict[str, Any]:
    """兼容早期新项目骨架的 temporal summary 名称。"""

    return summarize_predictions(
        records,
        cuda_peak_memory_bytes=cuda_peak_memory_bytes,
        include_groups=include_groups,
        include_temporal_metrics=include_temporal_metrics,
    )


def point_in_box(point: list[float] | tuple[float, float], box: list[float]) -> float:
    """判断 normalized point 是否落在 normalized box 内。"""

    x, y = float(point[0]), float(point[1])
    x1, y1, x2, y2 = (float(value) for value in box)
    return float(x1 <= x <= x2 and y1 <= y <= y2)


def box_iou(predicted: list[float] | None, ground_truth: list[float] | None) -> float:
    """计算 normalized box IoU。"""

    if predicted is None or ground_truth is None:
        return 0.0
    px1, py1, px2, py2 = (float(value) for value in predicted)
    gx1, gy1, gx2, gy2 = (float(value) for value in ground_truth)
    inter_x1 = max(px1, gx1)
    inter_y1 = max(py1, gy1)
    inter_x2 = min(px2, gx2)
    inter_y2 = min(py2, gy2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    pred_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    gt_area = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
    union = pred_area + gt_area - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def summarize_spatial_predictions(
    records: list[dict[str, Any]],
    *,
    cuda_peak_memory_bytes: int | None = None,
    include_groups: bool = True,
) -> dict[str, Any]:
    """从逐条预测聚合空间定位基线指标。"""

    total = len(records)
    inference_success = [
        record for record in records if record.get("error") in (None, "")
    ]
    valid = [
        record
        for record in records
        if record.get("error") in (None, "") and record.get("parse_success") is True
    ]
    pointing_values: list[float] = []
    box_iou_values: list[float] = []
    gt_box_count = 0
    pred_box_count = 0
    pred_point_count = 0
    for record in valid:
        gt_box = _gt_box_for_record(record)
        if gt_box is None:
            continue
        gt_box_count += 1
        pred_point = _pred_point_for_record(record)
        pred_box = record.get("pred_box")
        if isinstance(pred_point, list) and len(pred_point) == 2:
            pred_point_count += 1
            pointing_values.append(point_in_box(pred_point, gt_box))
        else:
            pointing_values.append(0.0)
        if isinstance(pred_box, list) and len(pred_box) == 4:
            pred_box_count += 1
            box_iou_values.append(box_iou(pred_box, gt_box))
        else:
            box_iou_values.append(0.0)

    latencies = [
        float(record["latency_seconds"])
        for record in inference_success
        if isinstance(record.get("latency_seconds"), int | float)
    ]
    pointing_accuracy = _mean(pointing_values)
    mean_box_iou = _mean(box_iou_values)
    summary: dict[str, Any] = {
        "total_samples": total,
        "inference_success_count": len(inference_success),
        "inference_success_rate": _rate(len(inference_success), total),
        "parse_success_count": sum(1 for record in records if record.get("parse_success")),
        "parse_success_rate": _rate(
            sum(1 for record in records if record.get("parse_success")),
            total,
        ),
        "valid_prediction_count": len(valid),
        "answer_accuracy": None,
        "normalized_exact_match": None,
        "answer_token_f1": None,
        "spatial_metrics_enabled": True,
        "evidence_accuracy": pointing_accuracy,
        "spatial_evidence_accuracy": pointing_accuracy,
        "pointing_accuracy": pointing_accuracy,
        "spatial_box_iou": mean_box_iou,
        "box_iou": mean_box_iou,
        "recall_at_box_iou_0_3": (
            _mean([float(value >= 0.3) for value in box_iou_values])
            if box_iou_values
            else None
        ),
        "recall_at_box_iou_0_5": (
            _mean([float(value >= 0.5) for value in box_iou_values])
            if box_iou_values
            else None
        ),
        "gt_box_count": gt_box_count,
        "pred_point_count": pred_point_count,
        "pred_box_count": pred_box_count,
        "average_latency_seconds": _mean(latencies),
        "cuda_peak_memory_bytes": cuda_peak_memory_bytes,
    }

    if include_groups:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[str(record.get("source_dataset") or "unknown")].append(record)
        summary["by_source_dataset"] = {
            key: summarize_spatial_predictions(value, include_groups=False)
            for key, value in sorted(grouped.items())
        }
    return summary


def _pred_point_for_record(record: dict[str, Any]) -> list[float] | None:
    point = record.get("pred_point")
    if isinstance(point, list) and len(point) == 2:
        return [float(point[0]), float(point[1])]
    box = record.get("pred_box")
    if isinstance(box, list) and len(box) == 4:
        return [
            (float(box[0]) + float(box[2])) / 2.0,
            (float(box[1]) + float(box[3])) / 2.0,
        ]
    return None


def _gt_box_for_record(record: dict[str, Any]) -> list[float] | None:
    frame_index = record.get("pred_frame_index")
    if not isinstance(frame_index, int):
        return None
    track = record.get("gt_box_track")
    if not isinstance(track, list) or not track:
        return None
    item = _closest_track_item(track, frame_index)
    if not isinstance(item, dict):
        return None
    box = item.get("box")
    if not isinstance(box, list) or len(box) != 4:
        return None
    return [float(value) for value in box]


def _closest_track_item(track: list[Any], frame_index: int) -> dict[str, Any] | None:
    candidates = [item for item in track if isinstance(item, dict)]
    if not candidates:
        return None
    exact = [
        item for item in candidates if isinstance(item.get("frame_index"), int)
        and item.get("frame_index") == frame_index
    ]
    if exact:
        return exact[0]
    return min(
        candidates,
        key=lambda item: abs(
            int(item.get("frame_index"))
            if isinstance(item.get("frame_index"), int)
            else frame_index
        )
    )


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator

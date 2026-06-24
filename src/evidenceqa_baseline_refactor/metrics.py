"""答案、时间证据和空间证据指标。"""

from __future__ import annotations

import re
import string
from collections import defaultdict
from typing import Any

Interval = list[float] | tuple[float, float]


def normalize_answer(text: Any) -> str:
    """归一化答案文本，用于 EM 和 F1。"""

    if text is None:
        return ""
    value = str(text).lower().strip()
    value = re.sub(f"[{re.escape(string.punctuation)}]", " ", value)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def exact_match(prediction: Any, ground_truth: Any) -> float:
    """计算归一化 Exact Match。"""

    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def token_f1(prediction: Any, ground_truth: Any) -> float:
    """计算归一化 token F1。"""

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
    """计算时间区间集合 IoU。"""

    pred = merge_intervals(predicted)
    gt = merge_intervals(ground_truth)
    pred_duration = intervals_duration(pred)
    gt_duration = intervals_duration(gt)
    if pred_duration == 0.0 and gt_duration == 0.0:
        return 1.0
    if pred_duration == 0.0 or gt_duration == 0.0:
        return 0.0
    intersection = intersection_duration(pred, gt)
    union = pred_duration + gt_duration - intersection
    return intersection / union if union > 0 else 0.0


def merge_intervals(intervals: list[Interval]) -> list[list[float]]:
    """合并重叠时间区间。"""

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
    """计算区间总时长。"""

    return sum(max(0.0, float(end) - float(start)) for start, end in intervals)


def intersection_duration(left: list[Interval], right: list[Interval]) -> float:
    """计算两个时间区间集合的交集时长。"""

    left_merged = merge_intervals(left)
    right_merged = merge_intervals(right)
    i = 0
    j = 0
    total = 0.0
    while i < len(left_merged) and j < len(right_merged):
        start = max(left_merged[i][0], right_merged[j][0])
        end = min(left_merged[i][1], right_merged[j][1])
        if end > start:
            total += end - start
        if left_merged[i][1] < right_merged[j][1]:
            i += 1
        else:
            j += 1
    return total


def point_in_box(point: list[float], box: list[float]) -> float:
    """判断点是否位于 box 内。"""

    x, y = float(point[0]), float(point[1])
    x1, y1, x2, y2 = (float(value) for value in box)
    return float(x1 <= x <= x2 and y1 <= y <= y2)


def box_iou(predicted: list[float] | None, ground_truth: list[float] | None) -> float:
    """计算两个归一化 box 的 IoU。"""

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
    return inter_area / union if union > 0 else 0.0


def summarize_temporal_predictions(
    records: list[dict[str, Any]],
    *,
    include_temporal_metrics: bool,
    cuda_peak_memory_bytes: int | None = None,
    include_groups: bool = True,
) -> dict[str, Any]:
    """从逐条 temporal 预测聚合 summary 指标。"""

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
    iou_values = (
        [
            temporal_iou(
                record.get("pred_temporal_evidence") or [],
                record.get("gt_temporal_evidence") or [],
            )
            for record in valid
        ]
        if include_temporal_metrics
        else []
    )
    answer_accuracy = _mean(em_values)
    acc_ev_03 = _acc_correct_with_iou(em_values, iou_values, threshold=0.3)
    acc_ev_05 = _acc_correct_with_iou(em_values, iou_values, threshold=0.5)
    summary: dict[str, Any] = {
        "total_samples": total,
        "inference_success_count": len(inference_success),
        "inference_success_rate": _rate(len(inference_success), total),
        "parse_success_count": _parse_success_count(records),
        "parse_success_rate": _rate(_parse_success_count(records), total),
        "valid_prediction_count": len(valid),
        "answer_accuracy": answer_accuracy,
        "normalized_exact_match": answer_accuracy,
        "answer_token_f1": _mean(f1_values),
        "temporal_metrics_enabled": include_temporal_metrics,
        "temporal_evidence_iou": (
            _mean(iou_values) if include_temporal_metrics else None
        ),
        "recall_at_iou_0_3": _recall_at_iou(iou_values, threshold=0.3)
        if include_temporal_metrics
        else None,
        "recall_at_iou_0_5": _recall_at_iou(iou_values, threshold=0.5)
        if include_temporal_metrics
        else None,
        "acc_correct_evidence_iou_0_3": acc_ev_03
        if include_temporal_metrics
        else None,
        "acc_correct_evidence_iou_0_5": acc_ev_05
        if include_temporal_metrics
        else None,
        "answer_evidence_gap_iou_0_5": (
            answer_accuracy - acc_ev_05
            if include_temporal_metrics
            and answer_accuracy is not None
            and acc_ev_05 is not None
            else None
        ),
        "average_latency_seconds": _mean(_latencies(inference_success)),
        "cuda_peak_memory_bytes": cuda_peak_memory_bytes,
    }
    if include_groups:
        summary["by_source_dataset"] = {
            key: summarize_temporal_predictions(
                value,
                include_temporal_metrics=include_temporal_metrics,
                include_groups=False,
            )
            for key, value in _group_by_dataset(records).items()
        }
    return summary


def summarize_spatial_predictions(
    records: list[dict[str, Any]],
    *,
    cuda_peak_memory_bytes: int | None = None,
    include_groups: bool = True,
) -> dict[str, Any]:
    """从逐条 spatial 预测聚合 summary 指标。"""

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
    pred_point_count = 0
    pred_box_count = 0
    for record in valid:
        gt_box = _gt_box_for_record(record)
        if gt_box is None:
            continue
        gt_box_count += 1
        pred_point = _pred_point_for_record(record)
        if pred_point is not None:
            pred_point_count += 1
            pointing_values.append(point_in_box(pred_point, gt_box))
        else:
            pointing_values.append(0.0)
        pred_box = record.get("pred_box")
        if isinstance(pred_box, list) and len(pred_box) == 4:
            pred_box_count += 1
            box_iou_values.append(box_iou(pred_box, gt_box))
        else:
            box_iou_values.append(0.0)

    pointing_accuracy = _mean(pointing_values)
    mean_box_iou = _mean(box_iou_values)
    summary: dict[str, Any] = {
        "total_samples": total,
        "inference_success_count": len(inference_success),
        "inference_success_rate": _rate(len(inference_success), total),
        "parse_success_count": _parse_success_count(records),
        "parse_success_rate": _rate(_parse_success_count(records), total),
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
        "recall_at_box_iou_0_3": _recall_at_iou(box_iou_values, threshold=0.3),
        "recall_at_box_iou_0_5": _recall_at_iou(box_iou_values, threshold=0.5),
        "gt_box_count": gt_box_count,
        "pred_point_count": pred_point_count,
        "pred_box_count": pred_box_count,
        "average_latency_seconds": _mean(_latencies(inference_success)),
        "cuda_peak_memory_bytes": cuda_peak_memory_bytes,
    }
    if include_groups:
        summary["by_source_dataset"] = {
            key: summarize_spatial_predictions(value, include_groups=False)
            for key, value in _group_by_dataset(records).items()
        }
    return summary


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _rate(count: int, total: int) -> float | None:
    if total == 0:
        return None
    return count / total


def _parse_success_count(records: list[dict[str, Any]]) -> int:
    return sum(1 for record in records if record.get("parse_success") is True)


def _latencies(records: list[dict[str, Any]]) -> list[float]:
    return [
        float(record["latency_seconds"])
        for record in records
        if isinstance(record.get("latency_seconds"), int | float)
    ]


def _recall_at_iou(values: list[float], *, threshold: float) -> float | None:
    if not values:
        return None
    return _mean([float(value >= threshold) for value in values])


def _acc_correct_with_iou(
    em_values: list[float],
    iou_values: list[float],
    *,
    threshold: float,
) -> float | None:
    if not iou_values:
        return None
    return _mean(
        [
            float(answer_ok == 1.0 and iou >= threshold)
            for answer_ok, iou in zip(em_values, iou_values, strict=True)
        ]
    )


def _group_by_dataset(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("source_dataset") or "unknown")].append(record)
    return dict(sorted(grouped.items()))


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
    track = record.get("gt_box_track")
    if not isinstance(track, list) or not track:
        return None
    frame_index = record.get("pred_frame_index")
    candidates = [
        item
        for item in track
        if isinstance(item, dict)
        and (frame_index is None or item.get("frame_index") == frame_index)
    ]
    item = candidates[0] if candidates else track[0]
    if not isinstance(item, dict):
        return None
    box = item.get("box")
    if not isinstance(box, list) or len(box) != 4:
        return None
    return [float(value) for value in box]

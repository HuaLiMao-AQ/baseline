"""答案、时间证据和空间证据指标。"""

from __future__ import annotations

import re
import string
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


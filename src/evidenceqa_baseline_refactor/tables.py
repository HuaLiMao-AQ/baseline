"""baseline summary 表格导出。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

MODEL_LABELS = {
    "qwen2.5-vl-7b-instruct": "Qwen2.5-VL-7B",
    "llava-onevision-qwen2-7b-ov-hf": "LLaVA-OneVision-7B",
    "internvl2-5-8b": "InternVL2.5-8B",
}
TEMPORAL_STAGES = ("answer_only", "grounded")
SPATIAL_STAGE = "ref"


def export_metric_tables(root: Path, output_dir: Path) -> list[Path]:
    """导出 baseline 主指标表。

    Args:
        root: 包含 `suite_summary.json` 的 baseline 结果目录。
        output_dir: CSV 输出目录。

    Returns:
        已写出的 CSV 路径列表。
    """

    summary = _read_suite_summary(root)
    output_dir.mkdir(parents=True, exist_ok=True)

    temporal_path = output_dir / "temporal_main.csv"
    spatial_path = output_dir / "spatial_main.csv"
    write_csv(temporal_path, TEMPORAL_FIELDS, collect_temporal_rows(summary))
    write_csv(spatial_path, SPATIAL_FIELDS, collect_spatial_rows(summary))
    return [temporal_path, spatial_path]


def collect_temporal_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """从 suite summary 收集 answer-only 和 grounded 主指标。"""

    rows: list[dict[str, Any]] = []
    for model_slug, model_run in _iter_model_runs(summary):
        model_label = MODEL_LABELS.get(model_slug, model_slug)
        model_id = str(model_run.get("model") or "")
        stages = model_run.get("stages") or {}
        for stage in TEMPORAL_STAGES:
            payload = (stages.get(stage) or {}).get("summary_payload") or {}
            rows.append(
                {
                    "model": model_label,
                    "model_id": model_id,
                    "stage": stage,
                    "total_samples": payload.get("total_samples"),
                    "valid_prediction_count": payload.get("valid_prediction_count"),
                    "parse_success_rate": payload.get("parse_success_rate"),
                    "answer_accuracy": payload.get("answer_accuracy"),
                    "answer_token_f1": payload.get("answer_token_f1"),
                    "temporal_evidence_iou": payload.get("temporal_evidence_iou"),
                    "recall_at_iou_0_3": payload.get("recall_at_iou_0_3"),
                    "recall_at_iou_0_5": payload.get("recall_at_iou_0_5"),
                    "acc_correct_evidence_iou_0_5": payload.get(
                        "acc_correct_evidence_iou_0_5"
                    ),
                    "answer_evidence_gap_iou_0_5": payload.get(
                        "answer_evidence_gap_iou_0_5"
                    ),
                    "average_latency_seconds": payload.get("average_latency_seconds"),
                }
            )
    return rows


def collect_spatial_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """从 suite summary 收集 spatial/ref 主指标。"""

    rows: list[dict[str, Any]] = []
    for model_slug, model_run in _iter_model_runs(summary):
        model_label = MODEL_LABELS.get(model_slug, model_slug)
        model_id = str(model_run.get("model") or "")
        stages = model_run.get("stages") or {}
        payload = (stages.get(SPATIAL_STAGE) or {}).get("summary_payload") or {}
        rows.append(
            {
                "model": model_label,
                "model_id": model_id,
                "stage": SPATIAL_STAGE,
                "total_samples": payload.get("total_samples"),
                "valid_prediction_count": payload.get("valid_prediction_count"),
                "parse_success_rate": payload.get("parse_success_rate"),
                "pointing_accuracy": payload.get("pointing_accuracy"),
                "spatial_box_iou": payload.get("spatial_box_iou"),
                "recall_at_box_iou_0_3": payload.get("recall_at_box_iou_0_3"),
                "recall_at_box_iou_0_5": payload.get("recall_at_box_iou_0_5"),
                "average_latency_seconds": payload.get("average_latency_seconds"),
            }
        )
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """写出稳定字段顺序的 CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def _read_suite_summary(root: Path) -> dict[str, Any]:
    path = root / "suite_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} 不是 JSON object")
    return payload


def _iter_model_runs(summary: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    model_runs = summary.get("model_runs")
    if not isinstance(model_runs, dict):
        raise ValueError("suite_summary.json 缺少 model_runs")
    return [
        (slug, payload)
        for slug, payload in model_runs.items()
        if isinstance(payload, dict)
    ]


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


TEMPORAL_FIELDS = [
    "model",
    "model_id",
    "stage",
    "total_samples",
    "valid_prediction_count",
    "parse_success_rate",
    "answer_accuracy",
    "answer_token_f1",
    "temporal_evidence_iou",
    "recall_at_iou_0_3",
    "recall_at_iou_0_5",
    "acc_correct_evidence_iou_0_5",
    "answer_evidence_gap_iou_0_5",
    "average_latency_seconds",
]

SPATIAL_FIELDS = [
    "model",
    "model_id",
    "stage",
    "total_samples",
    "valid_prediction_count",
    "parse_success_rate",
    "pointing_accuracy",
    "spatial_box_iou",
    "recall_at_box_iou_0_3",
    "recall_at_box_iou_0_5",
    "average_latency_seconds",
]


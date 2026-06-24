"""grounded 阶段错误类型统计。"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .jsonl import read_jsonl
from .metrics import exact_match, temporal_iou
from .tables import MODEL_LABELS, write_csv

TAXONOMY_FIELDS = [
    "model",
    "source_dataset",
    "total_samples",
    "valid_prediction_count",
    "parse_fail",
    "inference_error",
    "answer_correct_evidence_correct",
    "answer_correct_evidence_wrong",
    "answer_wrong_evidence_correct",
    "answer_wrong_evidence_wrong",
    "answer_correct_evidence_correct_rate_total",
    "answer_correct_evidence_wrong_rate_total",
    "answer_wrong_evidence_correct_rate_total",
    "answer_wrong_evidence_wrong_rate_total",
    "parse_fail_rate_total",
]


def export_grounded_taxonomy(root: Path, output_path: Path) -> Path:
    """导出 grounded 阶段 A/E 错误类型表。

    Args:
        root: baseline 结果目录。
        output_path: CSV 输出路径。

    Returns:
        写出的 CSV 路径。
    """

    rows = collect_grounded_taxonomy(root)
    write_csv(output_path, TAXONOMY_FIELDS, rows)
    return output_path


def collect_grounded_taxonomy(root: Path) -> list[dict[str, Any]]:
    """收集所有模型 grounded 阶段错误类型。"""

    rows: list[dict[str, Any]] = []
    for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        predictions = model_dir / "grounded" / "predictions.jsonl"
        if not predictions.exists():
            continue
        model_label = MODEL_LABELS.get(model_dir.name, model_dir.name)
        records = read_jsonl(predictions)
        rows.extend(_taxonomy_rows_for_model(model_label, records))
    return rows


def _taxonomy_rows_for_model(
    model_label: str,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {"all": records}
    by_dataset: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_dataset[str(record.get("source_dataset") or "unknown")].append(record)
    groups.update(by_dataset)

    return [
        _summarize_group(model_label, source_dataset, group_records)
        for source_dataset, group_records in sorted(
            groups.items(), key=lambda item: (item[0] != "all", item[0])
        )
    ]


def _summarize_group(
    model_label: str,
    source_dataset: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    valid_count = 0
    for record in records:
        if record.get("error") not in (None, ""):
            counts["inference_error"] += 1
            continue
        if record.get("parse_success") is not True:
            counts["parse_fail"] += 1
            continue

        valid_count += 1
        answer_ok = exact_match(record.get("pred_answer"), record.get("gt_answer")) == 1.0
        evidence_ok = (
            temporal_iou(
                record.get("pred_temporal_evidence") or [],
                record.get("gt_temporal_evidence") or [],
            )
            >= 0.5
        )
        key = _taxonomy_key(answer_ok=answer_ok, evidence_ok=evidence_ok)
        counts[key] += 1

    total = len(records)
    return {
        "model": model_label,
        "source_dataset": source_dataset,
        "total_samples": total,
        "valid_prediction_count": valid_count,
        "parse_fail": counts["parse_fail"],
        "inference_error": counts["inference_error"],
        "answer_correct_evidence_correct": counts[
            "answer_correct_evidence_correct"
        ],
        "answer_correct_evidence_wrong": counts["answer_correct_evidence_wrong"],
        "answer_wrong_evidence_correct": counts["answer_wrong_evidence_correct"],
        "answer_wrong_evidence_wrong": counts["answer_wrong_evidence_wrong"],
        "answer_correct_evidence_correct_rate_total": _rate(
            counts["answer_correct_evidence_correct"], total
        ),
        "answer_correct_evidence_wrong_rate_total": _rate(
            counts["answer_correct_evidence_wrong"], total
        ),
        "answer_wrong_evidence_correct_rate_total": _rate(
            counts["answer_wrong_evidence_correct"], total
        ),
        "answer_wrong_evidence_wrong_rate_total": _rate(
            counts["answer_wrong_evidence_wrong"], total
        ),
        "parse_fail_rate_total": _rate(counts["parse_fail"], total),
    }


def _taxonomy_key(*, answer_ok: bool, evidence_ok: bool) -> str:
    if answer_ok and evidence_ok:
        return "answer_correct_evidence_correct"
    if answer_ok and not evidence_ok:
        return "answer_correct_evidence_wrong"
    if not answer_ok and evidence_ok:
        return "answer_wrong_evidence_correct"
    return "answer_wrong_evidence_wrong"


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


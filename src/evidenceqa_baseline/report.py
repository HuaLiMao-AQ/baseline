"""baseline 分析报告生成。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .tables import collect_spatial_rows, collect_temporal_rows
from .tables import export_metric_tables, read_suite_summary
from .taxonomy import collect_grounded_taxonomy, export_grounded_taxonomy


def write_analysis_report(root: Path, output_dir: Path) -> Path:
    """生成 Markdown 分析报告和配套 CSV。

    Args:
        root: baseline 结果目录。
        output_dir: 分析输出目录。

    Returns:
        Markdown 报告路径。
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    export_metric_tables(root, output_dir)
    export_grounded_taxonomy(root, output_dir / "grounded_error_taxonomy_iou_0_5.csv")

    summary = read_suite_summary(root)
    temporal_rows = collect_temporal_rows(summary)
    spatial_rows = collect_spatial_rows(summary)
    taxonomy_rows = [
        row for row in collect_grounded_taxonomy(root) if row["source_dataset"] == "all"
    ]

    report_path = output_dir / "analysis.md"
    report_path.write_text(
        _render_report(
            temporal_rows=temporal_rows,
            spatial_rows=spatial_rows,
            taxonomy_rows=taxonomy_rows,
        ),
        encoding="utf-8",
    )
    return report_path


def _render_report(
    *,
    temporal_rows: list[dict[str, Any]],
    spatial_rows: list[dict[str, Any]],
    taxonomy_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Baseline analysis",
        "",
        "本报告由 baseline artifact 自动生成，主要用于数据与代码交付检查。",
        "",
        "## Temporal metrics",
        "",
        _markdown_table(
            [
                "Model",
                "Stage",
                "Parse %",
                "Answer %",
                "F1 %",
                "Temporal IoU",
                "R@0.5 %",
                "Acc+Ev@0.5 %",
                "Gap@0.5 %",
            ],
            [
                [
                    row["model"],
                    row["stage"],
                    _pct(row["parse_success_rate"]),
                    _pct(row["answer_accuracy"]),
                    _pct(row["answer_token_f1"]),
                    _num(row["temporal_evidence_iou"]),
                    _pct(row["recall_at_iou_0_5"]),
                    _pct(row["acc_correct_evidence_iou_0_5"]),
                    _pct(row["answer_evidence_gap_iou_0_5"]),
                ]
                for row in temporal_rows
            ],
        ),
        "",
        "## Grounded taxonomy",
        "",
        _markdown_table(
            [
                "Model",
                "Total",
                "Valid",
                "Parse fail",
                "A+E+",
                "A+E-",
                "A-E+",
                "A-E-",
            ],
            [
                [
                    row["model"],
                    row["total_samples"],
                    row["valid_prediction_count"],
                    row["parse_fail"],
                    row["answer_correct_evidence_correct"],
                    row["answer_correct_evidence_wrong"],
                    row["answer_wrong_evidence_correct"],
                    row["answer_wrong_evidence_wrong"],
                ]
                for row in taxonomy_rows
            ],
        ),
        "",
        "## Spatial metrics",
        "",
        _markdown_table(
            [
                "Model",
                "Parse %",
                "Pointing %",
                "Box IoU",
                "Box R@0.3 %",
                "Box R@0.5 %",
                "Valid/Total",
            ],
            [
                [
                    row["model"],
                    _pct(row["parse_success_rate"]),
                    _pct(row["pointing_accuracy"]),
                    _num(row["spatial_box_iou"]),
                    _pct(row["recall_at_box_iou_0_3"]),
                    _pct(row["recall_at_box_iou_0_5"]),
                    f"{row['valid_prediction_count']}/{row['total_samples']}",
                ]
                for row in spatial_rows
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _pct(value: Any) -> str:
    if value is None or value == "":
        return "-"
    return f"{float(value) * 100:.1f}"


def _num(value: Any) -> str:
    if value is None or value == "":
        return "-"
    return f"{float(value):.3f}"

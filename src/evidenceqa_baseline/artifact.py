"""baseline 结果目录校验。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_STAGES = ("smoke", "answer_only", "grounded", "ref")


@dataclass(frozen=True, slots=True)
class ArtifactIssue:
    """Artifact 校验问题。"""

    severity: str
    path: str
    message: str


def validate_artifact(root: Path) -> list[ArtifactIssue]:
    """校验 baseline-all-models 风格的结果目录。

    Args:
        root: baseline 结果目录。

    Returns:
        校验问题列表；空列表表示没有发现阻断问题。
    """

    issues: list[ArtifactIssue] = []
    suite_summary = root / "suite_summary.json"
    if not suite_summary.exists():
        return [
            ArtifactIssue(
                severity="error",
                path=str(suite_summary),
                message="缺少 suite_summary.json",
            )
        ]

    payload = _read_json(suite_summary)
    if payload.get("status") != "complete":
        issues.append(
            ArtifactIssue("error", str(suite_summary), "suite status 不是 complete")
        )

    model_runs = payload.get("model_runs")
    if not isinstance(model_runs, dict) or not model_runs:
        issues.append(ArtifactIssue("error", str(suite_summary), "缺少 model_runs"))
        return issues

    for model_slug, model_run in sorted(model_runs.items()):
        stages = model_run.get("stages") if isinstance(model_run, dict) else None
        if not isinstance(stages, dict):
            issues.append(ArtifactIssue("error", model_slug, "缺少 stages"))
            continue
        for stage in REQUIRED_STAGES:
            stage_root = root / model_slug / stage
            if stage not in stages:
                issues.append(ArtifactIssue("error", str(stage_root), "suite 中缺少阶段"))
                continue
            issues.extend(_validate_stage(stage_root))
    return issues


def _validate_stage(stage_root: Path) -> list[ArtifactIssue]:
    """校验单个阶段输出。"""

    issues: list[ArtifactIssue] = []
    required_files = ("summary.json", "run_config.json", "predictions.jsonl", "run.log")
    for filename in required_files:
        path = stage_root / filename
        if not path.exists():
            issues.append(ArtifactIssue("error", str(path), "缺少阶段输出文件"))

    summary_path = stage_root / "summary.json"
    if not summary_path.exists():
        return issues
    summary = _read_json(summary_path)
    export = summary.get("export_completeness") or {}
    if export.get("incomplete_records") not in (0, None):
        issues.append(ArtifactIssue("error", str(summary_path), "存在导出字段缺失记录"))
    parse_rate = summary.get("parse_success_rate")
    if isinstance(parse_rate, int | float) and parse_rate < 0.5:
        issues.append(
            ArtifactIssue("warning", str(summary_path), "parse success rate 低于 0.5")
        )
    return issues


def _read_json(path: Path) -> dict[str, Any]:
    """读取 JSON object 文件。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} 不是 JSON object")
    return payload


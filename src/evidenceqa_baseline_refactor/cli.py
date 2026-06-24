"""命令行入口。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from . import __version__
from .artifact import validate_artifact
from .tables import export_metric_tables


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI parser。"""

    parser = argparse.ArgumentParser(
        prog="evidenceqa-baseline-refactor",
        description="Clean EvidenceQA baseline rewrite scaffold.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("version", help="显示项目版本")

    validate = subparsers.add_parser("validate-artifact", help="校验 baseline 结果目录")
    validate.add_argument("root", type=Path)
    validate.add_argument("--json", action="store_true", help="以 JSON 输出校验结果")

    export = subparsers.add_parser("export-tables", help="导出 baseline 主指标 CSV")
    export.add_argument("root", type=Path)
    export.add_argument("output_dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """执行 CLI。"""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "version":
        print(__version__)
        return 0
    if args.command == "validate-artifact":
        issues = validate_artifact(args.root)
        if args.json:
            print(
                json.dumps(
                    [asdict(issue) for issue in issues],
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            if not issues:
                print("artifact 校验通过")
            for issue in issues:
                print(f"{issue.severity}: {issue.path}: {issue.message}")
        return 1 if any(issue.severity == "error" for issue in issues) else 0
    if args.command == "export-tables":
        paths = export_metric_tables(args.root, args.output_dir)
        for path in paths:
            print(path)
        return 0
    parser.error(f"未知命令: {args.command}")
    return 2

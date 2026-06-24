"""Summarize baseline ``failed_samples.jsonl`` files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


DEFAULT_TOP_K = 8


def summarize_failure_tree(root: Path, *, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    """Summarize every ``failed_samples.jsonl`` below ``root``."""

    files = sorted(root.rglob("failed_samples.jsonl"))
    file_summaries = [
        summarize_failure_file(path, root=root, top_k=top_k) for path in files
    ]
    total_failed = sum(int(item["failed_records"]) for item in file_summaries)
    total_inference_errors = sum(
        int(item["inference_error_count"]) for item in file_summaries
    )
    total_parse_errors = sum(int(item["parse_error_count"]) for item in file_summaries)
    return {
        "root": str(root),
        "failed_sample_files": len(file_summaries),
        "failed_records": total_failed,
        "inference_error_count": total_inference_errors,
        "parse_error_count": total_parse_errors,
        "files": file_summaries,
    }


def summarize_failure_file(
    path: Path,
    *,
    root: Path | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Summarize one JSONL failure file."""

    parse_errors: Counter[str] = Counter()
    inference_errors: Counter[str] = Counter()
    prompt_modes: Counter[str] = Counter()
    models: Counter[str] = Counter()
    records = 0
    parse_error_count = 0
    inference_error_count = 0
    for record in _read_jsonl(path):
        records += 1
        model = record.get("model")
        if model:
            models[str(model)] += 1
        prompt_mode = record.get("prompt_mode")
        if prompt_mode:
            prompt_modes[str(prompt_mode)] += 1
        error = record.get("error")
        if error:
            inference_error_count += 1
            inference_errors[_shorten(str(error))] += 1
        parse_error = record.get("parse_error")
        if parse_error:
            parse_error_count += 1
            parse_errors[_shorten(str(parse_error))] += 1

    display_path = path.relative_to(root) if root is not None else path
    return {
        "path": str(display_path),
        "failed_records": records,
        "inference_error_count": inference_error_count,
        "parse_error_count": parse_error_count,
        "models": _top_counts(models, top_k),
        "prompt_modes": _top_counts(prompt_modes, top_k),
        "top_inference_errors": _top_counts(inference_errors, top_k),
        "top_parse_errors": _top_counts(parse_errors, top_k),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for failure summaries."""

    parser = argparse.ArgumentParser(
        prog="python -m evidenceqa_baseline.failure_report",
        description="Summarize failed_samples.jsonl files from a baseline run tree.",
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args(argv)
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    summary = summarize_failure_tree(args.root, top_k=args.top_k)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                rows.append(
                    {
                        "error": (
                            f"failed_samples JSONL decode failed at line "
                            f"{line_number}: {exc.msg}"
                        )
                    }
                )
                continue
            if isinstance(payload, dict):
                rows.append(payload)
            else:
                rows.append(
                    {
                        "error": (
                            f"failed_samples JSONL line {line_number} is not an object"
                        )
                    }
                )
    return rows


def _top_counts(counter: Counter[str], top_k: int) -> list[dict[str, Any]]:
    return [
        {"value": value, "count": count}
        for value, count in counter.most_common(top_k)
    ]


def _shorten(text: str, *, limit: int = 240) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[: limit - 3]}..."


if __name__ == "__main__":
    raise SystemExit(main())

"""JSONL 读写工具。"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件。

    Args:
        path: JSONL 文件路径。

    Returns:
        逐行 JSON object 列表。

    Raises:
        ValueError: 任意一行不是 JSON object。
    """

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} 不是 JSON object")
            rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """写入 JSONL 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """写入格式化 JSON 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


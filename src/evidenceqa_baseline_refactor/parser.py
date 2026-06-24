"""模型输出解析。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ParsedOutput:
    """解析后的模型输出。"""

    data: dict[str, Any] | None
    parse_success: bool
    parse_error: str | None
    raw_output: str
    was_repaired: bool = False


def parse_json_object(raw_output: str) -> ParsedOutput:
    """从模型输出中解析第一个 JSON object。

    Args:
        raw_output: 模型原始输出文本。

    Returns:
        解析结果；解析失败时保留错误原因和原文。
    """

    text = raw_output.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as first_error:
        candidate = _extract_first_json_object(text)
        if candidate is None:
            return ParsedOutput(
                data=None,
                parse_success=False,
                parse_error=f"json parse failed: {first_error.msg}",
                raw_output=raw_output,
            )
        try:
            repaired = json.loads(candidate)
        except json.JSONDecodeError as repair_error:
            return ParsedOutput(
                data=None,
                parse_success=False,
                parse_error=f"json repair failed: {repair_error.msg}",
                raw_output=raw_output,
            )
        if not isinstance(repaired, dict):
            return ParsedOutput(
                data=None,
                parse_success=False,
                parse_error="parsed JSON is not an object",
                raw_output=raw_output,
            )
        return ParsedOutput(
            data=repaired,
            parse_success=True,
            parse_error=None,
            raw_output=raw_output,
            was_repaired=True,
        )

    if not isinstance(data, dict):
        return ParsedOutput(
            data=None,
            parse_success=False,
            parse_error="parsed JSON is not an object",
            raw_output=raw_output,
        )
    return ParsedOutput(data=data, parse_success=True, parse_error=None, raw_output=raw_output)


def _extract_first_json_object(text: str) -> str | None:
    """提取第一个花括号平衡的 JSON object 文本。"""

    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


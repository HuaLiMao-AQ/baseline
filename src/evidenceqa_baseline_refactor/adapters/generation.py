"""Hugging Face 适配器共享的生成参数工具。"""

from __future__ import annotations

from typing import Any


def generation_token_kwargs(
    tokenizer_or_processor: Any,
    model: Any | None = None,
) -> dict[str, int]:
    tokenizer = getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    generation_config = getattr(model, "generation_config", None)
    if pad_token_id is None and generation_config is not None:
        pad_token_id = getattr(generation_config, "pad_token_id", None)
    if eos_token_id is None and generation_config is not None:
        eos_token_id = getattr(generation_config, "eos_token_id", None)

    if pad_token_id is None:
        pad_token_id = _first_token_id(eos_token_id)
    if pad_token_id is None:
        return {}
    return {"pad_token_id": int(pad_token_id)}


def _first_token_id(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, (list, tuple)) and value:
        first = value[0]
        if isinstance(first, int):
            return first
    return None

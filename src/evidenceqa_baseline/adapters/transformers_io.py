"""Transformers 模型加载和生成解码的共享工具。"""

from __future__ import annotations

from typing import Any


def load_pretrained_with_dtype(
    model_class: Any,
    model_id: str,
    *,
    dtype: Any,
    cache_kwargs: dict[str, str],
    low_cpu_mem_usage: bool = False,
    trust_remote_code: bool = False,
    extra_kwargs: dict[str, Any] | None = None,
) -> Any:
    """按当前 Transformers 版本兼容加载模型。

    Args:
        model_class: 具体模型类或 AutoModel 类。
        model_id: Hugging Face 模型 ID 或本地路径。
        dtype: 已由设备选择逻辑解析出的 torch dtype。
        cache_kwargs: HF 缓存目录参数。
        low_cpu_mem_usage: 是否启用低 CPU 内存加载。
        trust_remote_code: 是否允许远程模型代码。
        extra_kwargs: 额外传给 ``from_pretrained`` 的参数。

    Returns:
        已加载但尚未移动到目标设备的模型对象。
    """

    kwargs: dict[str, Any] = dict(cache_kwargs)
    if low_cpu_mem_usage:
        kwargs["low_cpu_mem_usage"] = True
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    try:
        return model_class.from_pretrained(model_id, dtype=dtype, **kwargs)
    except TypeError:
        return model_class.from_pretrained(model_id, torch_dtype=dtype, **kwargs)


def decode_generated_suffix(processor: Any, inputs: Any, generated: Any) -> str:
    """只解码模型新生成的 token。

    Args:
        processor: 对应模型的 processor。
        inputs: 送入模型的输入张量容器，必须包含 ``input_ids``。
        generated: ``model.generate`` 的完整输出 token。

    Returns:
        解码后的第一条文本；没有输出时返回空字符串。
    """

    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated)
    ]
    decoded = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return decoded[0] if decoded else ""

"""InternVL 直接生成路径的 prompt 构造和解码工具。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

IMAGE_PLACEHOLDER = "<image>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMG_END_TOKEN = "</img>"
IMG_START_TOKEN = "<img>"


class InternVLGenerateError(RuntimeError):
    """InternVL prompt 构造或生成失败时抛出。"""


@dataclass(frozen=True, slots=True)
class InternVLPrompt:
    query: str
    stop_text: str
    response_prefix: str


def generate_internvl_response(
    *,
    model: Any,
    tokenizer: Any,
    pixel_values: Any | None,
    question: str,
    generation_config: dict[str, Any],
    num_patches_list: list[int] | None = None,
    response_prefix: str = "",
) -> str:
    """不依赖远程 chat/generate 方法生成单条 InternVL 响应。"""

    prompt = build_internvl_prompt(
        model=model,
        tokenizer=tokenizer,
        pixel_values=pixel_values,
        question=question,
        num_patches_list=num_patches_list,
        response_prefix=response_prefix,
    )
    model_inputs = tokenizer(prompt.query, return_tensors="pt")
    device = _model_device(model, pixel_values)
    input_ids = model_inputs["input_ids"].to(device)
    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    config = dict(generation_config)
    stop_token_id = _single_token_id_from_token(tokenizer, prompt.stop_text)
    if stop_token_id is not None:
        config["eos_token_id"] = _merge_token_ids(
            config.get("eos_token_id"),
            stop_token_id,
        )
    config["suppress_token_ids"] = _suppressed_special_token_ids(
        tokenizer,
        existing=config.get("suppress_token_ids"),
        allowed=_token_id_set(config.get("eos_token_id")),
    )

    generation_ids = generate_internvl_tokens(
        model=model,
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=config,
    )
    response = _decode_generated_response(tokenizer, generation_ids)
    response = f"{prompt.response_prefix}{response}"
    return response.split(prompt.stop_text)[0].strip()


def build_internvl_prompt(
    *,
    model: Any,
    tokenizer: Any,
    pixel_values: Any | None,
    question: str,
    num_patches_list: list[int] | None,
    response_prefix: str = "",
) -> InternVLPrompt:
    if pixel_values is not None and IMAGE_PLACEHOLDER not in question:
        question = f"{IMAGE_PLACEHOLDER}\n{question}"

    patches = _normalise_num_patches(pixel_values, num_patches_list)
    _validate_patch_count(pixel_values, patches)
    model.img_context_token_id = _image_context_token_id(tokenizer)

    template = _conversation_template(model)
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)

    query = _expand_image_placeholders(
        template.get_prompt(),
        patches,
        num_image_token=_num_image_token(model),
    )
    response_prefix = str(response_prefix)
    if response_prefix:
        query = f"{query}{response_prefix}"
    return InternVLPrompt(
        query=query,
        stop_text=str(template.sep).strip(),
        response_prefix=response_prefix,
    )


def generate_internvl_tokens(
    *,
    model: Any,
    pixel_values: Any | None,
    input_ids: Any,
    attention_mask: Any | None,
    generation_config: dict[str, Any] | Any | None,
    visual_features: Any | None = None,
) -> Any:
    if not hasattr(model, "language_model"):
        raise InternVLGenerateError("InternVL 模型缺少 language_model")
    if getattr(model, "img_context_token_id", None) is None:
        raise InternVLGenerateError("InternVL 图像上下文 token 尚未初始化")

    input_embeds = _internvl_prompt_embeds(
        model,
        pixel_values=pixel_values,
        input_ids=input_ids,
        visual_features=visual_features,
    )
    return _decode_language_model(
        model.language_model,
        input_embeds=input_embeds,
        attention_mask=attention_mask,
        generation_config=generation_config,
    )


def _conversation_template(model: Any) -> Any:
    template = getattr(model, "conv_template", None)
    copy = getattr(template, "copy", None)
    if not callable(copy):
        raise InternVLGenerateError("InternVL 模型缺少 conversation template")

    template = copy()
    system_message = getattr(model, "system_message", None)
    if system_message is not None and hasattr(template, "system_message"):
        template.system_message = system_message
    if not callable(getattr(template, "append_message", None)):
        raise InternVLGenerateError("InternVL conversation template 无法追加消息")
    if not callable(getattr(template, "get_prompt", None)):
        raise InternVLGenerateError("InternVL conversation template 无法构造 prompt")
    if not getattr(template, "roles", None):
        raise InternVLGenerateError("InternVL conversation template 缺少 roles")
    if getattr(template, "sep", None) is None:
        raise InternVLGenerateError("InternVL conversation template 缺少停止文本")
    return template


def _expand_image_placeholders(
    query: str,
    num_patches_list: list[int],
    *,
    num_image_token: int,
) -> str:
    expanded = query
    for num_patches in num_patches_list:
        if num_patches <= 0:
            raise InternVLGenerateError("InternVL 图像 patch 数必须为正数")
        if IMAGE_PLACEHOLDER not in expanded:
            raise InternVLGenerateError("InternVL prompt 中图像槽位少于帧数")
        image_tokens = (
            IMG_START_TOKEN
            + IMG_CONTEXT_TOKEN * num_image_token * num_patches
            + IMG_END_TOKEN
        )
        expanded = expanded.replace(IMAGE_PLACEHOLDER, image_tokens, 1)

    if IMAGE_PLACEHOLDER in expanded:
        raise InternVLGenerateError("InternVL prompt 中图像槽位多于帧数")
    return expanded


def _internvl_prompt_embeds(
    model: Any,
    *,
    pixel_values: Any | None,
    input_ids: Any,
    visual_features: Any | None,
) -> Any:
    input_embeds = model.language_model.get_input_embeddings()(input_ids).clone()
    if pixel_values is None:
        return input_embeds

    if visual_features is not None:
        vit_embeds = visual_features
    else:
        vit_embeds = model.extract_feature(pixel_values)

    batch_size, sequence_length, hidden_size = input_embeds.shape
    flat_embeds = input_embeds.reshape(batch_size * sequence_length, hidden_size)
    flat_input_ids = input_ids.reshape(batch_size * sequence_length)
    selected = flat_input_ids == model.img_context_token_id
    selected_count = int(selected.sum().item())
    if selected_count <= 0:
        raise InternVLGenerateError("InternVL prompt 中没有图像上下文 token")

    flat_vit_embeds = vit_embeds.reshape(-1, hidden_size).to(
        device=flat_embeds.device,
        dtype=flat_embeds.dtype,
    )
    if flat_vit_embeds.shape[0] != selected_count:
        raise InternVLGenerateError(
            "InternVL 视觉 token 数与 prompt 图像 token 数不一致"
        )
    flat_embeds[selected] = flat_vit_embeds
    return flat_embeds.reshape(batch_size, sequence_length, hidden_size)


def _decode_language_model(
    language_model: Any,
    *,
    input_embeds: Any,
    attention_mask: Any | None,
    generation_config: dict[str, Any] | Any | None,
) -> Any:
    import torch

    max_new_tokens = int(_generation_value("max_new_tokens", generation_config, 128))
    min_new_tokens = int(_generation_value("min_new_tokens", generation_config, 0))
    do_sample = bool(_generation_value("do_sample", generation_config, False))
    temperature = float(_generation_value("temperature", generation_config, 1.0))
    eos_token_ids = _token_id_set(
        _generation_value("eos_token_id", generation_config, None)
    )
    pad_token_id = _first_token_id(
        _generation_value("pad_token_id", generation_config, None)
    )
    blocked_pad_token_ids = _blocked_pad_token_ids(eos_token_ids, pad_token_id)
    suppressed_token_ids = _token_id_set(
        _generation_value("suppress_token_ids", generation_config, None)
    )

    device = input_embeds.device
    batch_size = input_embeds.shape[0]
    if max_new_tokens <= 0:
        return torch.empty((batch_size, 0), dtype=torch.long, device=device)

    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, input_embeds.shape[1]),
            dtype=torch.long,
            device=device,
        )
    else:
        attention_mask = attention_mask.to(device=device)

    generated: list[Any] = []
    unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)
    past_key_values = None
    next_token = None

    for step in range(max_new_tokens):
        if step == 0:
            logits, past_key_values = _language_model_next_logits(
                language_model,
                input_embeds=input_embeds,
                input_ids=None,
                attention_mask=attention_mask,
                past_key_values=None,
            )
        else:
            logits, past_key_values = _language_model_next_logits(
                language_model,
                input_embeds=None,
                input_ids=next_token[:, None],
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

        if eos_token_ids and step < min_new_tokens:
            logits[:, sorted(eos_token_ids)] = -torch.inf
        blocked_token_ids = _valid_token_ids(
            suppressed_token_ids | blocked_pad_token_ids,
            vocab_size=logits.shape[-1],
        )
        if blocked_token_ids:
            logits[:, blocked_token_ids] = -torch.inf
        next_token = _select_next_token(
            logits,
            do_sample=do_sample,
            temperature=temperature,
            torch=torch,
        )
        if pad_token_id is not None:
            pad_tensor = torch.full_like(next_token, pad_token_id)
            next_token = torch.where(unfinished, next_token, pad_tensor)
        generated.append(next_token[:, None])

        if eos_token_ids:
            eos_tensor = torch.tensor(sorted(eos_token_ids), device=device)
            is_eos = (next_token[:, None] == eos_tensor[None, :]).any(dim=1)
            unfinished = unfinished & ~is_eos
            if not bool(unfinished.any()):
                break

        if past_key_values is None:
            raise InternVLGenerateError("InternVL 语言模型没有返回 cache")
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (batch_size, 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ],
            dim=1,
        )

    if not generated:
        return torch.empty((batch_size, 0), dtype=torch.long, device=device)
    return torch.cat(generated, dim=1)


def _language_model_next_logits(
    language_model: Any,
    *,
    input_embeds: Any | None,
    input_ids: Any | None,
    attention_mask: Any | None,
    past_key_values: Any | None,
) -> tuple[Any, Any | None]:
    decoder = getattr(language_model, "model", None)
    get_output_embeddings = getattr(language_model, "get_output_embeddings", None)
    output_embeddings = (
        get_output_embeddings() if callable(get_output_embeddings) else None
    )

    if callable(decoder) and output_embeddings is not None:
        outputs = decoder(
            input_ids=input_ids,
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        hidden_states = getattr(outputs, "last_hidden_state", None)
        if hidden_states is None:
            hidden_states = outputs[0]
        logits = output_embeddings(hidden_states[:, -1:, :]).float()
    else:
        outputs = language_model(
            input_ids=input_ids,
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        logits = outputs.logits[:, -1:, :].float()

    past = getattr(outputs, "past_key_values", None)
    if past is None and isinstance(outputs, (list, tuple)) and len(outputs) > 1:
        past = outputs[1]
    return logits[:, -1, :], past


def _generation_value(
    name: str,
    generation_config: dict[str, Any] | Any | None,
    default: Any,
) -> Any:
    if isinstance(generation_config, dict) and name in generation_config:
        return generation_config[name]
    if generation_config is not None and hasattr(generation_config, name):
        return getattr(generation_config, name)
    return default


def _token_id_set(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    if isinstance(value, (list, tuple, set)):
        return {int(item) for item in value if isinstance(item, int)}
    return set()


def _merge_token_ids(*values: Any) -> int | list[int] | None:
    token_ids: list[int] = []
    seen: set[int] = set()
    for value in values:
        for token_id in _token_id_set(value):
            if token_id not in seen:
                token_ids.append(token_id)
                seen.add(token_id)
    if not token_ids:
        return None
    if len(token_ids) == 1:
        return token_ids[0]
    return token_ids


def _blocked_pad_token_ids(
    eos_token_ids: set[int],
    pad_token_id: int | None,
) -> set[int]:
    if pad_token_id is None or pad_token_id in eos_token_ids:
        return set()
    return {pad_token_id}


def _suppressed_special_token_ids(
    tokenizer: Any,
    *,
    existing: Any,
    allowed: set[int],
) -> list[int]:
    token_ids = set(_token_id_set(existing))
    for token_id in getattr(tokenizer, "all_special_ids", []) or []:
        if isinstance(token_id, int) and token_id not in allowed:
            token_ids.add(int(token_id))
    return sorted(token_ids)


def _valid_token_ids(token_ids: set[int], *, vocab_size: int) -> list[int]:
    return sorted(token_id for token_id in token_ids if 0 <= token_id < vocab_size)


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


def _single_token_id_from_token(tokenizer: Any, token: str) -> int | None:
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        return None
    token_id = convert(token)
    if isinstance(token_id, int) and token_id >= 0:
        return token_id
    return None


def _image_context_token_id(tokenizer: Any) -> int:
    token_id = _single_token_id_from_token(tokenizer, IMG_CONTEXT_TOKEN)
    if token_id is None:
        raise InternVLGenerateError("InternVL tokenizer 缺少图像上下文 token")
    return token_id


def _num_image_token(model: Any) -> int:
    num_image_token = getattr(model, "num_image_token", None)
    if not isinstance(num_image_token, int) or num_image_token <= 0:
        raise InternVLGenerateError("InternVL 模型的 num_image_token 非法")
    return num_image_token


def _normalise_num_patches(
    pixel_values: Any | None,
    num_patches_list: list[int] | None,
) -> list[int]:
    if num_patches_list is not None:
        return [int(value) for value in num_patches_list]
    if pixel_values is None:
        return []
    return [int(pixel_values.shape[0])]


def _validate_patch_count(pixel_values: Any | None, num_patches_list: list[int]) -> None:
    if pixel_values is None:
        if num_patches_list:
            raise InternVLGenerateError("InternVL 在没有图像时收到了图像 patch 数")
        return
    if int(pixel_values.shape[0]) != sum(num_patches_list):
        raise InternVLGenerateError("InternVL 图像 patch 数与张量不匹配")


def _model_device(model: Any, pixel_values: Any | None) -> Any:
    if pixel_values is not None:
        return pixel_values.device
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def _batch_decode(tokenizer: Any, generation_ids: Any) -> list[str]:
    try:
        return tokenizer.batch_decode(
            generation_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.batch_decode(generation_ids, skip_special_tokens=True)


def _decode_generated_response(tokenizer: Any, generation_ids: Any) -> str:
    decoded = _batch_decode(tokenizer, generation_ids)[0]
    if decoded.strip():
        return decoded
    return _batch_decode_with_special_tokens(tokenizer, generation_ids)[0]


def _batch_decode_with_special_tokens(tokenizer: Any, generation_ids: Any) -> list[str]:
    try:
        return tokenizer.batch_decode(
            generation_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.batch_decode(generation_ids, skip_special_tokens=False)


def _select_next_token(
    logits: Any,
    *,
    do_sample: bool,
    temperature: float,
    torch: Any,
) -> Any:
    if not do_sample:
        return logits.argmax(dim=-1)
    if temperature > 0:
        logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(1)

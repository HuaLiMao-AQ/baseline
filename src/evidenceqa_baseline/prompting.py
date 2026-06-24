"""Video-LMM temporal QA 的集中提示词。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

PROMPT_MODE_ANSWER_ONLY = "answer_only"
PROMPT_MODE_GROUNDED = "grounded"
PROMPT_MODE_SPATIAL = "spatial"
PROMPT_MODES = (PROMPT_MODE_ANSWER_ONLY, PROMPT_MODE_GROUNDED)

ANSWER_ONLY_SYSTEM_PROMPT = """You are a video question-answering model.

Given a video and a question, answer using only information visible in the video.

Rules:
1. Give a concise answer.
2. Do not guess.
3. If the answer cannot be determined, return "unknown".
4. Return valid JSON only.

Output:
{
  "answer": "concise answer"
}"""

ANSWER_ONLY_USER_PROMPT_TEMPLATE = """Question:
{question}

Return JSON only:
{{
  "answer": "..."
}}"""

GROUNDED_SYSTEM_PROMPT = """You are a video question-answering and temporal grounding model.

Given a video and a question, answer using only information visible in the video. Then identify the minimum time interval or intervals that directly support the answer.

Rules:
1. Times are measured in seconds from the beginning of the video.
2. Each interval must be [start_time, end_time].
3. Use the smallest sufficient evidence intervals.
4. Multiple intervals are allowed.
5. Do not guess.
6. If the answer cannot be determined, return "unknown" and an empty evidence list.
7. Return valid JSON only.

Output:
{
  "answer": "concise answer",
  "temporal_evidence": [[start_time, end_time]]
}"""

GROUNDED_USER_PROMPT_TEMPLATE = """Video duration: {duration_seconds:g} seconds

Question:
{question}

Return JSON only:
{{
  "answer": "...",
  "temporal_evidence": [[start_time, end_time]]
}}"""

SPATIAL_SYSTEM_PROMPT = """You are a video spatial grounding model.

Given ordered video frames and a referring question, locate the referred object in one supplied frame.

Rules:
1. Use normalized coordinates in [0, 1], where [0, 0] is the top-left and [1, 1] is the bottom-right.
2. Every point and box value must be a decimal fraction between 0 and 1.
3. Never output pixel coordinates. Values like [384, 240] or [100, 100, 200, 200] are invalid.
4. Pick the supplied frame where the referred object is clearly visible.
5. Return one point inside the object and one tight box around the object.
6. Do not guess. If the object cannot be determined, return an empty point and box.
7. Return valid JSON only.

Output:
{
  "target": "referred object",
  "frame_index": 0,
  "point": [0.50, 0.50],
  "box": [0.10, 0.20, 0.80, 0.90]
}"""

SPATIAL_USER_PROMPT_TEMPLATE = """Supplied frame indices:
{frame_indices}

Question:
{question}

Return JSON only:
{{
  "target": "...",
  "frame_index": one supplied frame index,
  "point": [0.50, 0.50],
  "box": [0.10, 0.20, 0.80, 0.90]
}}"""


def build_user_prompt(
    question: str,
    duration_seconds: float,
    *,
    prompt_mode: str = PROMPT_MODE_GROUNDED,
) -> str:
    """构造单个样本的用户提示词。

    Args:
        question: 问题文本。
        duration_seconds: 视频时长秒数。
        prompt_mode: ``answer_only`` 或 ``grounded``。

    Returns:
        填充时长和问题后的用户提示词。
    """

    if prompt_mode == PROMPT_MODE_ANSWER_ONLY:
        return ANSWER_ONLY_USER_PROMPT_TEMPLATE.format(question=question)
    if prompt_mode != PROMPT_MODE_GROUNDED:
        raise ValueError(f"unsupported prompt_mode={prompt_mode!r}")
    return GROUNDED_USER_PROMPT_TEMPLATE.format(
        duration_seconds=duration_seconds,
        question=question,
    )


def build_qwen_messages(
    *,
    question: str,
    duration_seconds: float,
    media_path: Path,
    fps: float | None,
    max_frames: int,
    max_pixels: int | None,
    prompt_mode: str = PROMPT_MODE_GROUNDED,
) -> list[dict[str, Any]]:
    """构造 Qwen-VL chat messages。

    Args:
        question: 问题文本。
        duration_seconds: 视频时长秒数。
        media_path: 本地视频路径。
        fps: 可选采样 FPS。
        max_frames: 最大采样帧数。
        max_pixels: 可选视觉输入最大像素数。
        prompt_mode: ``answer_only`` 或 ``grounded``。

    Returns:
        可传给 Qwen-VL processor 的消息列表。
    """

    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"unsupported prompt_mode={prompt_mode!r}")
    video_payload: dict[str, Any] = {"type": "video", "video": _video_path(media_path)}
    if fps is not None:
        video_payload["fps"] = fps
    if max_frames > 0:
        video_payload["max_frames"] = max_frames
    if max_pixels is not None and max_pixels > 0:
        video_payload["max_pixels"] = max_pixels

    return [
        {
            "role": "system",
            "content": (
                ANSWER_ONLY_SYSTEM_PROMPT
                if prompt_mode == PROMPT_MODE_ANSWER_ONLY
                else GROUNDED_SYSTEM_PROMPT
            ),
        },
        {
            "role": "user",
            "content": [
                video_payload,
                {
                    "type": "text",
                    "text": build_user_prompt(
                        question,
                        duration_seconds,
                        prompt_mode=prompt_mode,
                    ),
                },
            ],
        },
    ]


def build_qwen_spatial_messages(
    *,
    question: str,
    frame_paths: list[tuple[int, Path]],
    max_pixels: int | None,
) -> list[dict[str, Any]]:
    """构造 Qwen-VL spatial grounding chat messages。"""

    if not frame_paths:
        raise ValueError("spatial prompt requires at least one frame")
    frame_indices = ", ".join(str(frame_index) for frame_index, _ in frame_paths)
    content: list[dict[str, Any]] = []
    for frame_index, path in frame_paths:
        content.append({"type": "text", "text": f"Frame index {frame_index}:"})
        payload: dict[str, Any] = {"type": "image", "image": _image_path(path)}
        if max_pixels is not None and max_pixels > 0:
            payload["max_pixels"] = max_pixels
        content.append(payload)
    content.append(
        {
            "type": "text",
            "text": SPATIAL_USER_PROMPT_TEMPLATE.format(
                frame_indices=frame_indices,
                question=question,
            ),
        }
    )
    return [
        {"role": "system", "content": SPATIAL_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def build_frame_temporal_prompt(
    *,
    question: str,
    duration_seconds: float,
    frame_context: str,
    prompt_mode: str = PROMPT_MODE_GROUNDED,
) -> str:
    """Build a text prompt for adapters that consume sampled frames as images."""

    if prompt_mode == PROMPT_MODE_ANSWER_ONLY:
        output_schema = '{\n  "answer": "..."\n}'
    elif prompt_mode == PROMPT_MODE_GROUNDED:
        output_schema = (
            '{\n  "answer": "...",\n'
            '  "temporal_evidence": [[start_time, end_time]]\n}'
        )
    else:
        raise ValueError(f"unsupported prompt_mode={prompt_mode!r}")

    return f"""You are given sampled frames from a video.

Video duration: {duration_seconds:g} seconds

Sampled frames:
{frame_context}

Question:
{question}

Rules:
1. Answer using only the visible frames.
2. Times are measured in seconds from the beginning of the video.
3. Replace start_time and end_time with numeric second values.
4. Do not copy the output schema or use placeholder names.
5. Return valid JSON only.
6. The first character of your response must be {{.
7. Do not include markdown fences, explanations, or extra text.

Return JSON only:
{output_schema}"""


def build_spatial_text_prompt(*, question: str, frame_context: str) -> str:
    """Build a text prompt for spatial adapters that consume frame images."""

    user_prompt = SPATIAL_USER_PROMPT_TEMPLATE.format(
        frame_indices=frame_context,
        question=question,
    )
    return f"{SPATIAL_SYSTEM_PROMPT}\n\n{user_prompt}"


def _video_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    return str(resolved)


def _image_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    return str(resolved)

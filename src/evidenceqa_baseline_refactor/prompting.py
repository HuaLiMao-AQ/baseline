"""baseline prompt 构造工具。"""

from __future__ import annotations

from dataclasses import dataclass

from evidenceqa_baseline_refactor.config import PromptMode

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

ANSWER_ONLY_USER_TEMPLATE = """Question:
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

GROUNDED_USER_TEMPLATE = """Video duration: {duration_seconds:g} seconds

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

SPATIAL_USER_TEMPLATE = """Supplied frame indices:
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


@dataclass(frozen=True, slots=True)
class PromptBundle:
    """系统 prompt 与用户 prompt 的组合。"""

    system: str
    user: str

    def as_text(self) -> str:
        """合并为纯文本 prompt，供非 chat 模型或轻量 adapter 使用。"""

        return f"{self.system}\n\n{self.user}"


def build_temporal_prompt(
    *,
    question: str,
    duration_seconds: float,
    prompt_mode: PromptMode,
) -> PromptBundle:
    """构造 temporal QA prompt。

    Args:
        question: 问题文本。
        duration_seconds: 视频时长秒数。
        prompt_mode: `answer_only` 或 `grounded`。

    Returns:
        系统和用户 prompt。
    """

    if prompt_mode == "answer_only":
        return PromptBundle(
            system=ANSWER_ONLY_SYSTEM_PROMPT,
            user=ANSWER_ONLY_USER_TEMPLATE.format(question=question),
        )
    if prompt_mode == "grounded":
        return PromptBundle(
            system=GROUNDED_SYSTEM_PROMPT,
            user=GROUNDED_USER_TEMPLATE.format(
                question=question,
                duration_seconds=duration_seconds,
            ),
        )
    raise ValueError(f"不支持 temporal prompt_mode: {prompt_mode}")


def build_spatial_prompt(
    *,
    question: str,
    frame_indices: list[int] | tuple[int, ...],
) -> PromptBundle:
    """构造 spatial grounding prompt。"""

    if not frame_indices:
        raise ValueError("spatial prompt 至少需要一个 frame index")
    frame_context = ", ".join(str(index) for index in frame_indices)
    return PromptBundle(
        system=SPATIAL_SYSTEM_PROMPT,
        user=SPATIAL_USER_TEMPLATE.format(
            question=question,
            frame_indices=frame_context,
        ),
    )

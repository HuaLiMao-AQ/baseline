"""面向图像序列 VLM 适配器的帧处理工具。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class FrameImage:
    """已抽样或已加载的单帧图像及其显示信息。"""

    frame_index: int
    label: str
    image: Any
    time_seconds: float | None = None


def sample_video_frames(
    media_path: Path,
    *,
    duration_seconds: float | None,
    max_frames: int,
) -> list[FrameImage]:
    """按均匀间隔从视频中抽样 PIL 图像帧。"""

    try:
        from decord import VideoReader, cpu
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "基于帧图像的适配器需要 decord 和 pillow；请安装 `.[vl]`。"
        ) from exc

    video_reader = VideoReader(str(media_path), ctx=cpu(0))
    total_frames = len(video_reader)
    if total_frames <= 0:
        raise RuntimeError("视频没有可读取帧")

    frame_count = total_frames if max_frames <= 0 else min(max_frames, total_frames)
    indices = _uniform_indices(total_frames, frame_count)
    batch = video_reader.get_batch(indices).asnumpy()
    fps = float(video_reader.get_avg_fps() or 0.0)

    frames: list[FrameImage] = []
    for index, array in zip(indices, batch):
        if fps > 0:
            time_seconds = float(index) / fps
        elif duration_seconds is not None and total_frames > 1:
            time_seconds = duration_seconds * float(index) / float(total_frames - 1)
        else:
            time_seconds = None
        label = f"帧 {len(frames)}"
        if time_seconds is not None:
            label += f" @ {time_seconds:.2f}s"
        frames.append(
            FrameImage(
                frame_index=int(index),
                label=label,
                image=Image.fromarray(array).convert("RGB"),
                time_seconds=time_seconds,
            )
        )
    return frames


def load_spatial_frames(frame_paths: list[tuple[int, Path]]) -> list[FrameImage]:
    """把空间定位任务的帧文件加载为 PIL 图像。"""

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "基于帧图像的适配器需要 pillow；请安装 `.[vl]`。"
        ) from exc

    frames: list[FrameImage] = []
    for frame_index, path in frame_paths:
        frames.append(
            FrameImage(
                frame_index=frame_index,
                label=f"帧索引 {frame_index}",
                image=Image.open(path).convert("RGB"),
                time_seconds=None,
            )
        )
    return frames


def frame_context(frames: list[FrameImage]) -> str:
    """返回用于 prompt 的紧凑帧标签列表。"""

    return "\n".join(frame.label for frame in frames)


def _uniform_indices(total_frames: int, frame_count: int) -> list[int]:
    if frame_count <= 1:
        return [0]
    step = (total_frames - 1) / float(frame_count - 1)
    return [
        min(total_frames - 1, max(0, round(index * step)))
        for index in range(frame_count)
    ]

"""EvidenceQA Core 数据读取、字段适配与可复现抽样。"""

from __future__ import annotations

import os
import random
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, TypeVar

from .cache import hf_hub_cache_dir
from .jsonl import read_jsonl

SampleMode = Literal["sequential", "random"]
DEFAULT_REPO_ID = "HuaLiMaoAQ/evidenceqa-core"
DEFAULT_REVISION = "main"
DEFAULT_SPLIT = "validation"
TEMPORAL_TASK_TYPE = "temporal_qa"
SPATIAL_TASK_TYPE = "spatial_grounding"
DEFAULT_LIMIT = 100
DEFAULT_SEED = 20260621
DEFAULT_SAMPLE_MODE: SampleMode = "random"
DEFAULT_PLATFORM = "huggingface_hub"
SampleT = TypeVar("SampleT")


class DatasetError(ValueError):
    """数据样本字段不满足 baseline 契约。"""


@dataclass(frozen=True, slots=True)
class EvidenceSample:
    """统一样本视图。

    Args:
        sample_id: 样本 ID。
        task_type: 任务类型。
        source_dataset: 来源数据集。
        raw: 原始 JSON 记录。
    """

    sample_id: str
    task_type: str
    source_dataset: str
    raw: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TemporalSample:
    """Runner 使用的 temporal QA 样本。"""

    sample_id: str
    video_id: str
    source_dataset: str
    source_split: str
    task_type: str
    question: str
    gt_answer: str
    gt_temporal_evidence: list[list[float]]
    media_path: str | None
    duration_seconds: float | None
    raw: dict[str, Any]


# 兼容原 baseline 的命名；新项目内部逐步使用更具体的 TemporalSample。
DatasetSample = TemporalSample


@dataclass(frozen=True, slots=True)
class FrameRef:
    """Runner 使用的 frame sequence 引用。"""

    frame_id: str
    frame_index: int
    path: str
    video_id: str


@dataclass(frozen=True, slots=True)
class BoxTrackItem:
    """单帧 normalized box 标注。"""

    frame_id: str
    frame_index: int
    video_id: str
    box: list[float]
    coordinate_space: str = "normalized_0_1"


@dataclass(frozen=True, slots=True)
class PointTrackItem:
    """单帧 normalized point 标注。"""

    frame_id: str
    frame_index: int
    video_id: str
    point: list[float]
    coordinate_space: str = "normalized_0_1"


@dataclass(frozen=True, slots=True)
class SpatialSample:
    """Runner 使用的 spatial grounding 样本。"""

    sample_id: str
    video_id: str
    source_dataset: str
    source_split: str
    task_type: str
    question: str
    frames: list[FrameRef]
    gt_box_track: list[BoxTrackItem]
    gt_point_track: list[PointTrackItem]
    reference_mask_path: str | None
    target_ref: str | None
    raw: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TemporalDatasetLoadResult:
    """一次 split 读取和 temporal QA 抽样的结果。

    Args:
        split_path: 本地 JSONL 路径，可以是缓存文件或本地 fixture。
        total_rows: split 中读到的总行数。
        temporal_rows: 筛选出的 temporal QA 行数。
        selected_samples: 最终进入本次运行的样本。
    """

    split_path: Path
    total_rows: int
    temporal_rows: int
    selected_samples: list[TemporalSample]


@dataclass(frozen=True, slots=True)
class SpatialDatasetLoadResult:
    """一次 split 读取和 spatial grounding 抽样的结果。"""

    split_path: Path
    total_rows: int
    spatial_rows: int
    selected_samples: list[SpatialSample]


def hf_resolve_url(repo_id: str, revision: str, path: str) -> str:
    """构造 Hugging Face dataset 文件的直接 resolve URL。

    Args:
        repo_id: Hugging Face dataset repo ID。
        revision: 固定 revision 或 tag。
        path: repo 内的相对文件路径。

    Returns:
        可直接下载该文件的 HTTPS URL。
    """

    quoted_path = urllib.parse.quote(path.lstrip("/"), safe="/")
    quoted_revision = urllib.parse.quote(revision, safe="")
    return (
        "https://huggingface.co/datasets/"
        f"{repo_id}/resolve/{quoted_revision}/{quoted_path}"
    )


def hf_auth_headers() -> dict[str, str]:
    """返回 Hugging Face 下载请求鉴权 header。

    ``huggingface_hub`` 会优先读取本机 login 状态；这里保留 HTTPS fallback，
    让没有安装 SDK 的环境仍然能下载公开数据或使用 ``HF_TOKEN``。
    """

    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def cached_split_path(
    *,
    repo_id: str,
    revision: str,
    split: str,
    cache_dir: Path,
) -> Path:
    """返回 split JSONL 的本地缓存路径。"""

    safe_repo = repo_id.replace("/", "--")
    safe_revision = revision.replace("/", "--")
    return cache_dir / "hf" / "datasets" / safe_repo / safe_revision / f"{split}.jsonl"


def download_hf_dataset_file(
    *,
    repo_id: str,
    revision: str,
    file_path: str,
    cache_dir: Path,
    target_path: Path,
    force: bool = False,
) -> Path:
    """下载 Hugging Face dataset 中的单个文件。

    Args:
        repo_id: Hugging Face dataset repo ID。
        revision: 固定 revision 或 tag。
        file_path: repo 内的相对文件路径。
        cache_dir: baseline 缓存根目录。
        target_path: 无 SDK fallback 时写入的本地缓存路径。
        force: 是否忽略现有缓存并重新下载。

    Returns:
        下载后或已存在的本地文件路径。若使用 Hub SDK，可能返回 SDK 缓存路径。

    Raises:
        DatasetError: 网络、鉴权或下载失败时抛出。
    """

    if target_path.exists() and target_path.stat().st_size > 0 and not force:
        return target_path

    sdk_path = _download_hf_dataset_file_with_hub(
        repo_id=repo_id,
        revision=revision,
        file_path=file_path,
        cache_dir=cache_dir,
    )
    if sdk_path is not None:
        return sdk_path

    url = hf_resolve_url(repo_id, revision, file_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers=hf_auth_headers())

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with tempfile.NamedTemporaryFile(
                "wb",
                delete=False,
                dir=target_path.parent,
                prefix=f".{target_path.name}.",
            ) as tmp:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    tmp.write(chunk)
                tmp_path = Path(tmp.name)
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise DatasetError(
                "Hugging Face 数据文件访问被拒绝；请先执行 `hf auth login`，"
                "或设置有读取权限的 HF_TOKEN。"
            ) from exc
        raise DatasetError(
            f"下载 Hugging Face 文件 {file_path!r} 失败: HTTP {exc.code}"
        ) from exc
    except OSError as exc:
        raise DatasetError(
            f"下载 Hugging Face 文件 {file_path!r} 失败: {exc}。"
            "当前环境无法访问 Hugging Face；请检查网络/代理，或先下载 split JSONL 后"
            "通过 --local-jsonl /path/to/validation.jsonl 运行。"
        ) from exc

    if tmp_path.stat().st_size <= 0:
        tmp_path.unlink(missing_ok=True)
        raise DatasetError(f"下载到的 Hugging Face 文件为空: {file_path}")
    tmp_path.replace(target_path)
    return target_path


def download_split_jsonl(
    *,
    repo_id: str,
    revision: str,
    split: str,
    cache_dir: Path,
    force: bool = False,
) -> Path:
    """只下载指定 split 的 JSONL 到本地缓存。"""

    return download_hf_dataset_file(
        repo_id=repo_id,
        revision=revision,
        file_path=f"{split}.jsonl",
        cache_dir=cache_dir,
        target_path=cached_split_path(
            repo_id=repo_id,
            revision=revision,
            split=split,
            cache_dir=cache_dir,
        ),
        force=force,
    )


def load_samples(path: Path) -> list[EvidenceSample]:
    """读取 EvidenceQA JSONL 并转换为轻量样本对象。"""

    samples: list[EvidenceSample] = []
    for row in read_jsonl(path):
        sample_id = str(row.get("id") or row.get("sample_id") or "")
        if not sample_id:
            raise ValueError("样本缺少 id")
        samples.append(
            EvidenceSample(
                sample_id=sample_id,
                task_type=str(row.get("task_type") or row.get("task") or "unknown"),
                source_dataset=str(row.get("source_dataset") or row.get("dataset") or "unknown"),
                raw=row,
            )
        )
    return samples


def adapt_temporal_sample(sample: EvidenceSample | dict[str, Any]) -> TemporalSample:
    """把原始样本适配成 temporal QA 契约。"""

    row = _sample_raw(sample)
    sample_id = _sample_id(row)
    question = _question(row, sample_id=sample_id)
    media_path, duration_seconds = _media(row)
    return TemporalSample(
        sample_id=sample_id,
        video_id=_video_id(row, media_path=media_path),
        source_dataset=_source_dataset(row),
        source_split=_source_split(row),
        task_type=str(row.get("task_type") or row.get("task") or TEMPORAL_TASK_TYPE),
        question=question,
        gt_answer=_answer(row),
        gt_temporal_evidence=_temporal_evidence(row),
        media_path=media_path,
        duration_seconds=duration_seconds,
        raw=row,
    )


def adapt_spatial_sample(sample: EvidenceSample | dict[str, Any]) -> SpatialSample:
    """把原始样本适配成 spatial grounding 契约。"""

    row = _sample_raw(sample)
    sample_id = _sample_id(row)
    question = _question(row, sample_id=sample_id)
    media = row.get("media")
    if not isinstance(media, dict):
        raise DatasetError(f"{sample_id}: 缺少 media object")
    video_id = _video_id(row, media_path=None)
    target = row.get("target")
    if not isinstance(target, dict):
        raise DatasetError(f"{sample_id}: 缺少 spatial target")
    return SpatialSample(
        sample_id=sample_id,
        video_id=video_id,
        source_dataset=_source_dataset(row),
        source_split=_source_split(row),
        task_type=str(row.get("task_type") or row.get("task") or SPATIAL_TASK_TYPE),
        question=question,
        frames=_frames(media, sample_id=sample_id, video_id=video_id),
        gt_box_track=_box_track(target, sample_id=sample_id, video_id=video_id),
        gt_point_track=_point_track(target, sample_id=sample_id, video_id=video_id),
        reference_mask_path=_optional_string(target.get("reference_mask_path")),
        target_ref=_target_ref(row),
        raw=row,
    )


def filter_temporal_samples(samples: list[EvidenceSample]) -> list[TemporalSample]:
    """筛选并适配 temporal QA 样本。"""

    return [
        adapt_temporal_sample(sample)
        for sample in samples
        if sample.task_type == TEMPORAL_TASK_TYPE
    ]


def filter_spatial_samples(samples: list[EvidenceSample]) -> list[SpatialSample]:
    """筛选并适配 spatial grounding 样本。"""

    return [
        adapt_spatial_sample(sample)
        for sample in samples
        if sample.task_type == SPATIAL_TASK_TYPE
    ]


def filter_by_task(samples: list[EvidenceSample], task_type: str) -> list[EvidenceSample]:
    """按任务类型筛选样本。"""

    return [sample for sample in samples if sample.task_type == task_type]


def filter_temporal_qa(
    rows: Iterable[dict[str, Any]],
    *,
    task_type: str = TEMPORAL_TASK_TYPE,
) -> list[TemporalSample]:
    """筛选 temporal QA 行并完成字段适配。"""

    samples: list[TemporalSample] = []
    for row in rows:
        row_task = row.get("task") or row.get("task_type")
        if row_task != task_type:
            continue
        samples.append(adapt_temporal_sample(row))
    return samples


def filter_spatial_grounding(
    rows: Iterable[dict[str, Any]],
    *,
    task_type: str = SPATIAL_TASK_TYPE,
) -> list[SpatialSample]:
    """筛选 spatial grounding 行并完成字段适配。"""

    samples: list[SpatialSample] = []
    for row in rows:
        row_task = row.get("task") or row.get("task_type")
        if row_task != task_type:
            continue
        samples.append(adapt_spatial_sample(row))
    return samples


def select_samples(
    samples: list[SampleT],
    *,
    limit: int | None,
    seed: int,
    mode: SampleMode,
) -> list[SampleT]:
    """稳定选择实验样本。"""

    if limit is None or limit >= len(samples):
        return list(samples)
    if limit < 0:
        raise ValueError("limit 不能为负数")
    if mode == "sequential":
        return list(samples[:limit])
    if mode == "random":
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(samples)), limit))
        return [samples[index] for index in indices]
    raise ValueError(f"未知 sample mode: {mode}")


def load_temporal_samples(
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
    split: str = DEFAULT_SPLIT,
    task_type: str = TEMPORAL_TASK_TYPE,
    limit: int | None = DEFAULT_LIMIT,
    seed: int = DEFAULT_SEED,
    sample_mode: SampleMode = DEFAULT_SAMPLE_MODE,
    cache_dir: Path = Path(".cache/evidenceqa-baseline"),
    local_jsonl: Path | None = None,
) -> TemporalDatasetLoadResult:
    """加载、筛选并抽样 temporal QA 数据。

    Args:
        repo_id: Hugging Face dataset repo ID。
        revision: 固定 revision 或 tag。
        split: split 名称。
        task_type: 需要筛选的任务类型。
        limit: 最大样本数。
        seed: 随机抽样 seed。
        sample_mode: ``random`` 或 ``sequential``。
        cache_dir: 下载缓存根目录。
        local_jsonl: 可选本地 JSONL，测试和 smoke run 使用。

    Returns:
        split 读取、筛选和抽样结果。
    """

    split_path = (
        local_jsonl
        if local_jsonl is not None
        else download_split_jsonl(
            repo_id=repo_id,
            revision=revision,
            split=split,
            cache_dir=cache_dir,
        )
    )
    rows = read_jsonl(split_path)
    temporal_samples = filter_temporal_qa(rows, task_type=task_type)
    if local_jsonl is not None:
        temporal_samples = _resolve_local_media_paths(temporal_samples, local_jsonl)
    selected = select_samples(
        temporal_samples,
        limit=limit,
        seed=seed,
        mode=sample_mode,
    )
    return TemporalDatasetLoadResult(
        split_path=split_path,
        total_rows=len(rows),
        temporal_rows=len(temporal_samples),
        selected_samples=selected,
    )


def load_spatial_samples(
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
    split: str = DEFAULT_SPLIT,
    task_type: str = SPATIAL_TASK_TYPE,
    limit: int | None = DEFAULT_LIMIT,
    seed: int = DEFAULT_SEED,
    sample_mode: SampleMode = DEFAULT_SAMPLE_MODE,
    cache_dir: Path = Path(".cache/evidenceqa-baseline"),
    local_jsonl: Path | None = None,
) -> SpatialDatasetLoadResult:
    """加载、筛选并抽样 spatial grounding 数据。"""

    split_path = (
        local_jsonl
        if local_jsonl is not None
        else download_split_jsonl(
            repo_id=repo_id,
            revision=revision,
            split=split,
            cache_dir=cache_dir,
        )
    )
    rows = read_jsonl(split_path)
    spatial_samples = filter_spatial_grounding(rows, task_type=task_type)
    if local_jsonl is not None:
        spatial_samples = _resolve_local_spatial_paths(spatial_samples, local_jsonl)
    selected = select_samples(
        spatial_samples,
        limit=limit,
        seed=seed,
        mode=sample_mode,
    )
    return SpatialDatasetLoadResult(
        split_path=split_path,
        total_rows=len(rows),
        spatial_rows=len(spatial_samples),
        selected_samples=selected,
    )


def _download_hf_dataset_file_with_hub(
    *,
    repo_id: str,
    revision: str,
    file_path: str,
    cache_dir: Path,
) -> Path | None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None

    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            repo_type="dataset",
            revision=revision,
            cache_dir=str(hf_hub_cache_dir(cache_dir)),
        )
    except Exception as exc:  # noqa: BLE001 - SDK 异常需要转成清晰 CLI 信息。
        raise DatasetError(
            f"Hugging Face Hub 下载 {repo_id}/{file_path!r} 失败: {exc}"
        ) from exc
    path = Path(downloaded)
    if path.exists() and path.stat().st_size > 0:
        return path
    raise DatasetError(
        f"Hugging Face Hub 返回了空文件或缺失文件: {file_path!r}: {path}"
    )


def _sample_raw(sample: EvidenceSample | dict[str, Any]) -> dict[str, Any]:
    if isinstance(sample, EvidenceSample):
        return sample.raw
    return sample


def _sample_id(row: dict[str, Any]) -> str:
    sample_id = str(row.get("id") or row.get("sample_id") or row.get("qa_id") or "")
    if not sample_id:
        raise DatasetError("样本缺少 id")
    return sample_id


def _question(row: dict[str, Any], *, sample_id: str) -> str:
    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        raise DatasetError(f"{sample_id}: 缺少 question")
    return question


def _source_dataset(row: dict[str, Any]) -> str:
    value = row.get("source_dataset") or row.get("dataset")
    if value is None and isinstance(row.get("source"), dict):
        value = row["source"].get("dataset")
    if isinstance(value, str) and value.strip():
        return value
    raise DatasetError("样本缺少 source_dataset")


def _source_split(row: dict[str, Any]) -> str:
    value = row.get("source_split") or row.get("split")
    if value is None and isinstance(row.get("source"), dict):
        value = row["source"].get("split")
    return value if isinstance(value, str) and value.strip() else "unknown"


def _video_id(row: dict[str, Any], *, media_path: str | None) -> str:
    value = row.get("video_id")
    media = row.get("media")
    if value is None and isinstance(media, dict):
        value = media.get("video_id")
        if value is None and isinstance(media.get("raw"), dict):
            value = media["raw"].get("id") or media["raw"].get("video_id")
    if value is None and media_path:
        value = Path(media_path).stem
    if isinstance(value, str) and value.strip():
        return value
    raise DatasetError(f"{_sample_id(row)}: 缺少 video_id")


def _answer(row: dict[str, Any]) -> str:
    answer = row.get("gt_answer") or row.get("answer")
    if isinstance(answer, str):
        return answer
    if isinstance(answer, dict):
        for key in ("text", "canonical", "value"):
            value = answer.get(key)
            if isinstance(value, str) and value.strip():
                return value
    raise DatasetError(f"{_sample_id(row)}: 缺少 answer")


def _temporal_evidence(row: dict[str, Any]) -> list[list[float]]:
    for key in ("gt_temporal_evidence", "temporal_evidence"):
        value = row.get(key)
        if value is not None:
            return _intervals(value)
    evidence = row.get("evidence")
    if isinstance(evidence, dict) and evidence.get("segments") is not None:
        return _intervals(evidence["segments"])
    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, dict) and item.get("type") == "temporal_segments":
                return _intervals(item.get("segments", []))
    raise DatasetError(f"{_sample_id(row)}: 缺少 temporal evidence")


def _intervals(value: Any) -> list[list[float]]:
    if not isinstance(value, list):
        raise DatasetError("temporal evidence 必须是列表")
    intervals: list[list[float]] = []
    for item in value:
        if isinstance(item, dict):
            start = item.get("start_seconds", item.get("start"))
            end = item.get("end_seconds", item.get("end"))
        elif isinstance(item, list | tuple) and len(item) == 2:
            start, end = item
        else:
            raise DatasetError("temporal evidence 每项必须是区间")
        if not isinstance(start, int | float) or not isinstance(end, int | float):
            raise DatasetError("temporal evidence 边界必须是数值")
        intervals.append([float(start), float(end)])
    return intervals


def _media(row: dict[str, Any]) -> tuple[str | None, float | None]:
    media = row.get("media")
    if isinstance(media, dict) and isinstance(media.get("raw"), dict):
        media = media["raw"]

    media_path: str | None = None
    duration_seconds: float | None = None
    if isinstance(media, dict):
        value = media.get("path") or media.get("url")
        if isinstance(value, str) and value.strip():
            media_path = value
        duration = media.get("duration_seconds") or media.get("duration")
        if isinstance(duration, int | float):
            duration_seconds = float(duration)
    return media_path, duration_seconds


def _frames(
    media: dict[str, Any],
    *,
    sample_id: str,
    video_id: str,
) -> list[FrameRef]:
    frames = media.get("frames")
    if not isinstance(frames, list) or not frames:
        raise DatasetError(f"{sample_id}: spatial sample 没有 frames")
    result: list[FrameRef] = []
    for position, item in enumerate(frames):
        if not isinstance(item, dict):
            raise DatasetError(f"{sample_id}: frame 必须是 object")
        path = item.get("path")
        if not isinstance(path, str) or not path.strip():
            raise DatasetError(f"{sample_id}: frame 缺少 path")
        frame_index = item.get("frame_index", position)
        if not isinstance(frame_index, int):
            raise DatasetError(f"{sample_id}: frame_index 必须是整数")
        result.append(
            FrameRef(
                frame_id=str(item.get("frame_id") or frame_index),
                frame_index=frame_index,
                path=path,
                video_id=str(item.get("video_id") or video_id),
            )
        )
    return result


def _box_track(
    target: dict[str, Any],
    *,
    sample_id: str,
    video_id: str,
) -> list[BoxTrackItem]:
    value = target.get("box_track")
    if not isinstance(value, list) or not value:
        raise DatasetError(f"{sample_id}: 缺少 target.box_track")
    result: list[BoxTrackItem] = []
    for item in value:
        if not isinstance(item, dict):
            raise DatasetError(f"{sample_id}: box_track item 必须是 object")
        result.append(
            BoxTrackItem(
                frame_id=str(item.get("frame_id") or item.get("frame_index")),
                frame_index=_frame_index(item, sample_id=sample_id),
                video_id=str(item.get("video_id") or video_id),
                box=_numbers(item.get("box"), expected=4, label="box"),
                coordinate_space=str(item.get("coordinate_space") or "normalized_0_1"),
            )
        )
    return result


def _point_track(
    target: dict[str, Any],
    *,
    sample_id: str,
    video_id: str,
) -> list[PointTrackItem]:
    value = target.get("point_track")
    if not isinstance(value, list) or not value:
        raise DatasetError(f"{sample_id}: 缺少 target.point_track")
    result: list[PointTrackItem] = []
    for item in value:
        if not isinstance(item, dict):
            raise DatasetError(f"{sample_id}: point_track item 必须是 object")
        result.append(
            PointTrackItem(
                frame_id=str(item.get("frame_id") or item.get("frame_index")),
                frame_index=_frame_index(item, sample_id=sample_id),
                video_id=str(item.get("video_id") or video_id),
                point=_numbers(item.get("point"), expected=2, label="point"),
                coordinate_space=str(item.get("coordinate_space") or "normalized_0_1"),
            )
        )
    return result


def _frame_index(item: dict[str, Any], *, sample_id: str) -> int:
    frame_index = item.get("frame_index")
    if not isinstance(frame_index, int):
        raise DatasetError(f"{sample_id}: target frame_index 必须是整数")
    return frame_index


def _numbers(value: Any, *, expected: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != expected:
        raise DatasetError(f"{label} 必须包含 {expected} 个数值")
    result: list[float] = []
    for item in value:
        if not isinstance(item, int | float):
            raise DatasetError(f"{label} 坐标必须是数值")
        result.append(float(item))
    return result


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _target_ref(row: dict[str, Any]) -> str | None:
    target = row.get("target")
    if isinstance(target, dict):
        for key in ("text", "ref", "expression", "target_ref"):
            value = target.get(key)
            if isinstance(value, str) and value.strip():
                return value
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        semantic = metadata.get("semantic")
        if isinstance(semantic, dict):
            answer = semantic.get("answer")
            if isinstance(answer, dict):
                for key in ("canonical", "text", "value"):
                    value = answer.get(key)
                    if isinstance(value, str) and value.strip():
                        return value
    return None


def _resolve_local_media_paths(
    samples: list[TemporalSample],
    local_jsonl: Path,
) -> list[TemporalSample]:
    """把本地 JSONL 中的相对媒体路径解析到 JSONL 所在目录。

    远程数据集里的媒体路径本来就是 repo-relative；但用户传入 ``--local-jsonl``
    时，相对路径通常是相对该 JSONL 文件所在的数据集根目录。如果仍按当前工作
    目录解析，真实运行会误以为本地文件不存在并触发远程下载。
    """

    base_dir = local_jsonl.resolve().parent
    resolved: list[TemporalSample] = []
    for sample in samples:
        media_path = sample.media_path
        if (
            media_path is None
            or media_path.startswith(("http://", "https://"))
            or Path(media_path).is_absolute()
            or Path(media_path).exists()
        ):
            resolved.append(sample)
            continue

        local_candidate = base_dir / media_path
        if local_candidate.exists():
            resolved.append(replace(sample, media_path=str(local_candidate)))
        else:
            resolved.append(sample)
    return resolved


def _resolve_local_spatial_paths(
    samples: list[SpatialSample],
    local_jsonl: Path,
) -> list[SpatialSample]:
    """把本地 JSONL 中的相对 frame/mask 路径解析到 JSONL 所在目录。"""

    base_dir = local_jsonl.resolve().parent
    resolved: list[SpatialSample] = []
    for sample in samples:
        frames = [
            replace(frame, path=_resolve_local_ref(frame.path, base_dir))
            for frame in sample.frames
        ]
        mask_path = (
            _resolve_local_ref(sample.reference_mask_path, base_dir)
            if sample.reference_mask_path
            else None
        )
        resolved.append(replace(sample, frames=frames, reference_mask_path=mask_path))
    return resolved


def _resolve_local_ref(media_ref: str, base_dir: Path) -> str:
    if (
        media_ref.startswith(("http://", "https://"))
        or Path(media_ref).is_absolute()
        or Path(media_ref).exists()
    ):
        return media_ref
    local_candidate = base_dir / media_ref
    if local_candidate.exists():
        return str(local_candidate)
    return media_ref

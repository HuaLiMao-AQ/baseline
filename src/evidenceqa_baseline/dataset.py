"""EvidenceQA Core 数据读取、字段适配与可复现抽样。"""

from __future__ import annotations

import json
import os
import random
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, TypeVar

from .cache import hf_hub_cache_dir

DEFAULT_REPO_ID = "evidence-video-reasoning/evidenceqa-core"
DEFAULT_REVISION = "main"
DEFAULT_SPLIT = "validation"
DEFAULT_TASK_TYPE = "temporal_qa"
SPATIAL_TASK_TYPE = "spatial_grounding"
DEFAULT_LIMIT = 100
DEFAULT_SEED = 20260621
DEFAULT_SAMPLE_MODE = "random"
DEFAULT_PLATFORM = "huggingface_hub"
SampleT = TypeVar("SampleT")


class DatasetError(RuntimeError):
    """split 加载或样本字段适配失败时抛出。"""


@dataclass(frozen=True, slots=True)
class DatasetSample:
    """Runner 使用的归一化 temporal QA 样本。

    Attributes:
        id: 样本唯一 ID。
        video_id: 视频唯一 ID；数据缺失时从媒体路径兜底推导。
        source_dataset: 样本来源数据集名称。
        source_split: 原始或发布 split。
        task_type: 任务类型，当前应为 ``temporal_qa``。
        question: 问题文本。
        gt_answer: 标准答案文本。
        gt_temporal_evidence: 标准时间证据区间列表。
        media_path: 数据集中引用的视频路径或 URL。
        duration_seconds: 视频时长；数据缺失时为 ``None``。
        raw: 原始 JSONL 行，便于调试与复查 schema。
    """

    id: str
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


@dataclass(frozen=True, slots=True)
class DatasetLoadResult:
    """一次 split 读取和抽样的结果。

    Attributes:
        split_path: 本地 JSONL 路径，可以是缓存文件或本地 fixture。
        total_rows: split 中读到的总行数。
        temporal_rows: 筛选出的 temporal QA 行数。
        selected_samples: 最终进入本次运行的样本。
    """

    split_path: Path
    total_rows: int
    temporal_rows: int
    selected_samples: list[DatasetSample]


@dataclass(frozen=True, slots=True)
class FrameRef:
    """Runner 使用的归一化 frame sequence 引用。"""

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
    """Runner 使用的归一化 spatial grounding 样本。"""

    id: str
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

    ``huggingface_hub`` 会优先读取本机 login 状态；该 header 只服务于无 SDK 的
    HTTPS fallback。
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
    """返回 split JSONL 的本地缓存路径。

    Args:
        repo_id: Hugging Face dataset repo ID。
        revision: 固定 revision 或 tag。
        split: split 名称，例如 ``validation``。
        cache_dir: baseline 缓存根目录。

    Returns:
        该 split JSONL 应写入或复用的本地路径。
    """

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
    headers = hf_auth_headers()
    request = urllib.request.Request(url, headers=headers)

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
                "Hugging Face access denied for dataset file. "
                "Run `hf auth login` or set HF_TOKEN with read access."
            ) from exc
        raise DatasetError(
            f"failed to download Hugging Face file {file_path!r}: HTTP {exc.code}"
        ) from exc
    except OSError as exc:
        raise DatasetError(
            f"failed to download Hugging Face file {file_path!r}: {exc}. "
            "当前环境无法访问 Hugging Face；请检查网络/代理，或先下载 split JSONL 后通过 "
            "--local-jsonl /path/to/validation.jsonl 运行。"
        ) from exc

    if tmp_path.stat().st_size <= 0:
        tmp_path.unlink(missing_ok=True)
        raise DatasetError(f"downloaded Hugging Face file is empty: {file_path}")
    tmp_path.replace(target_path)
    return target_path


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
    except Exception as exc:  # noqa: BLE001 - SDK error needs a clean CLI message.
        raise DatasetError(
            f"Hugging Face Hub failed to download {file_path!r} from {repo_id}: {exc}"
        ) from exc
    path = Path(downloaded)
    if path.exists() and path.stat().st_size > 0:
        return path
    raise DatasetError(
        f"Hugging Face Hub returned an empty or missing file for {file_path!r}: {path}"
    )


def download_split_jsonl(
    *,
    repo_id: str,
    revision: str,
    split: str,
    cache_dir: Path,
    force: bool = False,
) -> Path:
    """只下载指定 split 的 JSONL 到本地缓存。

    Args:
        repo_id: Hugging Face dataset repo ID。
        revision: 固定 revision 或 tag。
        split: split 名称，例如 ``validation``。
        cache_dir: baseline 缓存根目录。
        force: 是否忽略现有缓存并重新下载。

    Returns:
        下载后或已存在的本地 JSONL 路径。

    Raises:
        DatasetError: 网络、鉴权或下载失败时抛出。
    """

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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件。

    Args:
        path: JSONL 文件路径。

    Returns:
        每行 JSON object 组成的列表。

    Raises:
        DatasetError: 行格式不是 JSON object 或无法解析时抛出。
    """

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{line_number}: invalid JSONL row") from exc
            if not isinstance(row, dict):
                raise DatasetError(f"{path}:{line_number}: row must be an object")
            rows.append(row)
    return rows


def adapt_temporal_sample(row: dict[str, Any]) -> DatasetSample:
    """把冻结数据集行适配成最小样本契约。

    Args:
        row: 原始 JSONL 行。

    Returns:
        归一化后的 temporal QA 样本。

    Raises:
        DatasetError: 必需字段缺失或字段类型不符合预期时抛出。
    """

    task_type = str(row.get("task") or row.get("task_type") or "")
    sample_id = str(row.get("id") or row.get("qa_id") or "")
    if not sample_id:
        raise DatasetError("sample is missing id/qa_id")

    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        raise DatasetError(f"{sample_id}: missing question")

    source_dataset = _extract_source_dataset(row)
    source_split = _extract_source_split(row)
    gt_answer = _extract_answer(row)
    gt_temporal_evidence = _extract_temporal_evidence(row)
    media_path, duration_seconds = _extract_media(row)
    video_id = _extract_video_id(row, media_path)

    return DatasetSample(
        id=sample_id,
        video_id=video_id,
        source_dataset=source_dataset,
        source_split=source_split,
        task_type=task_type,
        question=question,
        gt_answer=gt_answer,
        gt_temporal_evidence=gt_temporal_evidence,
        media_path=media_path,
        duration_seconds=duration_seconds,
        raw=row,
    )


def adapt_spatial_sample(row: dict[str, Any]) -> SpatialSample:
    """把冻结数据集行适配成 spatial grounding 样本契约。"""

    task_type = str(row.get("task") or row.get("task_type") or "")
    sample_id = str(row.get("id") or row.get("qa_id") or "")
    if not sample_id:
        raise DatasetError("sample is missing id/qa_id")

    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        raise DatasetError(f"{sample_id}: missing question")

    source_dataset = _extract_source_dataset(row)
    source_split = _extract_source_split(row)
    media = row.get("media")
    if not isinstance(media, dict):
        raise DatasetError(f"{sample_id}: missing media object")
    video_id = _extract_video_id(row, None)
    frames = _extract_frames(media, sample_id=sample_id, video_id=video_id)
    target = row.get("target")
    if not isinstance(target, dict):
        raise DatasetError(f"{sample_id}: missing spatial target")
    box_track = _extract_box_track(target, sample_id=sample_id, video_id=video_id)
    point_track = _extract_point_track(target, sample_id=sample_id, video_id=video_id)
    reference_mask_path = target.get("reference_mask_path")
    if reference_mask_path is not None and not isinstance(reference_mask_path, str):
        raise DatasetError(f"{sample_id}: reference_mask_path must be a string")

    return SpatialSample(
        id=sample_id,
        video_id=video_id,
        source_dataset=source_dataset,
        source_split=source_split,
        task_type=task_type,
        question=question,
        frames=frames,
        gt_box_track=box_track,
        gt_point_track=point_track,
        reference_mask_path=reference_mask_path,
        target_ref=_extract_target_ref(row),
        raw=row,
    )


def filter_temporal_qa(
    rows: Iterable[dict[str, Any]],
    *,
    task_type: str = DEFAULT_TASK_TYPE,
) -> list[DatasetSample]:
    """筛选 temporal QA 行并完成字段适配。

    Args:
        rows: 原始 JSONL 行迭代器。
        task_type: 需要筛选的任务类型。

    Returns:
        已归一化的 temporal QA 样本列表。
    """

    samples: list[DatasetSample] = []
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
    sample_mode: str,
) -> list[SampleT]:
    """按固定规则选择可复现样本子集。

    Args:
        samples: 候选样本列表。
        limit: 最大样本数；为 ``None`` 时保留全部。
        seed: 随机抽样 seed。
        sample_mode: ``random`` 或 ``sequential``。

    Returns:
        最终样本列表。

    Raises:
        DatasetError: 参数非法时抛出。
    """

    if limit is not None and limit < 0:
        raise DatasetError("limit must be non-negative")
    if sample_mode not in {"random", "sequential"}:
        raise DatasetError("sample_mode must be 'random' or 'sequential'")

    if limit is None or limit >= len(samples):
        selected = list(samples)
    elif sample_mode == "sequential":
        selected = list(samples[:limit])
    else:
        rng = random.Random(seed)
        selected = rng.sample(samples, limit)
    return selected


def load_temporal_samples(
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
    split: str = DEFAULT_SPLIT,
    task_type: str = DEFAULT_TASK_TYPE,
    limit: int | None = DEFAULT_LIMIT,
    seed: int = DEFAULT_SEED,
    sample_mode: str = DEFAULT_SAMPLE_MODE,
    cache_dir: Path = Path(".cache/evidenceqa-baseline"),
    local_jsonl: Path | None = None,
) -> DatasetLoadResult:
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

    Raises:
        DatasetError: split 读取、下载或字段适配失败时抛出。
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
        sample_mode=sample_mode,
    )
    return DatasetLoadResult(
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
    sample_mode: str = DEFAULT_SAMPLE_MODE,
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
        sample_mode=sample_mode,
    )
    return SpatialDatasetLoadResult(
        split_path=split_path,
        total_rows=len(rows),
        spatial_rows=len(spatial_samples),
        selected_samples=selected,
    )


def _resolve_local_media_paths(
    samples: list[DatasetSample],
    local_jsonl: Path,
) -> list[DatasetSample]:
    """把本地 JSONL 中的相对媒体路径解析到 JSONL 所在目录。

    远程数据集里的媒体路径本来就是 repo-relative；但用户传入 ``--local-jsonl``
    时，相对路径通常是相对该 JSONL 文件所在的数据集根目录。如果仍按当前工作
    目录解析，真实运行会误以为本地文件不存在并触发远程下载。
    """

    base_dir = local_jsonl.resolve().parent
    resolved: list[DatasetSample] = []
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
        frames: list[FrameRef] = []
        for frame in sample.frames:
            frames.append(replace(frame, path=_resolve_local_ref(frame.path, base_dir)))
        mask_path = (
            _resolve_local_ref(sample.reference_mask_path, base_dir)
            if sample.reference_mask_path
            else None
        )
        resolved.append(
            replace(sample, frames=frames, reference_mask_path=mask_path)
        )
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


def _extract_source_dataset(row: dict[str, Any]) -> str:
    value = row.get("source_dataset") or row.get("dataset")
    if value is None and isinstance(row.get("source"), dict):
        value = row["source"].get("dataset")
    if not isinstance(value, str) or not value.strip():
        raise DatasetError("sample is missing source dataset")
    return value


def _extract_source_split(row: dict[str, Any]) -> str:
    value = row.get("source_split") or row.get("split")
    if value is None and isinstance(row.get("source"), dict):
        value = row["source"].get("split")
    if not isinstance(value, str) or not value.strip():
        return "unknown"
    return value


def _extract_video_id(row: dict[str, Any], media_path: str | None) -> str:
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
    raise DatasetError(f"{row.get('id') or row.get('qa_id')}: missing video_id")


def _extract_frames(
    media: dict[str, Any],
    *,
    sample_id: str,
    video_id: str,
) -> list[FrameRef]:
    frames = media.get("frames")
    if not isinstance(frames, list) or not frames:
        raise DatasetError(f"{sample_id}: spatial sample has no frames")
    result: list[FrameRef] = []
    for position, item in enumerate(frames):
        if not isinstance(item, dict):
            raise DatasetError(f"{sample_id}: frame item must be an object")
        path = item.get("path")
        if not isinstance(path, str) or not path.strip():
            raise DatasetError(f"{sample_id}: frame item is missing path")
        frame_index = item.get("frame_index", position)
        if not isinstance(frame_index, int):
            raise DatasetError(f"{sample_id}: frame_index must be an integer")
        frame_id = item.get("frame_id")
        result.append(
            FrameRef(
                frame_id=str(frame_id if frame_id is not None else frame_index),
                frame_index=frame_index,
                path=path,
                video_id=str(item.get("video_id") or video_id),
            )
        )
    return result


def _extract_box_track(
    target: dict[str, Any],
    *,
    sample_id: str,
    video_id: str,
) -> list[BoxTrackItem]:
    value = target.get("box_track")
    if not isinstance(value, list) or not value:
        raise DatasetError(f"{sample_id}: missing target.box_track")
    result: list[BoxTrackItem] = []
    for item in value:
        if not isinstance(item, dict):
            raise DatasetError(f"{sample_id}: box_track item must be an object")
        box = item.get("box")
        if not isinstance(box, list) or len(box) != 4:
            raise DatasetError(f"{sample_id}: box must contain four numbers")
        result.append(
            BoxTrackItem(
                frame_id=str(item.get("frame_id") or item.get("frame_index")),
                frame_index=_extract_frame_index(item, sample_id=sample_id),
                video_id=str(item.get("video_id") or video_id),
                box=_coerce_normalized_numbers(box, expected=4, label="box"),
                coordinate_space=str(item.get("coordinate_space") or "normalized_0_1"),
            )
        )
    return result


def _extract_point_track(
    target: dict[str, Any],
    *,
    sample_id: str,
    video_id: str,
) -> list[PointTrackItem]:
    value = target.get("point_track")
    if not isinstance(value, list) or not value:
        raise DatasetError(f"{sample_id}: missing target.point_track")
    result: list[PointTrackItem] = []
    for item in value:
        if not isinstance(item, dict):
            raise DatasetError(f"{sample_id}: point_track item must be an object")
        point = item.get("point")
        if not isinstance(point, list) or len(point) != 2:
            raise DatasetError(f"{sample_id}: point must contain two numbers")
        result.append(
            PointTrackItem(
                frame_id=str(item.get("frame_id") or item.get("frame_index")),
                frame_index=_extract_frame_index(item, sample_id=sample_id),
                video_id=str(item.get("video_id") or video_id),
                point=_coerce_normalized_numbers(point, expected=2, label="point"),
                coordinate_space=str(item.get("coordinate_space") or "normalized_0_1"),
            )
        )
    return result


def _extract_frame_index(item: dict[str, Any], *, sample_id: str) -> int:
    frame_index = item.get("frame_index")
    if not isinstance(frame_index, int):
        raise DatasetError(f"{sample_id}: target frame_index must be an integer")
    return frame_index


def _coerce_normalized_numbers(
    value: list[Any],
    *,
    expected: int,
    label: str,
) -> list[float]:
    if len(value) != expected:
        raise DatasetError(f"{label} must contain {expected} numbers")
    result: list[float] = []
    for item in value:
        if not isinstance(item, int | float):
            raise DatasetError(f"{label} coordinates must be numeric")
        result.append(float(item))
    return result


def _extract_target_ref(row: dict[str, Any]) -> str | None:
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


def _extract_answer(row: dict[str, Any]) -> str:
    answer = row.get("answer")
    if isinstance(answer, str):
        return answer
    if isinstance(answer, dict):
        for key in ("text", "canonical", "answer", "value"):
            value = answer.get(key)
            if isinstance(value, str):
                return value
    value = row.get("gt_answer")
    if isinstance(value, str):
        return value
    raise DatasetError(f"{row.get('id') or row.get('qa_id')}: missing text answer")


def _extract_temporal_evidence(row: dict[str, Any]) -> list[list[float]]:
    for key in ("gt_temporal_evidence", "temporal_evidence"):
        value = row.get(key)
        if value is not None:
            return _coerce_intervals(value)

    evidence = row.get("evidence")
    if isinstance(evidence, dict):
        segments = evidence.get("segments")
        if segments is not None:
            return _coerce_intervals(segments)
    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, dict) and item.get("type") == "temporal_segments":
                return _coerce_intervals(item.get("segments", []))
    raise DatasetError(
        f"{row.get('id') or row.get('qa_id')}: missing temporal evidence"
    )


def _coerce_intervals(value: Any) -> list[list[float]]:
    if not isinstance(value, list):
        raise DatasetError("temporal evidence must be a list")
    intervals: list[list[float]] = []
    for item in value:
        if isinstance(item, dict):
            start = item.get("start_seconds", item.get("start"))
            end = item.get("end_seconds", item.get("end"))
        elif isinstance(item, list | tuple) and len(item) == 2:
            start, end = item
        else:
            raise DatasetError("each temporal evidence item must be a pair")
        if not isinstance(start, int | float) or not isinstance(end, int | float):
            raise DatasetError("temporal evidence bounds must be numeric")
        intervals.append([float(start), float(end)])
    return intervals


def _extract_media(row: dict[str, Any]) -> tuple[str | None, float | None]:
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

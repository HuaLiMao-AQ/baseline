"""媒体路径解析、懒下载与轻量视频探测。"""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from .dataset import DatasetError, download_hf_dataset_file, hf_auth_headers


class MediaError(RuntimeError):
    """样本级媒体处理失败时抛出。"""


def resolve_or_download_media(
    *,
    media_ref: str | None,
    repo_id: str,
    revision: str,
    cache_dir: Path,
) -> Path:
    """解析本地媒体路径，必要时从数据集仓库懒下载。

    Args:
        media_ref: 样本中的媒体路径或 URL。
        repo_id: Hugging Face dataset repo ID。
        revision: 固定 revision 或 tag。
        cache_dir: baseline 缓存根目录。

    Returns:
        可供模型读取的本地媒体文件路径。

    Raises:
        MediaError: 媒体路径缺失、下载失败或下载结果为空时抛出。
    """

    if media_ref is None or not media_ref.strip():
        raise MediaError("sample has no media path")

    local_candidate = Path(media_ref)
    if local_candidate.exists() and local_candidate.stat().st_size > 0:
        return local_candidate

    if media_ref.startswith(("http://", "https://")):
        url = media_ref
        path = _cached_url_path(media_ref, cache_dir)
    else:
        path = _cached_dataset_media_path(
            media_ref=media_ref,
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache_dir,
        )
        if path.exists() and path.stat().st_size > 0:
            return path
        try:
            return download_hf_dataset_file(
                repo_id=repo_id,
                revision=revision,
                file_path=media_ref,
                cache_dir=cache_dir,
                target_path=path,
            )
        except DatasetError as exc:
            raise MediaError(str(exc)) from exc

    if path.exists() and path.stat().st_size > 0:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers=hf_auth_headers())

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            with tempfile.NamedTemporaryFile(
                "wb",
                delete=False,
                dir=path.parent,
                prefix=f".{path.name}.",
            ) as tmp:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    tmp.write(chunk)
                tmp_path = Path(tmp.name)
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise MediaError("Hugging Face media 下载被拒绝") from exc
        raise MediaError(f"media download failed: HTTP {exc.code}") from exc
    except OSError as exc:
        raise MediaError(
            f"media download failed: {exc}. "
            "当前环境无法访问媒体地址；请检查网络/代理，或配置 HF_TOKEN，"
            "或提前把视频放到样本 media.path 指向的本地路径。"
        ) from exc

    if tmp_path.stat().st_size <= 0:
        tmp_path.unlink(missing_ok=True)
        raise MediaError("downloaded media is empty")
    tmp_path.replace(path)
    return path


def probe_video_duration(path: Path) -> float | None:
    """使用 ffprobe 探测视频时长。

    Args:
        path: 本地视频文件路径。

    Returns:
        视频时长秒数；ffprobe 不可用或探测失败时返回 ``None``。
    """

    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return None

    try:
        duration = float(completed.stdout.strip())
    except ValueError:
        return None
    if duration <= 0:
        return None
    return duration


def _cached_dataset_media_path(
    *,
    media_ref: str,
    repo_id: str,
    revision: str,
    cache_dir: Path,
) -> Path:
    safe_repo = repo_id.replace("/", "--")
    safe_revision = revision.replace("/", "--")
    return cache_dir / "hf" / "media" / safe_repo / safe_revision / media_ref


def _cached_url_path(media_ref: str, cache_dir: Path) -> Path:
    digest = hashlib.sha256(media_ref.encode("utf-8")).hexdigest()
    suffix = Path(media_ref.split("?", 1)[0]).suffix
    return cache_dir / "media" / "urls" / f"{digest}{suffix}"

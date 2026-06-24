"""大模型实验运行时缓存目录配置。"""

from __future__ import annotations

import os
from pathlib import Path

_MANAGED_ENV: dict[str, str] = {}


def configure_runtime_cache(cache_dir: Path) -> dict[str, str]:
    """把 Hugging Face 相关缓存指向指定目录。

    Args:
        cache_dir: baseline 使用的缓存根目录。

    Returns:
        本函数管理的环境变量及其最终值。
    """

    root = cache_dir.expanduser().resolve()
    hf_home = root / "huggingface"
    hf_cache = hf_hub_cache_dir(root)
    hf_datasets = hf_home / "datasets"
    hf_assets = hf_home / "assets"

    for path in (hf_home, hf_cache, hf_datasets, hf_assets):
        path.mkdir(parents=True, exist_ok=True)

    defaults = {
        "HF_HOME": hf_home,
        "HF_HUB_CACHE": hf_cache,
        "HUGGINGFACE_HUB_CACHE": hf_cache,
        "HF_DATASETS_CACHE": hf_datasets,
        "HF_ASSETS_CACHE": hf_assets,
        "TRANSFORMERS_CACHE": hf_cache,
    }
    for name, path in defaults.items():
        _set_managed_default(name, str(path))

    return {name: os.environ[name] for name in defaults}


def model_cache_dir(cache_dir: Path) -> Path:
    """返回 Transformers 模型加载应使用的 HF Hub 缓存目录。"""

    return hf_hub_cache_dir(cache_dir)


def hf_hub_cache_dir(cache_dir: Path) -> Path:
    """返回 ``cache_dir`` 下标准 HF Hub 缓存目录。"""

    return cache_dir.expanduser() / "huggingface" / "hub"


def hf_cache_kwargs(cache_dir: Path | None) -> dict[str, str]:
    """构造 ``from_pretrained`` 可接受的缓存参数。

    Args:
        cache_dir: HF Hub 缓存目录；为 ``None`` 时不注入参数。

    Returns:
        可直接传入 Transformers/HF API 的关键字参数。
    """

    if cache_dir is None:
        return {}
    return {"cache_dir": str(cache_dir)}


def _set_managed_default(name: str, value: str) -> None:
    current = os.environ.get(name)
    previous = _MANAGED_ENV.get(name)
    if current is None or current == previous:
        os.environ[name] = value
        _MANAGED_ENV[name] = value

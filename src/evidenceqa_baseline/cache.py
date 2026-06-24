"""Runtime cache placement for large model experiments."""

from __future__ import annotations

import os
from pathlib import Path

_MANAGED_ENV: dict[str, str] = {}


def configure_runtime_cache(cache_dir: Path) -> dict[str, str]:
    """Point Hugging Face runtime caches at the configured cache directory."""

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
    """Return the Hugging Face cache directory used for model downloads."""

    return hf_hub_cache_dir(cache_dir)


def hf_hub_cache_dir(cache_dir: Path) -> Path:
    """Return the standard Hugging Face Hub cache directory under cache_dir."""

    return cache_dir.expanduser() / "huggingface" / "hub"


def hf_cache_kwargs(cache_dir: Path | None) -> dict[str, str]:
    """Build kwargs accepted by Transformers/Hugging Face from_pretrained calls."""

    if cache_dir is None:
        return {}
    return {"cache_dir": str(cache_dir)}


def _set_managed_default(name: str, value: str) -> None:
    current = os.environ.get(name)
    previous = _MANAGED_ENV.get(name)
    if current is None or current == previous:
        os.environ[name] = value
        _MANAGED_ENV[name] = value

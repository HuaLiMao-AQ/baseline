"""CUDA device selection and runtime diagnostics for the PRO 6000 baseline."""

from __future__ import annotations

from typing import Any


class DeviceSelectionError(RuntimeError):
    """Raised when the requested accelerator cannot be used."""


def select_device(torch: Any, requested: str) -> str:
    """Return a CUDA torch device string for the PRO 6000 single-GPU route."""

    normalized = requested.strip().lower()
    if normalized in {"gpu", "cuda"} or normalized.startswith("cuda:"):
        if _cuda_available(torch):
            return "cuda" if normalized == "gpu" else requested
        raise DeviceSelectionError(_gpu_unavailable_message(torch, requested))

    raise DeviceSelectionError(
        "PRO 6000 baseline only supports CUDA devices; "
        f"requested {requested!r}."
    )


def select_dtype(torch: Any, requested: str, device: str) -> Any:
    """Map a CLI dtype string to a torch dtype for the selected device."""

    normalized = requested.lower()
    if normalized == "auto":
        normalized = "bfloat16"
    if normalized in {"bf16", "bfloat16"}:
        if _is_cuda_device(device) and _cuda_bf16_supported(torch):
            return torch.bfloat16
        return torch.float16
    if normalized in {"fp16", "float16"}:
        return torch.float16 if device != "cpu" else torch.float32
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise DeviceSelectionError(f"unsupported dtype: {requested}")


def collect_torch_accelerator_info(torch: Any) -> dict[str, Any]:
    """Collect CUDA metadata for reproducible PRO 6000 runs."""

    cuda_available = _cuda_available(torch)
    cuda_info: dict[str, Any] = {
        "available": cuda_available,
        "device_name": None,
        "device_count": 0,
        "device_names": [],
        "cuda_version": getattr(torch.version, "cuda", None),
    }
    if cuda_available:
        device_count = torch.cuda.device_count()
        cuda_info["device_count"] = device_count
        cuda_info["device_names"] = [
            torch.cuda.get_device_name(index) for index in range(device_count)
        ]
        cuda_info["device_name"] = (
            cuda_info["device_names"][0] if cuda_info["device_names"] else None
        )
        try:
            cuda_info["current_device"] = torch.cuda.current_device()
        except Exception:  # noqa: BLE001 - diagnostics should never break a run.
            cuda_info["current_device"] = None
        try:
            cuda_info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        except Exception:  # noqa: BLE001
            cuda_info["bf16_supported"] = None

    return {"cuda": cuda_info}


def _cuda_available(torch: Any) -> bool:
    return bool(getattr(torch, "cuda", None) and torch.cuda.is_available())


def _is_cuda_device(device: str) -> bool:
    return device == "cuda" or device.startswith("cuda:")


def _cuda_bf16_supported(torch: Any) -> bool:
    if not _cuda_available(torch):
        return False
    try:
        return bool(torch.cuda.is_bf16_supported())
    except Exception:  # noqa: BLE001
        return False


def _gpu_unavailable_message(torch: Any, requested: str) -> str:
    cuda = getattr(getattr(torch, "version", None), "cuda", None)
    build = "CUDA" if cuda else "CPU-only"
    return (
        f"requested {requested!r}, but no CUDA GPU device is available. "
        f"Detected PyTorch build: {build}; torch.version.cuda={cuda!r}; "
        "this baseline is pinned to one NVIDIA RTX PRO 6000."
    )

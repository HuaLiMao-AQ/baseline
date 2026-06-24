"""CUDA 设备选择和运行时诊断。"""

from __future__ import annotations

from typing import Any


class DeviceSelectionError(RuntimeError):
    """请求的加速设备不可用时抛出。"""


def select_device(torch: Any, requested: str) -> str:
    """按原 baseline 的单卡 CUDA 路线选择设备。

    Args:
        torch: 已导入的 torch 模块。
        requested: CLI 或配置中请求的设备名。

    Returns:
        torch 可识别的设备字符串。

    Raises:
        DeviceSelectionError: 请求 CPU 或 CUDA 不可用时抛出。
    """

    normalized = requested.strip().lower()
    if normalized in {"gpu", "cuda"} or normalized.startswith("cuda:"):
        if _cuda_available(torch):
            return "cuda" if normalized == "gpu" else requested
        raise DeviceSelectionError(_gpu_unavailable_message(torch, requested))

    raise DeviceSelectionError(
        "baseline 目前只支持 CUDA 设备；"
        f"收到的设备请求是 {requested!r}。"
    )


def select_dtype(torch: Any, requested: str, device: str) -> Any:
    """把配置里的 dtype 字符串映射为 torch dtype。

    Args:
        torch: 已导入的 torch 模块。
        requested: dtype 名称，例如 ``bfloat16`` 或 ``float16``。
        device: 已选择的设备字符串。

    Returns:
        torch dtype 对象。
    """

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
    raise DeviceSelectionError(f"不支持的 dtype: {requested}")


def collect_torch_accelerator_info(torch: Any) -> dict[str, Any]:
    """收集 CUDA 信息，写入可复现实验配置。"""

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
        except Exception:  # noqa: BLE001 - 诊断信息不能阻断实验。
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
        f"请求了 {requested!r}，但当前没有可用 CUDA GPU。"
        f"检测到的 PyTorch 构建为 {build}；torch.version.cuda={cuda!r}。"
    )

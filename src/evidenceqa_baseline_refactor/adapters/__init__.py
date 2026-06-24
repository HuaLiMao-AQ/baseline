"""模型 adapter 接口。"""

from .base import (
    AdapterError,
    AdapterResponse,
    BaseModelAdapter,
    GenerationConfig,
    PredictionRequest,
)

__all__ = [
    "AdapterError",
    "AdapterResponse",
    "BaseModelAdapter",
    "GenerationConfig",
    "PredictionRequest",
]

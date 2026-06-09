"""Fast, installable Qwen3-ASR inference helpers."""

from .model import ASRResult, MODEL_ALIASES, QwenASR, from_pretrained, resolve_model_id

__version__ = "0.1.0"

__all__ = [
    "ASRResult",
    "MODEL_ALIASES",
    "QwenASR",
    "from_pretrained",
    "resolve_model_id",
]

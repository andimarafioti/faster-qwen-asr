"""Public Qwen3-ASR wrapper.

This module keeps heavyweight ML imports lazy. Importing `faster_qwen_asr` should not
initialize PyTorch, vLLM, transformers, CUDA, or model registries.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any, Iterable, Literal

from .audio import SAMPLE_RATE, is_batch_audio, normalize_audio_arg

Backend = Literal["auto", "torch", "transformers", "vllm"]

MODEL_ALIASES = {
    "0.6b": "Qwen/Qwen3-ASR-0.6B",
    "0.6B": "Qwen/Qwen3-ASR-0.6B",
    "600m": "Qwen/Qwen3-ASR-0.6B",
    "small": "Qwen/Qwen3-ASR-0.6B",
    "1.7b": "Qwen/Qwen3-ASR-1.7B",
    "1.7B": "Qwen/Qwen3-ASR-1.7B",
    "1700m": "Qwen/Qwen3-ASR-1.7B",
    "large": "Qwen/Qwen3-ASR-1.7B",
}


@dataclass(frozen=True)
class ASRResult:
    """One transcription result returned by `transcribe(..., return_result=True)`."""

    text: str
    language: str = ""
    time_stamps: Any = None


def resolve_model_id(model: str | None = None, size: str | None = None) -> str:
    """Resolve a model id from a repo id, local path, or short size alias."""
    if model and size:
        raise ValueError("Pass either `model` or `size`, not both.")
    value = model or size or "0.6B"
    return MODEL_ALIASES.get(value, MODEL_ALIASES.get(value.lower(), value))


def from_pretrained(*args: Any, **kwargs: Any) -> "QwenASR":
    """Load a Qwen3-ASR model and return a ready-to-use wrapper."""
    return QwenASR.from_pretrained(*args, **kwargs)


class QwenASR:
    """Small API wrapper around Qwen3-ASR backends.

    `backend="vllm"` is the highest-throughput CUDA path when installed.
    `backend="torch"` uses a manual greedy decoder on top of the official
    Torch model. `backend="auto"` prefers vLLM on CUDA, then the Torch path,
    then falls back to the official transformers backend.
    """

    def __init__(
        self,
        backend_model: Any,
        *,
        model_id: str,
        backend: str,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.model = backend_model
        self.model_id = model_id
        self.backend = backend
        self.sample_rate = sample_rate

    @classmethod
    def from_pretrained(
        cls,
        model: str | None = None,
        *,
        size: str | None = None,
        backend: Backend = "auto",
        device: str | None = None,
        dtype: str | Any | None = None,
        attn_implementation: str | None = None,
        max_batch_size: int = 32,
        max_new_tokens: int = 256,
        forced_aligner: str | None = None,
        forced_aligner_kwargs: dict[str, Any] | None = None,
        gpu_memory_utilization: float = 0.8,
        use_cuda_graph: bool = True,
        cuda_graph_stride: int = 128,
        use_torch_compile: bool = True,
        quantization: str | None = None,
        trust_remote_code: bool | None = None,
        **backend_kwargs: Any,
    ) -> "QwenASR":
        model_id = resolve_model_id(model=model, size=size)
        selected_backend = _select_backend(backend, device)
        _enable_cuda_defaults()

        if quantization and selected_backend != "torch":
            raise ValueError(
                f"quantization={quantization!r} is only supported by the torch backend, "
                f"got backend={selected_backend!r}."
            )

        if selected_backend == "vllm":
            qwen_model_cls = _load_qwen_model_cls()
            llm_kwargs = dict(backend_kwargs)
            llm_kwargs.setdefault("gpu_memory_utilization", gpu_memory_utilization)
            if dtype is not None:
                llm_kwargs.setdefault("dtype", _dtype_to_vllm(dtype))
            if trust_remote_code is not None:
                llm_kwargs.setdefault("trust_remote_code", trust_remote_code)

            backend_model = qwen_model_cls.LLM(
                model=model_id,
                forced_aligner=forced_aligner,
                forced_aligner_kwargs=forced_aligner_kwargs,
                max_inference_batch_size=max_batch_size,
                max_new_tokens=max_new_tokens,
                **llm_kwargs,
            )
        elif selected_backend == "torch":
            torch = _import_torch()
            device = _resolve_device(device, torch)
            torch_dtype = _resolve_torch_dtype(dtype, device, torch)

            model_kwargs = dict(backend_kwargs)
            if trust_remote_code is not None:
                model_kwargs.setdefault("trust_remote_code", trust_remote_code)
            if torch_dtype is not None:
                model_kwargs.setdefault("dtype", torch_dtype)
            model_kwargs.setdefault("device_map", _device_map_from_device(device))

            from .torch_backend import TorchQwenASRBackend

            torch_attn_implementation = attn_implementation or "eager"
            backend_model = TorchQwenASRBackend.from_pretrained(
                model_id,
                forced_aligner=forced_aligner,
                forced_aligner_kwargs=forced_aligner_kwargs,
                max_inference_batch_size=max_batch_size,
                max_new_tokens=max_new_tokens,
                use_cuda_graph=use_cuda_graph,
                cuda_graph_stride=cuda_graph_stride,
                use_torch_compile=use_torch_compile,
                quantization=quantization,
                attn_implementation=torch_attn_implementation,
                **model_kwargs,
            )
        elif selected_backend == "transformers":
            qwen_model_cls = _load_qwen_model_cls()
            torch = _import_torch()
            device = _resolve_device(device, torch)
            torch_dtype = _resolve_torch_dtype(dtype, device, torch)

            model_kwargs = dict(backend_kwargs)
            if trust_remote_code is not None:
                model_kwargs.setdefault("trust_remote_code", trust_remote_code)
            if torch_dtype is not None:
                model_kwargs.setdefault("dtype", torch_dtype)
            model_kwargs.setdefault("device_map", _device_map_from_device(device))
            model_kwargs.setdefault("attn_implementation", attn_implementation or "sdpa")

            backend_model = qwen_model_cls.from_pretrained(
                model_id,
                forced_aligner=forced_aligner,
                forced_aligner_kwargs=forced_aligner_kwargs,
                max_inference_batch_size=max_batch_size,
                max_new_tokens=max_new_tokens,
                **model_kwargs,
            )
            if attn_implementation:
                _set_qwen_attention_implementation(backend_model.model, attn_implementation)
        else:
            raise ValueError(f"Unsupported backend: {selected_backend!r}")

        return cls(backend_model, model_id=model_id, backend=selected_backend)

    def transcribe(
        self,
        audio: Any,
        *,
        context: str | list[str] = "",
        language: str | list[str | None] | None = None,
        return_time_stamps: bool = False,
        return_result: bool = False,
        sample_rate: int | None = None,
    ) -> str | list[str] | ASRResult | list[ASRResult]:
        """Transcribe one audio input or a batch.

        By default a single input returns a string, matching `nano-parakeet`.
        Pass `return_result=True` or `return_time_stamps=True` to receive
        `ASRResult` objects with language and timestamp metadata.
        """
        single = not is_batch_audio(audio)
        normalized_audio = normalize_audio_arg(audio, sample_rate=sample_rate or self.sample_rate)
        outputs = self.model.transcribe(
            audio=normalized_audio,
            context=context,
            language=language,
            return_time_stamps=return_time_stamps,
        )
        results = [_convert_result(item) for item in outputs]

        if return_result or return_time_stamps:
            return results[0] if single else results

        texts = [result.text for result in results]
        return texts[0] if single else texts

    def transcribe_batch(
        self,
        audio: Iterable[Any],
        *,
        context: str | list[str] = "",
        language: str | list[str | None] | None = None,
        return_time_stamps: bool = False,
        return_result: bool = True,
    ) -> list[ASRResult] | list[str]:
        """Batch-oriented alias for `transcribe`."""
        out = self.transcribe(
            list(audio),
            context=context,
            language=language,
            return_time_stamps=return_time_stamps,
            return_result=return_result,
        )
        return out if isinstance(out, list) else [out]

    def init_streaming_state(self, **kwargs: Any) -> Any:
        """Create a vLLM streaming state."""
        self._require_streaming_backend()
        return self.model.init_streaming_state(**kwargs)

    def streaming_transcribe(self, pcm16k: Any, state: Any) -> Any:
        """Feed one 16 kHz mono chunk into a vLLM streaming state."""
        self._require_streaming_backend()
        return self.model.streaming_transcribe(pcm16k, state)

    def finish_streaming_transcribe(self, state: Any) -> Any:
        """Flush the remaining audio in a vLLM streaming state."""
        self._require_streaming_backend()
        return self.model.finish_streaming_transcribe(state)

    def _require_streaming_backend(self) -> None:
        if self.backend != "vllm":
            raise ValueError("Streaming requires backend='vllm'.")


def _convert_result(item: Any) -> ASRResult:
    return ASRResult(
        text=getattr(item, "text", ""),
        language=getattr(item, "language", ""),
        time_stamps=getattr(item, "time_stamps", None),
    )


def _load_qwen_model_cls() -> Any:
    try:
        from qwen_asr import Qwen3ASRModel
    except Exception as exc:
        raise ImportError(
            "qwen-asr is required. Install this package with `pip install faster-qwen-asr` "
            "or, for the vLLM backend, `pip install 'faster-qwen-asr[fast]'`."
        ) from exc
    return Qwen3ASRModel


def _set_qwen_attention_implementation(model: Any, attn_implementation: str) -> None:
    configs = [
        getattr(model, "config", None),
        getattr(getattr(model, "thinker", None), "config", None),
        getattr(getattr(getattr(model, "thinker", None), "model", None), "config", None),
        getattr(getattr(getattr(model, "thinker", None), "audio_tower", None), "config", None),
    ]
    for config in configs:
        if config is not None:
            setattr(config, "_attn_implementation", attn_implementation)


def _select_backend(requested: Backend, device: str | None) -> str:
    if requested not in ("auto", "torch", "transformers", "vllm"):
        raise ValueError("backend must be one of: auto, torch, transformers, vllm")
    if requested != "auto":
        return requested
    if device == "cpu":
        return "transformers"

    torch = _try_import_torch()
    cuda = bool(torch is not None and torch.cuda.is_available())
    if cuda and importlib.util.find_spec("vllm") is not None:
        return "vllm"
    return "torch" if torch is not None else "transformers"


def _resolve_device(device: str | None, torch: Any) -> str:
    if device and device != "auto":
        return device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _device_map_from_device(device: str) -> str:
    if device == "cuda":
        return "cuda:0"
    return device


def _resolve_torch_dtype(dtype: str | Any | None, device: str, torch: Any) -> Any | None:
    if dtype is None or dtype == "auto":
        if device.startswith("cuda") and torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return None
    if isinstance(dtype, str):
        normalized = dtype.lower()
        if normalized in ("bf16", "bfloat16"):
            return torch.bfloat16
        if normalized in ("fp16", "float16", "half"):
            return torch.float16
        if normalized in ("fp32", "float32"):
            return torch.float32
        raise ValueError(f"Unsupported dtype string: {dtype!r}")
    return dtype


def _dtype_to_vllm(dtype: str | Any) -> str:
    if isinstance(dtype, str):
        normalized = dtype.lower()
        if normalized == "bf16":
            return "bfloat16"
        if normalized == "fp16":
            return "float16"
        if normalized == "fp32":
            return "float32"
        return normalized
    name = str(dtype).replace("torch.", "")
    return {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}.get(name, name)


def _enable_cuda_defaults() -> None:
    torch = _try_import_torch()
    if torch is None:
        return
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def _try_import_torch() -> Any | None:
    try:
        import torch

        return torch
    except Exception:
        return None


def _import_torch() -> Any:
    torch = _try_import_torch()
    if torch is None:
        raise ImportError("PyTorch is required to load Qwen3-ASR.")
    return torch

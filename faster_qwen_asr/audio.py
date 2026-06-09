"""Audio helpers used by the lightweight wrapper and streaming utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

SAMPLE_RATE = 16000


def is_batch_audio(audio: Any) -> bool:
    """Return True when `audio` should be treated as a batch of samples."""
    if isinstance(audio, (str, bytes, Path)):
        return False
    if _is_numpy_array(audio) or _is_torch_tensor(audio):
        return False
    if isinstance(audio, tuple) and len(audio) == 2:
        return False
    return isinstance(audio, list)


def normalize_audio_arg(audio: Any, sample_rate: int = SAMPLE_RATE) -> Any:
    """Normalize convenience audio forms to values accepted by `qwen_asr`.

    The upstream backend already handles paths, URLs, base64 strings, and
    `(np.ndarray, sr)` tuples. This function adds direct torch.Tensor support
    and applies the conversion recursively for batches.
    """
    if is_batch_audio(audio):
        return [normalize_audio_arg(item, sample_rate=sample_rate) for item in audio]

    if _is_torch_tensor(audio):
        arr = audio.detach().float().cpu().numpy()
        return (arr, sample_rate)

    return audio


def load_audio_16k(path: str | Path):
    """Load a local audio file as mono 16 kHz float32 numpy samples."""
    import librosa
    import numpy as np

    wav, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return np.asarray(wav, dtype="float32")


def audio_duration_seconds(audio: Any, sample_rate: int = SAMPLE_RATE) -> float | None:
    """Best-effort duration for a local path, tensor, numpy array, or `(array, sr)`."""
    if isinstance(audio, tuple) and len(audio) == 2:
        arr, sr = audio
        return float(len(arr)) / float(sr)
    if _is_numpy_array(audio) or _is_torch_tensor(audio):
        return float(len(audio)) / float(sample_rate)
    if isinstance(audio, (str, Path)):
        try:
            import soundfile as sf

            info = sf.info(str(audio))
            return float(info.frames) / float(info.samplerate)
        except Exception:
            return None
    return None


def _is_numpy_array(value: Any) -> bool:
    return value.__class__.__module__.startswith("numpy") and hasattr(value, "shape")


def _is_torch_tensor(value: Any) -> bool:
    return value.__class__.__module__.startswith("torch") and hasattr(value, "detach")

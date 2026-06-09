#!/usr/bin/env python3
"""Compare faster-qwen-asr against nano-parakeet on one audio file."""

from __future__ import annotations

import argparse
import time

import numpy as np
import soundfile as sf

from faster_qwen_asr import from_pretrained
from faster_qwen_asr.audio import audio_duration_seconds


def _sync_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _load_audio_tensor(path: str):
    import torch

    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa

        audio = librosa.resample(audio.astype("float32"), orig_sr=sr, target_sr=16000)
    return torch.from_numpy(np.asarray(audio, dtype="float32"))


def _time_call(fn, runs: int) -> tuple[float, str]:
    text = fn()
    _sync_cuda()
    times = []
    for _ in range(runs):
        _sync_cuda()
        start = time.perf_counter()
        text = fn()
        _sync_cuda()
        times.append(time.perf_counter() - start)
    return min(times), text


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Qwen3-ASR and nano-parakeet throughput")
    parser.add_argument("audio")
    parser.add_argument("--qwen-size", default="0.6B", choices=["0.6B", "1.7B"])
    parser.add_argument("--qwen-backend", default="torch", choices=["auto", "torch", "transformers", "vllm"])
    parser.add_argument("--qwen-language", default=None)
    parser.add_argument("--dtype", default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    duration = audio_duration_seconds(args.audio)
    if not duration:
        raise RuntimeError(f"Could not determine duration for {args.audio!r}")

    dtype = None if args.dtype == "auto" else args.dtype

    qwen = from_pretrained(
        size=args.qwen_size,
        backend=args.qwen_backend,
        dtype=dtype,
        local_files_only=args.local_files_only,
    )
    qwen_best, qwen_text = _time_call(
        lambda: qwen.transcribe(args.audio, language=args.qwen_language),
        args.runs,
    )

    parakeet_best = None
    parakeet_text = None
    try:
        from nano_parakeet import from_pretrained as parakeet_from_pretrained

        audio_t = _load_audio_tensor(args.audio)
        parakeet = parakeet_from_pretrained(device="cuda")

        def run_parakeet() -> str:
            token_ids = parakeet.transcribe_audio(audio_t)
            return parakeet.sp.DecodeIds(token_ids).strip()

        parakeet_best, parakeet_text = _time_call(run_parakeet, args.runs)
    except Exception as exc:
        parakeet_text = f"unavailable: {type(exc).__name__}: {exc}"

    print(f"audio_sec={duration:.4f}")
    print(f"qwen_model={qwen.model_id}")
    print(f"qwen_backend={qwen.backend}")
    print(f"qwen_best_sec={qwen_best:.4f}")
    print(f"qwen_rtf={duration / qwen_best:.2f}")
    print(f"qwen_text={qwen_text!r}")
    if parakeet_best is not None:
        print("parakeet_model=nvidia/parakeet-tdt-0.6b-v3")
        print(f"parakeet_best_sec={parakeet_best:.4f}")
        print(f"parakeet_rtf={duration / parakeet_best:.2f}")
    print(f"parakeet_text={parakeet_text!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

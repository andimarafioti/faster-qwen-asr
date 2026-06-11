#!/usr/bin/env python3
"""Measure single-file Qwen3-ASR throughput."""

from __future__ import annotations

import argparse
import time

from faster_qwen_asr import from_pretrained
from faster_qwen_asr.audio import audio_duration_seconds


def _cuda_synchronize() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen3-ASR throughput benchmark")
    parser.add_argument("audio")
    parser.add_argument("--model", default=None)
    parser.add_argument("--size", default="0.6B", choices=["0.6B", "1.7B", "small", "large"])
    parser.add_argument("--backend", default="auto", choices=["auto", "torch", "transformers", "vllm"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--attn-implementation", default="auto")
    parser.add_argument("--language", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--cuda-graph-stride", type=int, default=128)
    parser.add_argument("--no-torch-compile", action="store_true")
    parser.add_argument("--quantization", default=None, choices=["int8"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--repeat", type=int, default=1, help="repeat the same audio N times as a batch")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    dtype = None if args.dtype == "auto" else args.dtype
    attn_implementation = None if args.attn_implementation == "auto" else args.attn_implementation
    model = from_pretrained(
        model=args.model,
        size=args.size,
        backend=args.backend,
        device=None if args.device == "auto" else args.device,
        dtype=dtype,
        attn_implementation=attn_implementation,
        max_new_tokens=args.max_new_tokens,
        max_batch_size=args.batch_size,
        use_cuda_graph=not args.no_cuda_graph,
        cuda_graph_stride=args.cuda_graph_stride,
        use_torch_compile=not args.no_torch_compile,
        quantization=args.quantization,
        local_files_only=args.local_files_only,
    )

    audio_arg = args.audio if args.repeat <= 1 else [args.audio] * args.repeat

    for _ in range(args.warmup):
        model.transcribe(audio_arg, language=args.language)
    _cuda_synchronize()

    single_duration = audio_duration_seconds(args.audio)
    duration = single_duration * max(1, args.repeat) if single_duration else None
    times = []
    last_text = ""
    for _ in range(args.runs):
        _cuda_synchronize()
        start = time.perf_counter()
        last_text = model.transcribe(audio_arg, language=args.language)
        _cuda_synchronize()
        times.append(time.perf_counter() - start)

    best = min(times)
    rtf = (duration / best) if duration else None
    print(f"model={model.model_id}")
    print(f"backend={model.backend}")
    print(f"batch_size={max(1, args.repeat)}")
    print(f"best_sec={best:.4f}")
    if rtf is not None:
        print(f"audio_sec={duration:.4f}")
        print(f"rtf={rtf:.2f}")
    print(f"text={last_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

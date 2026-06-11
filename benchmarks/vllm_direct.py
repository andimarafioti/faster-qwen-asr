#!/usr/bin/env python3
"""Measure direct vLLM Qwen3-ASR throughput.

This follows the official README's direct vLLM deployment path (`vllm.LLM`
plus audio chat input) instead of going through `qwen_asr.Qwen3ASRModel.LLM`.
It is useful on stacks where the packaged `qwen-asr[vllm]` pin is not the
CUDA/vLLM build being benchmarked.
"""

from __future__ import annotations

import argparse
import base64
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from faster_qwen_asr.audio import audio_duration_seconds
from faster_qwen_asr.model import resolve_model_id


def _audio_conversation(path: Path) -> list[dict[str, object]]:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {"url": "data:audio/wav;base64," + data},
                }
            ],
        }
    ]


def _print_runtime() -> None:
    try:
        import torch
        import transformers
        import vllm

        print(f"torch={torch.__version__}")
        print(f"torch_cuda={torch.version.cuda}")
        print(f"cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"gpu={torch.cuda.get_device_name(0)}")
        print(f"vllm={vllm.__version__}")
        print(f"transformers={transformers.__version__}")
    except Exception as exc:
        print(f"runtime_info_error={type(exc).__name__}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Direct vLLM Qwen3-ASR throughput benchmark")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", default=None, help="Hugging Face repo id or local model path")
    parser.add_argument(
        "--size",
        default="0.6B",
        choices=["0.6B", "1.7B", "small", "large"],
        help="model size alias; ignored when --model is supplied",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--async-scheduling", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    model_id = args.model or resolve_model_id(size=args.size)
    conversation = _audio_conversation(args.audio)
    duration = audio_duration_seconds(args.audio)

    _print_runtime()
    load_start = time.perf_counter()
    llm = LLM(
        model=model_id,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        async_scheduling=args.async_scheduling,
    )
    print(f"load_sec={time.perf_counter() - load_start:.4f}")

    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    last_text = ""
    for _ in range(args.warmup):
        outputs = llm.chat(conversation, sampling_params=sampling_params, use_tqdm=False)
        last_text = outputs[0].outputs[0].text

    times = []
    for _ in range(args.runs):
        start = time.perf_counter()
        outputs = llm.chat(conversation, sampling_params=sampling_params, use_tqdm=False)
        times.append(time.perf_counter() - start)
        last_text = outputs[0].outputs[0].text

    best = min(times)
    print(f"model={model_id}")
    print("backend=vllm_direct")
    print(f"eager={args.enforce_eager}")
    print("batch_size=1")
    print(f"best_sec={best:.4f}")
    if duration is not None:
        print(f"audio_sec={duration:.4f}")
        print(f"rtf={duration / best:.2f}")
    print(f"text={last_text!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

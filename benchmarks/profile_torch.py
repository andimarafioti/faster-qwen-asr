#!/usr/bin/env python3
"""Profile the single-stream Torch backend stages for one audio file."""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from statistics import mean
from typing import Any

from faster_qwen_asr import from_pretrained
from faster_qwen_asr.torch_backend import _normalize_audios, _parse_asr_output, _split_audio_into_chunks


def _cuda_synchronize() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _record(times: dict[str, list[float]], name: str, start: float) -> float:
    now = time.perf_counter()
    times[name].append((now - start) * 1000.0)
    return now


def _format_ms(values: list[float]) -> str:
    return f"best={min(values):8.3f} ms  avg={mean(values):8.3f} ms"


def _decode_once(backend: Any, outputs: Any, inputs: Any, wav: Any) -> tuple[list[int], str, int]:
    import torch

    decode_mode = "dynamic"
    graph_budget = backend.max_new_tokens
    if backend._can_use_cuda_graph(torch):
        try:
            graph_budget = backend._estimate_graph_max_new_tokens(wav)
            generated = backend._decode_with_graph(outputs, max_new_tokens=graph_budget)
            decode_mode = "cuda_graph"
            if len(generated) >= graph_budget < backend.max_new_tokens:
                generated = backend._decode_dynamic(outputs, inputs["attention_mask"])
                decode_mode = "dynamic_budget_retry"
        except Exception:
            backend._graph_failed = True
            generated = backend._decode_dynamic(outputs, inputs["attention_mask"])
            decode_mode = "dynamic_graph_failed"
    else:
        generated = backend._decode_dynamic(outputs, inputs["attention_mask"])
    return generated, decode_mode, graph_budget


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile Qwen3-ASR Torch backend stages")
    parser.add_argument("audio")
    parser.add_argument("--model", default=None)
    parser.add_argument("--size", default="0.6B", choices=["0.6B", "1.7B", "small", "large"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--attn-implementation", default="auto")
    parser.add_argument("--language", default=None)
    parser.add_argument("--context", default="")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--cuda-graph-stride", type=int, default=128)
    parser.add_argument("--no-torch-compile", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    dtype = None if args.dtype == "auto" else args.dtype
    attn_implementation = None if args.attn_implementation == "auto" else args.attn_implementation
    model = from_pretrained(
        model=args.model,
        size=args.size,
        backend="torch",
        device=None if args.device == "auto" else args.device,
        dtype=dtype,
        attn_implementation=attn_implementation,
        max_new_tokens=args.max_new_tokens,
        max_batch_size=args.batch_size,
        use_cuda_graph=not args.no_cuda_graph,
        cuda_graph_stride=args.cuda_graph_stride,
        use_torch_compile=not args.no_torch_compile,
        local_files_only=args.local_files_only,
    )
    backend = model.model

    for _ in range(args.warmup):
        model.transcribe(args.audio, context=args.context, language=args.language)
    _cuda_synchronize()

    import torch

    times: dict[str, list[float]] = defaultdict(list)
    token_counts: list[int] = []
    graph_budgets: list[int] = []
    decode_modes: dict[str, int] = defaultdict(int)
    last_text = ""
    chunk_count = 0

    for _ in range(args.runs):
        t0 = time.perf_counter()
        wavs = _normalize_audios(args.audio)
        chunks = _split_audio_into_chunks(wavs[0])
        chunk_count = len(chunks)
        wav = chunks[0][0]
        t = _record(times, "audio", t0)

        prompt = backend.official_model._build_text_prompt(
            context=args.context,
            force_language=args.language,
        )
        t = _record(times, "prompt", t)

        inputs = backend.processor(text=[prompt], audio=[wav], return_tensors="pt", padding=True)
        t = _record(times, "processor", t)

        inputs = inputs.to(backend.model.device).to(backend.model.dtype)
        _cuda_synchronize()
        t = _record(times, "to_device", t)

        with torch.inference_mode():
            backend.model.thinker.rope_deltas = None
            outputs = backend.model.thinker(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                input_features=inputs["input_features"],
                feature_attention_mask=inputs["feature_attention_mask"],
                use_cache=True,
            )
            _cuda_synchronize()
            t = _record(times, "prefill", t)

            generated, decode_mode, graph_budget = _decode_once(backend, outputs, inputs, wav)
            _cuda_synchronize()
            t = _record(times, "decode", t)

        token_counts.append(len(generated))
        graph_budgets.append(graph_budget)
        decode_modes[decode_mode] += 1

        token_tensor = torch.tensor([generated], dtype=torch.long)
        raw_text = backend.processor.batch_decode(
            token_tensor,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        _, last_text = _parse_asr_output(raw_text, user_language=args.language)
        t = _record(times, "detokenize", t)
        times["total"].append((t - t0) * 1000.0)

    print(f"model={model.model_id}")
    print(f"backend={model.backend}")
    print(f"chunks_profiled=1/{chunk_count}")
    print(f"decode_modes={dict(decode_modes)}")
    print(f"tokens_best={min(token_counts)} tokens_avg={mean(token_counts):.1f}")
    if graph_budgets:
        print(f"graph_budget_best={min(graph_budgets)} graph_budget_avg={mean(graph_budgets):.1f}")
    for name in ("audio", "prompt", "processor", "to_device", "prefill", "decode", "detokenize", "total"):
        print(f"{name:>10}: {_format_ms(times[name])}")
    print(f"text={last_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

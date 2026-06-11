"""Command line interface for faster-qwen-asr."""

from __future__ import annotations

import argparse
import json
import sys
import time

from .audio import audio_duration_seconds
from .model import MODEL_ALIASES, QwenASR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fast Qwen3-ASR transcription")
    parser.add_argument("audio", nargs="*", help="audio file, URL, or base64 data URL")
    parser.add_argument("--model", default=None, help="Hugging Face repo id or local model path")
    parser.add_argument(
        "--size",
        default=None,
        choices=["0.6B", "1.7B", "small", "large"],
        help="model size alias; defaults to 0.6B",
    )
    parser.add_argument("--backend", default="auto", choices=["auto", "torch", "transformers", "vllm"])
    parser.add_argument("--device", default="auto", help="transformers device, e.g. cuda:0 or cpu")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--language", default=None, help="force output language, e.g. English")
    parser.add_argument("--context", default="", help="system prompt/context bias")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--attn-implementation", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--no-cuda-graph", action="store_true", help="disable Torch backend CUDA graph decode")
    parser.add_argument("--cuda-graph-stride", type=int, default=128, help="Torch backend graph cache bucket size")
    parser.add_argument("--no-torch-compile", action="store_true", help="disable torch.compile of the CUDA graph decode step")
    parser.add_argument("--local-files-only", action="store_true", help="use only cached model files")
    parser.add_argument("--timestamps", action="store_true", help="return forced-alignment timestamps")
    parser.add_argument("--forced-aligner", default=None, help="forced aligner repo id or local path")
    parser.add_argument("--json", action="store_true", help="print structured JSON results")
    parser.add_argument("--list-models", action="store_true", help="print built-in model aliases and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_models:
        for alias, model_id in sorted(MODEL_ALIASES.items()):
            print(f"{alias}\t{model_id}")
        return 0
    if not args.audio:
        parser.error("at least one audio input is required unless --list-models is used")

    dtype = None if args.dtype == "auto" else args.dtype
    device = None if args.device == "auto" else args.device

    start_load = time.perf_counter()
    attn_implementation = None if args.attn_implementation == "auto" else args.attn_implementation

    model = QwenASR.from_pretrained(
        model=args.model,
        size=args.size,
        backend=args.backend,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
        max_batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        forced_aligner=args.forced_aligner,
        gpu_memory_utilization=args.gpu_memory_utilization,
        use_cuda_graph=not args.no_cuda_graph,
        cuda_graph_stride=args.cuda_graph_stride,
        use_torch_compile=not args.no_torch_compile,
        local_files_only=args.local_files_only,
    )
    load_time = time.perf_counter() - start_load

    single = len(args.audio) == 1
    audio_arg = args.audio[0] if single else args.audio
    durations = [audio_duration_seconds(path) for path in args.audio]

    start = time.perf_counter()
    result = model.transcribe(
        audio_arg,
        context=args.context,
        language=args.language,
        return_time_stamps=args.timestamps,
        return_result=True,
    )
    infer_time = time.perf_counter() - start

    results = [result] if single else result
    payload = []
    for path, duration, item in zip(args.audio, durations, results):
        rtf = (duration / infer_time) if duration and infer_time > 0 and single else None
        payload.append(
            {
                "text": item.text,
                "language": item.language,
                "time_stamps": _jsonable_time_stamps(item.time_stamps),
                "audio": path,
                "duration_sec": duration,
                "rtf": rtf,
            }
        )

    if args.json:
        print(
            json.dumps(
                {
                    "model": model.model_id,
                    "backend": model.backend,
                    "load_sec": load_time,
                    "inference_sec": infer_time,
                    "results": payload,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        for item in payload:
            print(item["text"])
        if single and payload[0]["rtf"] is not None:
            print(
                f"Inference: {infer_time:.3f}s, RTF {payload[0]['rtf']:.2f}, backend {model.backend}",
                file=sys.stderr,
            )

    return 0


def _jsonable_time_stamps(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_jsonable_time_stamps(item) for item in value]
    if hasattr(value, "items"):
        return _jsonable_time_stamps(list(value.items))
    if hasattr(value, "__dict__"):
        return {k: _jsonable_time_stamps(v) for k, v in vars(value).items()}
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())

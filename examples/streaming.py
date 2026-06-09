#!/usr/bin/env python3
"""vLLM streaming example for a local audio file."""

from __future__ import annotations

import argparse

from faster_qwen_asr import from_pretrained
from faster_qwen_asr.audio import load_audio_16k


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--size", default="0.6B", choices=["0.6B", "1.7B", "small", "large"])
    parser.add_argument("--language", default=None)
    parser.add_argument("--step-ms", type=int, default=500)
    parser.add_argument("--chunk-sec", type=float, default=2.0)
    args = parser.parse_args()

    model = from_pretrained(size=args.size, backend="vllm", max_new_tokens=32)
    wav = load_audio_16k(args.audio)
    step = max(1, int(round(args.step_ms / 1000.0 * 16000)))

    state = model.init_streaming_state(language=args.language, chunk_size_sec=args.chunk_sec)
    for offset in range(0, len(wav), step):
        state = model.streaming_transcribe(wav[offset : offset + step], state)
        if state.text:
            print(state.text)

    state = model.finish_streaming_transcribe(state)
    print(state.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

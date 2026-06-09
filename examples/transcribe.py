#!/usr/bin/env python3
"""Minimal file transcription example."""

from __future__ import annotations

import argparse

from faster_qwen_asr import from_pretrained


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--size", default="0.6B", choices=["0.6B", "1.7B", "small", "large"])
    parser.add_argument("--backend", default="auto", choices=["auto", "torch", "transformers", "vllm"])
    parser.add_argument("--language", default=None)
    args = parser.parse_args()

    model = from_pretrained(size=args.size, backend=args.backend)
    print(model.transcribe(args.audio, language=args.language))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

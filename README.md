# faster-qwen-asr

Fast CUDA-oriented Qwen3-ASR inference wrapper.

This repository contains the Torch/vLLM path that was split out of
`qwenasr.cpp`. The C++/GGML implementation now lives in `qwenasr.cpp`; this repo
keeps the faster CUDA runtime and benchmarking utilities.

## Install

```bash
pip install -e .
```

For the vLLM backend:

```bash
pip install -e ".[fast]"
```

## Use

```python
from faster_qwen_asr import from_pretrained

model = from_pretrained(size="0.6B", backend="torch", dtype="bf16")
print(model.transcribe("sample.wav", language="English"))
```

CLI:

```bash
faster-qwen-asr sample.wav --size 0.6B --backend torch --dtype bf16 --language English
```

The Torch backend uses a manual greedy decoder and enables CUDA defaults when a
CUDA device is available. CUDA graph decode is on by default and can be disabled
with `--no-cuda-graph`. The decode step inside the graph is compiled with
`torch.compile` before capture (disable with `--no-torch-compile` or
`use_torch_compile=False`); the first request after loading pays a one-time
compile cost (seconds with a warm inductor cache, up to ~1 minute cold), and a
request longer than any previous one re-captures at a larger cache size and
pays it again.

Optional int8 weight-only quantization of the text decoder
(`pip install 'faster-qwen-asr[quant]'`, then `quantization="int8"` or
`--quantization int8`) roughly halves decode time again. It is off by default
because it changes the weights' numerics; transcripts may differ slightly from
bf16 on some clips, so spot-check on your own data before enabling it.

## Benchmarks

```bash
./benchmark.sh both sample.wav
python benchmarks/throughput.py sample.wav --size 0.6B --backend torch --repeat 4
python benchmarks/profile_torch.py sample.wav --size 0.6B --language English
python benchmarks/compare_parakeet.py sample.wav --qwen-size 0.6B
```

The root benchmark script compares the Torch backend's dynamic greedy decode
baseline against CUDA graph decode.

### NVIDIA GB10

Measured on June 11, 2026 with PyTorch 2.11.0+cu130, CUDA 13.0, driver
580.126.09. Audio was a 10.9s 16 kHz mono English clip, forced English,
`--backend torch --dtype bf16`, warmups excluded (the first CUDA graph request
pays the one-time `torch.compile` cost), five timed runs. The baseline is the
plain Torch dynamic greedy decode (`--no-cuda-graph`); the CUDA graph column is
the default configuration (self-feeding captured graph with a compiled decode
step). RTF > 1.0 is faster than real time.

**Full precision (bf16)**

| Model | Dynamic decode latency | Dynamic RTF | CUDA graph latency | CUDA graph RTF | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-ASR-0.6B | 0.4767s | 22.91 | 0.2044s | 53.45 | 2.33x |
| Qwen3-ASR-1.7B | 0.8989s | 12.15 | 0.4804s | 22.73 | 1.87x |

**int8 weight-only (opt-in, `--quantization int8`)**

| Model | Latency | RTF | Speedup vs dynamic | Speedup vs bf16 graph |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-ASR-0.6B | 0.1458s | 74.89 | 3.27x | 1.40x |
| Qwen3-ASR-1.7B | 0.3117s | 35.04 | 2.88x | 1.54x |

The bf16 configurations produce transcripts byte-identical to the
pre-optimization implementation. int8 quantizes the text decoder weights, so
its transcripts are not guaranteed identical to bf16: on a 4-clip verification
set, 7/8 matched exactly and one clip changed a single word.

Full run metadata is in `bench_results_NVIDIA_GB10.json`.

The C++ parity and GGUF conversion tooling remains in `qwenasr.cpp`.

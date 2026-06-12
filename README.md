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
python benchmarks/vllm_direct.py sample.wav --size 0.6B
python benchmarks/profile_torch.py sample.wav --size 0.6B --language English
python benchmarks/compare_parakeet.py sample.wav --qwen-size 0.6B
```

The root benchmark script compares the Torch backend's dynamic greedy decode
baseline against CUDA graph decode.

### NVIDIA GB10

Measured on June 12, 2026 with PyTorch 2.11.0+cu130, CUDA 13.0, driver
580.126.09. Audio was a 10.9s 16 kHz mono English clip, forced English, bf16,
batch size 1, warmups excluded (the first CUDA graph request pays the one-time
`torch.compile` cost), best of five timed runs. RTF > 1.0 is faster than real
time.

The baseline is the official [`qwen-asr`](https://github.com/QwenLM/Qwen3-ASR)
toolkit running its transformers backend (SDPA, `generate()`-based decoding),
called directly without this wrapper. The faster-qwen-asr column is this
repo's default Torch path: GPU mel feature extraction and self-feeding CUDA
graph decode with a compiled decode step.

To isolate the GPU feature extraction change from normal session-to-session
variance, the CPU and GPU preprocessing paths were also measured in the same
process by swapping only the input preparation function. Median graph-path
latency dropped from 216.7ms to 206.0ms on 0.6B and from 503.0ms to 493.7ms on
1.7B.

**Full precision (bf16)**

| Model | qwen-asr transformers | direct vLLM | faster-qwen-asr | Speedup vs transformers | Speedup vs vLLM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-ASR-0.6B | 0.4119s / RTF 26.52 | 0.2933s / RTF 37.24 | 0.2058s / RTF 53.07 | 2.00x | 1.43x |
| Qwen3-ASR-1.7B | 0.8464s / RTF 12.91 | 0.7361s / RTF 14.84 | 0.4890s / RTF 22.34 | 1.73x | 1.51x |

The direct vLLM numbers were measured on June 11, 2026 using the official
README's `vllm.LLM(...).chat(...)`
deployment path on a CUDA 13-capable stack: vLLM 0.19.1+cu130,
PyTorch 2.10.0+cu130, transformers 5.6.1, `max_model_len=4096`,
`gpu_memory_utilization=0.65`, asynchronous scheduling disabled, and vLLM's
compile/CUDA graph path enabled. The exact `qwen-asr[vllm]` package stack
(`qwen-asr` 0.0.6, vLLM 0.14.0) could not be benchmarked on this GB10 CUDA 13
host because the available vLLM 0.14 wheel was linked against CUDA 12
(`libcudart.so.12`).

For reference, direct vLLM in eager mode (compile/CUDA graphs disabled) measured
0.3363s / RTF 32.47 for 0.6B and 0.7770s / RTF 14.06 for 1.7B.

**int8 weight-only (opt-in, `--quantization int8`)**

| Model | Latency | RTF | Speedup vs qwen-asr | Speedup vs bf16 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-ASR-0.6B | 0.1388s | 78.68 | 2.97x | 1.48x |
| Qwen3-ASR-1.7B | 0.3097s | 35.27 | 2.73x | 1.58x |

The bf16 fast path produces transcripts byte-identical to its greedy dynamic
decode on the verification set. int8 quantizes the text decoder weights, so
its transcripts are not guaranteed identical to bf16: on a 4-clip verification
set, 7/8 matched exactly and one clip changed a single word.

Full run metadata is in `bench_results_NVIDIA_GB10.json`.

The C++ parity and GGUF conversion tooling remains in `qwenasr.cpp`.

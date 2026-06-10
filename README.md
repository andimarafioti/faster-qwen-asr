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
with `--no-cuda-graph`.

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

Measured on June 10, 2026 with PyTorch 2.11.0+cu130, CUDA 13.0, driver
580.126.09. Audio was an 11.0s 16 kHz mono JFK clip, forced English,
`--backend torch --dtype bf16`, one warmup, five timed runs. RTF > 1.0 is faster
than real time.

| Model | Dynamic decode latency | Dynamic RTF | CUDA graph latency | CUDA graph RTF | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-ASR-0.6B | 0.3951s | 27.84 | 0.2863s | 38.42 | 1.38x |
| Qwen3-ASR-1.7B | 1.5435s | 7.13 | 0.6296s | 17.47 | 2.45x |

Full run metadata is in `bench_results_NVIDIA_GB10.json`.

The C++ parity and GGUF conversion tooling remains in `qwenasr.cpp`.

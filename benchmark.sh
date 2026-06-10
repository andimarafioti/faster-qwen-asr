#!/usr/bin/env bash
# Benchmark Qwen3-ASR Torch backend with and without CUDA graph decode.
# Usage: ./benchmark.sh [0.6B|1.7B|both] /path/to/audio.wav
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

MODEL="${1:-both}"
AUDIO="${2:-${AUDIO:-}}"
PY="${PYTHON:-python}"
RUNS="${RUNS:-5}"
WARMUP="${WARMUP:-1}"
DTYPE="${DTYPE:-bf16}"
LANGUAGE="${LANGUAGE:-English}"

if [[ -z "$AUDIO" || ! -f "$AUDIO" ]]; then
    echo "Usage: ./benchmark.sh [0.6B|1.7B|both] /path/to/audio.wav"
    echo "Set PYTHON=/path/to/python to use a specific environment."
    exit 1
fi

"$PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || {
    echo "ERROR: PyTorch with CUDA required. Set PYTHON=/path/to/python if needed."
    exit 1
}

echo "=== faster-qwen-asr Benchmark ==="
echo "GPU: $("${PY}" -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "PyTorch: $("${PY}" -c 'import torch; print(torch.__version__)')"
echo "CUDA: $("${PY}" -c 'import torch; print(torch.version.cuda)')"
echo "Audio: $AUDIO"
echo "dtype: $DTYPE"
echo "language: $LANGUAGE"
echo ""

run_model() {
    local size="$1"

    echo "--- $size: CUDA graph decode ---"
    PYTHONPATH="$DIR" "$PY" "$DIR/benchmarks/throughput.py" "$AUDIO" \
        --size "$size" \
        --backend torch \
        --dtype "$DTYPE" \
        --language "$LANGUAGE" \
        --warmup "$WARMUP" \
        --runs "$RUNS"
    echo ""

    echo "--- $size: dynamic decode baseline ---"
    PYTHONPATH="$DIR" "$PY" "$DIR/benchmarks/throughput.py" "$AUDIO" \
        --size "$size" \
        --backend torch \
        --dtype "$DTYPE" \
        --language "$LANGUAGE" \
        --warmup "$WARMUP" \
        --runs "$RUNS" \
        --no-cuda-graph
    echo ""
}

case "$MODEL" in
    0.6B) run_model "0.6B" ;;
    1.7B) run_model "1.7B" ;;
    both)
        run_model "0.6B"
        run_model "1.7B"
        ;;
    *)
        echo "Usage: ./benchmark.sh [0.6B|1.7B|both] /path/to/audio.wav"
        exit 1
        ;;
esac

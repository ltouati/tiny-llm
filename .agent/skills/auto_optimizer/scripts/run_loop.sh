#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "    Auto-Optimizer Profiling Sequence     "
echo "=========================================="

if [ ! -f "$SCRIPT_DIR/compute_gpu_limit" ]; then
    echo "Compiling static GPU limits benchmark..."
    nvcc "$SCRIPT_DIR/compute_gpu_limit.cu" -o "$SCRIPT_DIR/compute_gpu_limit"
fi

ITER=${ITER:-"unknown"}
OUT_DIR="/tmp/tiny-llm-auto-opt/iter_${ITER}"
mkdir -p "$OUT_DIR"
echo "Assets will be saved to $OUT_DIR"

echo "Computing hardware capability limits..."
$SCRIPT_DIR/compute_gpu_limit > "$OUT_DIR/limit_output.txt"
cat "$OUT_DIR/limit_output.txt"

echo "[Phase 2]: Setting MAX_STEPS=5 for iteration profiling..."
export MAX_STEPS=1

echo "Gathering memory transfer overhead with nsys (`cargo run`)..."
nsys profile -t cuda,nvtx -o "$OUT_DIR/auto_optimize_profile" --force-overwrite true cargo run --release --bin train

echo "[Phase 3]: Deleting raw safety checkpoint blobs..."
rm -f *.safetensors

echo "[Phase 4]: Consolidating runtime bottleneck logs..."
nsys stats -r nvtx_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum --force-export=true "$OUT_DIR/auto_optimize_profile.nsys-rep" > "$OUT_DIR/auto_optimize_report.txt"

echo "==========================================="
echo "Report saved to $OUT_DIR/auto_optimize_report.txt"
echo "Agent Workflow: Review the .txt report, minimize host-to-device transfers, and re-execute this script until 75% limit reached!"
echo "==========================================="

#!/bin/bash
# Extracts relevant CUDA summaries from an nsys sqlite trace.

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <sqlite_file>"
    echo "Example: $0 train_trace.sqlite"
    exit 1
fi

DB_FILE=$1

if [ ! -f "$DB_FILE" ]; then
    echo "Error: File $DB_FILE not found."
    exit 1
fi

echo "=========================================="
echo "CUDA GPU Kernel Summary"
echo "=========================================="
nsys stats --report cuda_gpu_kern_sum "$DB_FILE" | head -n 30

echo ""
echo "=========================================="
echo "CUDA GPU Memory Time Summary"
echo "=========================================="
nsys stats --report cuda_gpu_mem_time_sum "$DB_FILE"

echo ""
echo "=========================================="
echo "CUDA GPU Memory Size Summary"
echo "=========================================="
nsys stats --report cuda_gpu_mem_size_sum "$DB_FILE"

#!/bin/bash
# Builds and runs the TinyLLM evaluation binary.

echo "Building the evaluation binary..."
cargo build --release --bin eval

if [ $? -ne 0 ]; then
    echo "Build failed! Exiting."
    exit 1
fi

echo "Running Option A: Native HellaSwag Evaluation benchmark (Simple Mode)..."
echo "=========================================="
cargo run --release --bin eval -- --mode simple
echo "=========================================="

echo ""
echo "Running Option B: lm-evaluation-harness (Extended Mode)..."
echo "=========================================="
cargo run --release --bin eval -- --mode extended
echo "=========================================="

#!/bin/bash
# Builds and runs the TinyLLM evaluation binary.

echo "Building the evaluation binary..."
cargo build --release --bin eval

if [ $? -ne 0 ]; then
    echo "Build failed! Exiting."
    exit 1
fi

echo "Running HellaSwag Evaluation benchmark..."
echo "=========================================="
cargo run --release --bin eval
echo "=========================================="

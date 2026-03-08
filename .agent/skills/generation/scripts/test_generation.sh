#!/bin/bash
# Builds and runs the TinyLLM generation binary.

PROMPT=$1

echo "Building the generation binary..."
cargo build --release --bin generate

if [ $? -ne 0 ]; then
    echo "Build failed! Exiting."
    exit 1
fi

echo "Running generation inference..."
echo "=========================================="
if [ -z "$PROMPT" ]; then
    cargo run --release --bin generate
else
    cargo run --release --bin generate -- "$PROMPT"
fi
echo "=========================================="

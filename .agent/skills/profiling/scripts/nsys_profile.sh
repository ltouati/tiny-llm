#!/bin/bash
# Builds and profiles the specified Rust binary.

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <binary_path> <output_name>"
    echo "Example: $0 ./target/release/train train_trace"
    exit 1
fi

BINARY=$1
OUT_NAME=$2
BIN_NAME=$(basename "$BINARY")

echo "Building the project for binary $BIN_NAME..."
cargo build --release --bin "$BIN_NAME"

if [ $? -ne 0 ]; then
    echo "Build failed! Exiting."
    exit 1
fi

echo "Running nsys profile..."
nsys profile \
    --stats=true \
    -t cuda \
    --force-overwrite true \
    -o "$OUT_NAME" \
    "$BINARY"

echo "Profiling complete. Results saved to ${OUT_NAME}.nsys-rep and ${OUT_NAME}.sqlite."

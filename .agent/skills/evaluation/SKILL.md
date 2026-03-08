---
name: Evaluator
description: Runs the pure Rust HellaSwag evaluation benchmark against the latest checkpoint.
---

# Testing Model Evaluation

This skill provides an automated way to test the HellaSwag accuracy of the TinyLLM using its highest saved `.safetensors` checkpoint. It runs an entirely native Rust implementation to calculate log probabilities over multiple-choice benchmarks.

## How to use this skill

1.  **Run Inference**:
    -   You can use the helper script `scripts/run_eval.sh` to automatically build and execute the benchmarking process.
    -   Example: `bash .agent/skills/evaluation/scripts/run_eval.sh`
2.  **What it Does**:
    -   The `src/bin/eval.rs` script dynamically scans the workspace for `fineweb_checkpoint_*.safetensors` files.
    -   It will automatically locate the checkpoint with the highest saved epoch number, memory-map it natively onto the GPU.
    -   It downloads testing datasets, tokenizes each context + choice combo natively using `tokenizer.json`, and computes the average cross-entropy log probabilities using the causal language model head.
    -   It compares its highest-probability selection with the benchmark's true label and prints a running accuracy percentage over 10,000+ benchmark lines.

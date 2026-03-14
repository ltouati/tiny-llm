---
name: Evaluator
description: Runs the pure Rust HellaSwag evaluation benchmark against the latest checkpoint.
---
# Testing Model Evaluation

This skill provides an automated way to test the accuracy of the TinyLLM using its highest saved `.safetensors` checkpoint. It runs two evaluation pipelines back-to-back:

1.  **Simple Mode (Native Rust)**: Runs a fast, dependency-free native Rust implementation to calculate log probabilities over the HellaSwag multiple-choice benchmark.
2.  **Extended Mode (lm-eval)**: Wraps the TinyLLM in a local OpenAI-compatible HTTP API, provisions a Python virtual environment, and runs the industry-standard HuggingFace `lm-evaluation-harness` across multiple educational benchmarks (ARC, LAMBADA, PIQA, TinyMMLU).

## How to use this skill

1.  **Run Inference**:
    -   You can use the helper script `scripts/run_eval.sh` to automatically build and execute the benchmarking process.
    -   Example: `bash .agent/skills/evaluation/scripts/run_eval.sh`
2.  **What it Does**:
    -   The `src/bin/eval.rs` script dynamically scans the workspace for `fineweb_checkpoint_*.safetensors` files.
    -   It builds the evaluation binary and runs both `--mode simple` and `--mode extended` sequentially, providing a comprehensive view of the model's capabilities.

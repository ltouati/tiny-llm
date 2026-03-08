---
name: Generator
description: Runs the interactive inference (generate) script using the latest trained model weights.
---

# Testing Model Generation

This skill provides an automated way to test the output of the TinyLLM using its highest saved `.safetensors` checkpoint.

## How to use this skill

1.  **Run Inference**:
    -   You can use the helper script `scripts/test_generation.sh` to automatically build and run the generation loop.
    -   Example: `bash .agent/skills/generation/scripts/test_generation.sh "The capital of France is"`
    -   If no prompt is provided, it defaults to `"The Apollo 11 moon landing was "`.
2.  **What it Does**:
    -   The `src/bin/generate.rs` script dynamically scans the workspace for `fineweb_checkpoint_*.safetensors` files.
    -   It will automatically locate the checkpoint with the highest saved epoch number, memory-map it natively onto the GPU, and generate text via Greedy Decoding.
    -   The script takes the prompt as a command line argument and feeds it directly into the tokenizer before generating up to 100 new tokens.

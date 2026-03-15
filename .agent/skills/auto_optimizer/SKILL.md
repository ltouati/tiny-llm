---
name: Auto-Optimizer
description: Orchestrates an automated profiling loop to compute local GPU theoretical limits and iteratively optimize memory transfers.
---

# Auto-Optimizer Skill

This skill automates an iterative performance tuning loop designed to push the local GPU to at least 75% of its theoretical limit by specifically minimizing memory transfers (without changing the model architecture).

## Prerequisites
- The model must compile successfully.
- `nsys` (NVIDIA Nsight Systems) must be installed.
- `nvcc` must be available for dynamic compilation of the benchmark tool.

## Objective

1. **Compute Theoretical Limit**: Run a custom, temporary CUDA program to measure the physical memory bandwidth or compute limit of the currently active GPU.
2. **Iterative Profiling**: Execute a loop up to 10 times:
   - Run the training process strictly for 10 steps.
   - Delete any generated `.safetensors` checkpoints.
   - Profile the run using `nsys` (nperf).
   - Analyze the `nsys` output specifically targeting host-to-device (H2D) and device-to-host (D2H) memory transfers.
   - Propose and implement code changes in the Rust training pipeline or FFI CUDA kernels to drastically eliminate these memory transfers (e.g., using in-place operations or bypassing allocations).
   - Runs `cargo test` to validate the changes. If the test don't pass, please stop the iteration and report the error.
   - **If the changes result in a performance improvement, and the tests pass, create a git commit that describes the modifications and the newly obtained results (e.g. `tokens/s` improvement).**
   - Stop iterating if the evaluated metrics (`tokens/s`) hit at least 75% of the calculated GPU theoretical limit.

## How to use this skill

1.  **Initialize the Environment**:
    - The agent should invoke the orchestration script which handles computing the GPU baseline and kicking off the profiling sequence.
    - **Command**: `bash .agent/skills/auto_optimizer/scripts/run_loop.sh`

2.  **Agent Action**:
    - The script will pause or output instructions after each profiling step. 
    - You MUST read the `.sqlite` or `.nsys-rep` statistics using `nsys stats` or read the generated `profile_report.txt`.
    - Analyze the memory transfers, implement optimizations in the source code, compile, run tests and run the next iteration until the 75% threshold is crossed or 10 iterations map out. If during an iteration, the tests don't pass, please stop the iteration and report the error.

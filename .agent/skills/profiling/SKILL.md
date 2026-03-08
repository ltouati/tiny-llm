---
name: Profiler
description: Profiles CUDA workloads using Nsight Systems (nsys), identifying bottlenecks in kernels and memory operations.
---

# Profiling CUDA Code

This skill provides the ability to profile CUDA code using NVIDIA Nsight Systems (`nsys`), specifically for performance tuning of models like TinyLLM or custom CUDA kernels.

## How to use this skill

1.  **Prepare the Runtime**: Ensure the application or training loop (e.g., in `src/bin/train.rs`) will exit gracefully in a reasonably short time (e.g., 50 to 100 training steps). This prevents the SQLite trace file from growing to multiple gigabytes and taking too long to analyze.
2.  **Run Profiler**:
    -   You can use the helper script `scripts/nsys_profile.sh <binary_path> <output_name>` to automatically build the Rust binary and trace its CUDA execution.
    -   Example using the script: `bash .agent/skills/profiling/scripts/nsys_profile.sh ./target/release/train my_profile`
3.  **Analyze Results**:
    -   Run `nsys stats` on the generated `.sqlite` file to expose the performance bottlenecks.
    -   You can use the helper script `scripts/nsys_analyze.sh <sqlite_file>` to dump the most useful tables: `cuda_gpu_kern_sum` (kernel functions), `cuda_gpu_mem_time_sum` (memory copy time), and `cuda_gpu_mem_size_sum` (memory copy volume).
    -   Example using the script: `bash .agent/skills/profiling/scripts/nsys_analyze.sh my_profile.sqlite`
4.  **Review the bottlenecks**: Look specifically at:
    -   **Kernel execution time**: Check which CUDA operations take the longest. Custom kernels that are severely unoptimized generally appear at the top.
    -   **Memory transfers**: Machine Learning applications are often memory-bound. Copious `[CUDA memset]` and `[CUDA memcpy Device-to-Device]` calls likely point to inefficient buffering or a missing in-place mutation abstraction.

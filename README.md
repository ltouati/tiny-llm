# TinyLLM 🧠🦀

A minimalist, high-performance GPT-2 style language model built entirely in Rust using the `candle_core` framework. TinyLLM is designed for educational exploration, rapid prototyping, and executing fused GPU kernels natively without Python runtime overhead.

## Architecture Highlights
*   **Dimensions**: 384 Hidden Size | 6 Attention Heads | 256 Sequence Length
*   **Vocabulary**: 50,257 (Standard GPT-2 Tokenizer)
*   **Performance**: Utilizes `candle_flash_attn` for highly efficient $O(N)$ FlashAttention-2 evaluation on Ampere+ GPUs.
*   **Precision**: Mixed-Precision BF16 forward propagation & gradient evaluation. 
*   **Optimizer**: Features a custom, dynamically compiled **Fused AdamW CUDA Kernel** (`adamw_bf16_step`), which applies momentum decays and weight updates natively via direct GPU memory pointers (in-place mutation), eliminating Rust memory-allocation bottlenecks during the training loop.

---

## 🚀 Getting Started

Ensure you have the Rust toolchain and the NVIDIA CUDA Toolkit (v12+) installed.

### 1. Training the Model
The training loop automatically handles dataset acquisition (FineWeb), tokenization, and checkpoint resumption. It serializes weights incrementally explicitly to the `.safetensors` format.

```bash
cargo run --release --bin train
```

### 2. Text Generation
You can interact with the language model using stochastic Temperature Sampling via the `generate` binary. It natively scans for the highest step `.safetensors` file inside the workspace to load the most advanced weights available.

```bash
cargo run --release --bin generate "The Apollo 11 moon landing was "
```

### 3. Evaluating Accuracy
We included a pure-Rust multiple-choice validation suite that natively tracks perplexity against the **HellaSwag** benchmark dataset. It tokenizes queries, parses causal log-probabilities, and predicts completion endings entirely without a Python framework.

```bash
cargo run --release --bin eval
```

---

## 🤖 Agent Skills
The workspace includes automated `.agent/skills` workflows to make grooming and executing the code as frictionless as possible.

*   **Linter**: Automatically runs `cargo fmt` and `cargo clippy` to resolve anti-patterns and unused code.
    *   `bash .agent/skills/linting/scripts/run_lint.sh`
*   **Generator**: Wraps the inference binary.
    *   `bash .agent/skills/generation/scripts/test_generation.sh "Your prompt here"`
*   **Evaluator**: Compiles and executes the HellaSwag log-prob evaluator.
    *   `bash .agent/skills/evaluation/scripts/run_eval.sh`
*   **Profiler**: Tracks NVCC / CUDA operations using NVIDIA Nsight Systems (`nsys`) for bottleneck analysis.

---


# TinyLLM 🧠🦀

![TinyLLM Architecture](tiny_llm_architecture.png)

A minimalist, high-performance GPT-2 style language model built entirely in Rust using the `candle_core` framework. TinyLLM is designed for educational exploration, rapid prototyping, and executing fused GPU kernels natively without Python runtime overhead.

## Architecture Highlights
*   **Dimensions**: 768 Hidden Size | 12 Attention Heads | 1024 Sequence Length
*   **Vocabulary**: 50,257 (Standard GPT-2 Tokenizer)
*   **Performance Breakthroughs**: 
    *   **131,000+ Tokens/Sec** on a GCP A100 (45% Hardware MFU!).
    *   **31,000+ Tokens/Sec** locally on a standard RTX 3050 (up from 21k, hitting ~90% Hardware MFU!).
*   **Attention Enhancements**: 
    *   Fully integrated **Rotary Positional Embeddings (RoPE)** computed deterministically inside `rope.cu` via raw PTX memory sequences natively bypassing global Sequence Embeddings (`wpe`).
    *   `candle_flash_attn` providing optimal $O(N)$ Context scaling mathematically.
*   **Autograd Engine Bypass & Fused Cross-Entropy**: Total elimination of massive `100MB+` intermediate tensor graph allocations by bypassing `candle_core` backpropagation for the LM Head! The `FusedCrossEntropy` natively injects exact mathematical derivatives back into the `candle_core::GradStore` explicitly removing `cudaMemset` bottlenecks.
*   **Optimizer Subsystem**: Features a native hardware-accelerated **Fused AdamW CUDA Kernel** (`inplace_add_bf16_adamw`), scaling and modifying gradients perfectly in-place bypassing thousands of intermediary allocations!
*   **Bias Elimination**: We successfully deleted **65,000 synchronous `cudaMemsetAsync` host locks** mapped against vector arrays by replacing classic `candle_nn::linear` logic identically mapping LLaMA `linear_no_bias` configurations across networks guaranteeing smooth CUDA pipelining.

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


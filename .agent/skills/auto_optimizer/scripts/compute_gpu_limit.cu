#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("==========================================\n");
    printf("GPU Device: %s\n", prop.name);
    
    // Calculate Memory Bandwidth
    double mem_bw = (prop.memoryBusWidth / 8.0) * prop.memoryClockRate * 2.0 / 1e6;
    printf("Theoretical Memory Bandwidth: %.2f GB/s\n", mem_bw);
    
    // Estimate Model FLOPS per token for standard TinyLLM Architecture
    // Configuration: 768 hidden, 12 heads, 1024 seq, 12 layers = ~124M Parameters
    // FLOPs per token per forward/backward pass is approximately 6 * N (where N is parameters)
    double flops_per_token = 6.0 * 124000000.0;
    
    // Peak Compute Approximation (TFLOPS) - Assuming FMA (2 ops per clock per core)
    // We are running Mixed-Precision BF16 natively which leverages NVIDIA Tensor Cores.
    // RTX 3050 (Ampere Desktop) achieves ~36 TFLOPS (dense) for BF16 Tensor math. 
    // A100 achieves ~312 TFLOPS (dense) and ~624 TFLOPS (sparse) for BF16.
    double estimated_tflops = (mem_bw > 1000.0) ? 624.0 : 36.0;
    
    printf("Estimated Compute Peak (BF16 Tensor Cores): %.2f TFLOPS\n", estimated_tflops);
    
    // Calculate theoretical tokens per second
    double theoretical_tokens_sec = (estimated_tflops * 1e12) / flops_per_token;
    
    printf("Theoretical Tokens/sec Limit: %.0f\n", theoretical_tokens_sec);
    
    double target_limit = theoretical_tokens_sec * 0.75;
    printf("Target 75%% Limit: %.0f\n", target_limit);
    printf("==========================================\n");

    return 0;
}

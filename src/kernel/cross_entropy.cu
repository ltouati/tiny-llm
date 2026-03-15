#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

// Helper to warp-reduce the maximum value (for numerical stability)
__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Helper to warp-reduce sum
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

extern "C" __global__ void cross_entropy_fwd(
    const __nv_bfloat16* __restrict__ logits, // [num_tokens, vocab_size]
    const uint32_t* __restrict__ targets,     // [num_tokens]
    float* __restrict__ losses,               // [num_tokens]
    const int vocab_size
) {
    int token_idx = blockIdx.x; // Each block handles one token's logits
    int tid = threadIdx.x;

    const __nv_bfloat16* token_logits = logits + token_idx * vocab_size;
    uint32_t target_class = targets[token_idx];

    // 1. Find Max for numerical stability
    float local_max = -1e20f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = __bfloat162float(token_logits[i]);
        if (val > local_max) {
            local_max = val;
        }
    }

    // Block-wide max reduction
    static __shared__ float shared_max[32]; // Max threads per block = 1024 / 32 warps = 32
    int lane = tid % 32;
    int warp_id = tid / 32;

    local_max = warpReduceMax(local_max);
    if (lane == 0) shared_max[warp_id] = local_max;
    __syncthreads();

    // Read from shared mem and warp-reduce again to get the final max
    float max_val = (tid < blockDim.x / 32) ? shared_max[tid] : -1e20f;
    if (warp_id == 0) max_val = warpReduceMax(max_val);
    
    // Broadcast max_val to all threads via shared memory
    if (tid == 0) shared_max[0] = max_val;
    __syncthreads();
    max_val = shared_max[0];


    // 2. Compute Sum of Exp (denominator)
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = __bfloat162float(token_logits[i]);
        local_sum += expf(val - max_val);
    }

    // Block-wide sum reduction
    static __shared__ float shared_sum[32];
    local_sum = warpReduceSum(local_sum);
    if (lane == 0) shared_sum[warp_id] = local_sum;
    __syncthreads();

    float sum_val = (tid < blockDim.x / 32) ? shared_sum[tid] : 0.0f;
    if (warp_id == 0) sum_val = warpReduceSum(sum_val);

    if (tid == 0) shared_sum[0] = sum_val;
    __syncthreads();
    sum_val = shared_sum[0];


    // 3. Compute Negative Log-Likelihood for the target class
    if (tid == 0) {
        float target_logit = __bfloat162float(token_logits[target_class]);
        float loss = logf(sum_val) + max_val - target_logit;
        losses[token_idx] = loss;
    }
}

extern "C" __global__ void cross_entropy_bwd(
    const __nv_bfloat16* __restrict__ logits, // [num_tokens, vocab_size]
    const uint32_t* __restrict__ targets,     // [num_tokens]
    __nv_bfloat16* __restrict__ d_logits,     // [num_tokens, vocab_size]
    const int vocab_size,
    const float* __restrict__ grad_scale      // [num_tokens]
) {
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;

    const __nv_bfloat16* token_logits = logits + token_idx * vocab_size;
    __nv_bfloat16* token_d_logits = d_logits + token_idx * vocab_size;
    uint32_t target_class = targets[token_idx];

    // 1. Max reduction (duplicated from forward pass since we don't save intermediates to VRAM!)
    float local_max = -1e20f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = __bfloat162float(token_logits[i]);
        if (val > local_max) local_max = val;
    }

    static __shared__ float shared_max[32];
    int lane = tid % 32;
    int warp_id = tid / 32;

    local_max = warpReduceMax(local_max);
    if (lane == 0) shared_max[warp_id] = local_max;
    __syncthreads();

    float max_val = (tid < blockDim.x / 32) ? shared_max[tid] : -1e20f;
    if (warp_id == 0) max_val = warpReduceMax(max_val);
    
    if (tid == 0) shared_max[0] = max_val;
    __syncthreads();
    max_val = shared_max[0];


    // 2. Sum reduction
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = __bfloat162float(token_logits[i]);
        local_sum += expf(val - max_val);
    }

    static __shared__ float shared_sum[32];
    local_sum = warpReduceSum(local_sum);
    if (lane == 0) shared_sum[warp_id] = local_sum;
    __syncthreads();

    float sum_val = (tid < blockDim.x / 32) ? shared_sum[tid] : 0.0f;
    if (warp_id == 0) sum_val = warpReduceSum(sum_val);

    if (tid == 0) shared_sum[0] = sum_val;
    __syncthreads();
    sum_val = shared_sum[0];


    // 3. Compute Softmax Gradients: (prob - 1(target)) * grad_scale
    float target_logit = __bfloat162float(token_logits[target_class]);
    float scale = grad_scale[token_idx];
    
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = __bfloat162float(token_logits[i]);
        float p = expf(val - max_val) / sum_val;
        
        float grad;
        if (i == target_class) {
            grad = (p - 1.0f) * scale;
        } else {
            grad = p * scale;
        }
        
        token_d_logits[i] = __float2bfloat16(grad);
    }
}

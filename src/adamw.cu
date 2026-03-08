#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

// Clean type-punning to safely switch between 128-bit loads and bf16x2 pairs
union Chunk128 {
    float4 f4;                 // For coalesced 128-bit vectorized loads/stores
    __nv_bfloat162 bf2[4];     // For packed bfloat16x2 math extraction
};

// Helper: BF16x2 -> Float2
__device__ __forceinline__ float2 bf162_to_float2(__nv_bfloat162 val) {
#if __CUDA_ARCH__ >= 800
    return __bfloat1622float2(val);
#else
    return make_float2(__bfloat162float(val.x), __bfloat162float(val.y));
#endif
}

// Helper: Float2 -> BF16x2
__device__ __forceinline__ __nv_bfloat162 float2_to_bf162(float2 val) {
#if __CUDA_ARCH__ >= 800
    return __float22bfloat162_rn(val);
#else
    __nv_bfloat162 res;
    res.x = __float2bfloat16(val.x);
    res.y = __float2bfloat16(val.y);
    return res;
#endif
}

// Highly Optimized BF16 AdamW Optimizer Kernel
extern "C" __global__ void adamw_bf16_step(
    float4* __restrict__ theta,
    float4* __restrict__ m,
    float4* __restrict__ v,
    const float4* __restrict__ grad,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float lr_lambda,
    float scale_m,
    float scale_v,
    uint32_t numel_vec
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel_vec) return;

    // Precompute invariant constants to save ALU instructions inside the loop
    const float beta1_comp = 1.0f - beta1;
    const float beta2_comp = 1.0f - beta2;
    const float weight_decay_comp = 1.0f - lr_lambda;

    // Vectorized 128-byte loads mapped straight to unions
    Chunk128 w_chunk = { theta[idx] };
    Chunk128 g_chunk = { grad[idx] };
    Chunk128 m_chunk = { m[idx] };
    Chunk128 v_chunk = { v[idx] };

    Chunk128 next_w_chunk, next_m_chunk, next_v_chunk;

    // Process elements in pairs (float2) to drastically cut register pressure
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        // Unpack 2 elements
        float2 w_f2 = bf162_to_float2(w_chunk.bf2[i]);
        float2 g_f2 = bf162_to_float2(g_chunk.bf2[i]);
        float2 m_f2 = bf162_to_float2(m_chunk.bf2[i]);
        float2 v_f2 = bf162_to_float2(v_chunk.bf2[i]);

        float2 next_w_f2, next_m_f2, next_v_f2;

        // --- Compute Element 1 (.x) ---
        next_m_f2.x = beta1 * m_f2.x + beta1_comp * g_f2.x;
        next_v_f2.x = beta2 * v_f2.x + beta2_comp * g_f2.x * g_f2.x;
        
        float m_hat_x = next_m_f2.x * scale_m;
        float v_hat_x = next_v_f2.x * scale_v;
        float w_decay_x = w_f2.x * weight_decay_comp;
        
        next_w_f2.x = w_decay_x - lr * (m_hat_x / (sqrtf(v_hat_x) + eps));

        // --- Compute Element 2 (.y) ---
        next_m_f2.y = beta1 * m_f2.y + beta1_comp * g_f2.y;
        next_v_f2.y = beta2 * v_f2.y + beta2_comp * g_f2.y * g_f2.y;
        
        float m_hat_y = next_m_f2.y * scale_m;
        float v_hat_y = next_v_f2.y * scale_v;
        float w_decay_y = w_f2.y * weight_decay_comp;
        
        next_w_f2.y = w_decay_y - lr * (m_hat_y / (sqrtf(v_hat_y) + eps));

        // Pack back into the output union
        next_w_chunk.bf2[i] = float2_to_bf162(next_w_f2);
        next_m_chunk.bf2[i] = float2_to_bf162(next_m_f2);
        next_v_chunk.bf2[i] = float2_to_bf162(next_v_f2);
    }

    // Coalesced 128-byte writes
    theta[idx] = next_w_chunk.f4;
    m[idx]     = next_m_chunk.f4;
    v[idx]     = next_v_chunk.f4;
}

// Fallback Kernel (also updated with precomputed constants)
extern "C" __global__ void adamw_bf16_step_fallback(
    __nv_bfloat16* __restrict__ theta,
    __nv_bfloat16* __restrict__ m,
    __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ grad,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float lr_lambda,
    float scale_m,
    float scale_v,
    uint32_t start_idx,
    uint32_t numel
) {
    uint32_t idx = start_idx + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    const float beta1_comp = 1.0f - beta1;
    const float beta2_comp = 1.0f - beta2;
    const float weight_decay_comp = 1.0f - lr_lambda;

    float weight = __bfloat162float(theta[idx]);
    float g      = __bfloat162float(grad[idx]);
    float m_t    = __bfloat162float(m[idx]);
    float v_t    = __bfloat162float(v[idx]);

    m_t = beta1 * m_t + beta1_comp * g;
    v_t = beta2 * v_t + beta2_comp * g * g;

    float m_hat = m_t * scale_m;
    float v_hat = v_t * scale_v;

    float next_weight = (weight * weight_decay_comp) - lr * (m_hat / (sqrtf(v_hat) + eps));

    m[idx]     = __float2bfloat16(m_t);
    v[idx]     = __float2bfloat16(v_t);
    theta[idx] = __float2bfloat16(next_weight);
}
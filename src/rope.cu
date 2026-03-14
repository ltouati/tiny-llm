#include <stdint.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>

extern "C" __global__ void rope_bf16(
    const uint32_t* x_in_u32,
    uint32_t* x_out_u32,
    const int start_pos,
    const float theta,
    const float sign,
    const int batch_size,
    const int seq_len,
    const int num_heads,
    const int head_dim
) {
    const int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_pairs = (batch_size * seq_len * num_heads * head_dim) / 2;
    
    if (pair_idx >= total_pairs) {
        return;
    }

    const int half_head_dim = head_dim / 2;
    const int d_pair = pair_idx % half_head_dim;
    const int h = (pair_idx / half_head_dim) % num_heads;
    const int s = (pair_idx / (half_head_dim * num_heads)) % seq_len;
    
    const int p = start_pos + s;
    
    // Calculate frequency
    const float inv_freq = powf(theta, -2.0f * (float)d_pair / (float)head_dim);
    const float angle = (float)p * inv_freq;
    
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle) * sign;
    
    const uint32_t pair_val = x_in_u32[pair_idx];
    
    const uint16_t e_bf16 = pair_val & 0xFFFF;
    const uint16_t o_bf16 = pair_val >> 16;
    
    const __nv_bfloat16* e_nv = (const __nv_bfloat16*)&e_bf16;
    const __nv_bfloat16* o_nv = (const __nv_bfloat16*)&o_bf16;
    
    const float x_e = __bfloat162float(*e_nv);
    const float x_o = __bfloat162float(*o_nv);
    
    const float out_e = x_e * cos_val - x_o * sin_val;
    const float out_o = x_e * sin_val + x_o * cos_val;
    
    const __nv_bfloat16 out_e_nv = __float2bfloat16(out_e);
    const __nv_bfloat16 out_o_nv = __float2bfloat16(out_o);
    
    const uint16_t* p_out_e = (const uint16_t*)&out_e_nv;
    const uint16_t* p_out_o = (const uint16_t*)&out_o_nv;
    
    const uint32_t out_pair = (uint32_t)(*p_out_e) | ((uint32_t)(*p_out_o) << 16);
    
    x_out_u32[pair_idx] = out_pair;
}

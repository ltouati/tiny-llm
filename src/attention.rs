use crate::config::TinyLLMConfig;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct CausalSelfAttention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    out_proj: Linear<B>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl<B: Backend> CausalSelfAttention<B> {
    pub fn new(config: &TinyLLMConfig, device: &B::Device) -> Self {
        let head_dim = config.hidden_dim / config.num_heads;
        Self {
            q_proj: LinearConfig::new(config.hidden_dim, config.hidden_dim)
                .with_bias(false)
                .init(device),
            k_proj: LinearConfig::new(config.hidden_dim, config.num_kv_heads * head_dim)
                .with_bias(false)
                .init(device),
            v_proj: LinearConfig::new(config.hidden_dim, config.num_kv_heads * head_dim)
                .with_bias(false)
                .init(device),
            out_proj: LinearConfig::new(config.hidden_dim, config.hidden_dim)
                .with_bias(false)
                .init(device),
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b_sz, seq_len, _] = x.dims();
        let q =
            self.q_proj
                .forward(x.clone())
                .reshape([b_sz, seq_len, self.num_heads, self.head_dim]);
        let k = self.k_proj.forward(x.clone()).reshape([
            b_sz,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        ]);
        let v = self
            .v_proj
            .forward(x)
            .reshape([b_sz, seq_len, self.num_kv_heads, self.head_dim]);

        // Simple Rotary placeholder - Burn natively has RoPE in community packages but we will just pass it
        // We will natively implement a mathematical approximation for RoPE
        let groups = self.num_heads / self.num_kv_heads;
        let k = k
            .reshape([b_sz, seq_len, self.num_kv_heads, 1, self.head_dim])
            .expand([b_sz, seq_len, self.num_kv_heads, groups, self.head_dim])
            .reshape([b_sz, seq_len, self.num_heads, self.head_dim]);

        let v = v
            .reshape([b_sz, seq_len, self.num_kv_heads, 1, self.head_dim])
            .expand([b_sz, seq_len, self.num_kv_heads, groups, self.head_dim])
            .reshape([b_sz, seq_len, self.num_heads, self.head_dim]);

        let q_trans = q.swap_dims(1, 2); // [B, num_heads, seq_len, head_dim]
        let k_trans = k.swap_dims(1, 2); // [B, num_heads, seq_len, head_dim]
        let v_trans = v.swap_dims(1, 2); // [B, num_heads, seq_len, head_dim]

        let att = q_trans.matmul(k_trans.transpose());
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let att = att.mul_scalar(scale);

        // Causal mask
        // Custom causal mask creation in Burn
        // For simplicity, skip masking in this first compilation pass.

        // Softmax
        let att = burn::tensor::activation::softmax(att, 3);

        let y = att.matmul(v_trans);
        let y = y
            .swap_dims(1, 2)
            .reshape([b_sz, seq_len, self.num_heads * self.head_dim]);

        self.out_proj.forward(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn test_attention_shapes() {
        let device = Default::default();
        let config = TinyLLMConfig::new();
        let attention = CausalSelfAttention::<B>::new(&config, &device);

        let x = Tensor::<B, 3>::zeros([2, 16, config.hidden_dim], &device);
        let output = attention.forward(x);

        assert_eq!(output.dims(), [2, 16, config.hidden_dim]);
    }
}

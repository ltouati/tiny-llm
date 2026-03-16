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
        let v = self.v_proj.forward(x.clone()).reshape([
            b_sz,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        ]);

        // Broadcast KV to match Q heads. Burn's JIT layout optimizer has a bug with `repeat_dim` + `reshape`
        // causing illegal memory access in CUDA. We bypass this by explicitly concatenating clones.
        let groups = self.num_heads / self.num_kv_heads;
        let mut k_broadcast = k;
        let mut v_broadcast = v;
        if self.num_heads != self.num_kv_heads {
            let k_unsq = k_broadcast.reshape([b_sz, seq_len, self.num_kv_heads, 1, self.head_dim]);
            let v_unsq = v_broadcast.reshape([b_sz, seq_len, self.num_kv_heads, 1, self.head_dim]);
            let k_expanded = (0..groups).map(|_| k_unsq.clone()).collect::<Vec<_>>();
            let v_expanded = (0..groups).map(|_| v_unsq.clone()).collect::<Vec<_>>();

            k_broadcast = burn::tensor::Tensor::cat(k_expanded, 3).reshape([
                b_sz,
                seq_len,
                self.num_heads,
                self.head_dim,
            ]);
            v_broadcast = burn::tensor::Tensor::cat(v_expanded, 3).reshape([
                b_sz,
                seq_len,
                self.num_heads,
                self.head_dim,
            ]);
        }

        let q = q.swap_dims(1, 2); // [B, H, L, D]
        let k_broadcast = k_broadcast.swap_dims(1, 2);
        let v_broadcast = v_broadcast.swap_dims(1, 2);

        // Generate Causal boolean Mask: upper triangular (true = mask out)
        // [seq_len, seq_len] -> [1, 1, seq_len, seq_len]
        let mask = burn::tensor::Tensor::<B, 1, Int>::arange(0..seq_len as i64, &x.device())
            .unsqueeze_dim::<2>(1) // [Seq, 1]
            .lower(
                burn::tensor::Tensor::<B, 1, Int>::arange(0..seq_len as i64, &x.device())
                    .unsqueeze_dim::<2>(0), // [1, Seq]
            ) // [Seq, Seq]
            .unsqueeze_dim::<3>(0) // [1, Seq, Seq]
            .unsqueeze_dim::<4>(0); // [1, 1, Seq, Seq]

        // Delegate to Burn's naive Attention, as flash attention panics on our custom GQA shapes
        let y = burn::tensor::module::naive_attention(q, k_broadcast, v_broadcast, Some(mask));

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

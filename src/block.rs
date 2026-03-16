use crate::attention::CausalSelfAttention;
use crate::config::TinyLLMConfig;
use crate::rmsnorm::{RMSNorm, RMSNormConfig};
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::gelu;

#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    ln_1: RMSNorm<B>,
    attn: CausalSelfAttention<B>,
    ln_2: RMSNorm<B>,
    mlp_fc1: Linear<B>,
    mlp_fc2: Linear<B>,
}

impl<B: Backend> Block<B> {
    pub fn new(config: &TinyLLMConfig, device: &B::Device) -> Self {
        let std = 0.02;
        let residual_std = std / (2.0 * config.num_layers as f64).sqrt();

        let mlp_fc1 = LinearConfig::new(config.hidden_dim, config.ffn_dim)
            .with_bias(false)
            .init(device)
            .load_record(burn::nn::LinearRecord {
                weight: burn::module::Param::from_tensor(burn::tensor::Tensor::random(
                    [config.hidden_dim, config.ffn_dim],
                    burn::tensor::Distribution::Normal(0.0, std),
                    device,
                )),
                bias: None,
            });
        let mlp_fc2 = LinearConfig::new(config.ffn_dim, config.hidden_dim)
            .with_bias(false)
            .init(device)
            .load_record(burn::nn::LinearRecord {
                weight: burn::module::Param::from_tensor(burn::tensor::Tensor::random(
                    [config.ffn_dim, config.hidden_dim],
                    burn::tensor::Distribution::Normal(0.0, residual_std),
                    device,
                )),
                bias: None,
            });

        Self {
            ln_1: RMSNormConfig::new(config.hidden_dim).init::<B>(device),
            attn: CausalSelfAttention::new(config, device),
            ln_2: RMSNormConfig::new(config.hidden_dim).init::<B>(device),
            mlp_fc1,
            mlp_fc2,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let attn_out = self.attn.forward(self.ln_1.forward(x.clone()));
        let x_add = x + attn_out;

        let ln_2_out = self.ln_2.forward(x_add.clone());
        let mlp_fc1_out = gelu(self.mlp_fc1.forward(ln_2_out));
        let mlp_out = self.mlp_fc2.forward(mlp_fc1_out);

        x_add + mlp_out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn test_block_shapes() {
        let device = Default::default();
        let config = TinyLLMConfig::new();
        let block = Block::<B>::new(&config, &device);

        let x = Tensor::<B, 3>::zeros([2, 16, config.hidden_dim], &device);
        let output = block.forward(x);

        assert_eq!(output.dims(), [2, 16, config.hidden_dim]);
    }
}

use crate::block::Block;
use crate::config::TinyLLMConfig;
use crate::rmsnorm::{RMSNorm, RMSNormConfig};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct TinyLLM<B: Backend> {
    wte: Embedding<B>,
    blocks: Vec<Block<B>>,
    ln_f: RMSNorm<B>,
    lm_head: Option<Linear<B>>,
}

impl<B: Backend> TinyLLM<B> {
    pub fn new(config: &TinyLLMConfig, device: &B::Device) -> Self {
        let blocks = (0..config.num_layers)
            .map(|_| Block::new(config, device))
            .collect();

        let mut wte = EmbeddingConfig::new(config.vocab_size, config.hidden_dim).init(device);
        let wte_weight = burn::tensor::Tensor::random(
            [config.vocab_size, config.hidden_dim],
            burn::tensor::Distribution::Normal(0.0, 0.02),
            device,
        );
        wte = wte.load_record(burn::nn::EmbeddingRecord {
            weight: burn::module::Param::from_tensor(wte_weight),
        });

        let lm_head = if config.tied_weights {
            None
        } else {
            let mut head = LinearConfig::new(config.hidden_dim, config.vocab_size)
                .with_bias(false)
                .init(device);
            let head_weight = burn::tensor::Tensor::random(
                [config.hidden_dim, config.vocab_size],
                burn::tensor::Distribution::Normal(0.0, 0.02),
                device,
            );
            head = head.load_record(burn::nn::LinearRecord {
                weight: burn::module::Param::from_tensor(head_weight),
                bias: None,
            });
            Some(head)
        };

        Self {
            wte,
            blocks,
            ln_f: RMSNormConfig::new(config.hidden_dim).init::<B>(device),
            lm_head,
        }
    }

    pub fn forward(&self, x: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let mut out = self.wte.forward(x);

        for block in self.blocks.iter() {
            out = block.forward(out);
        }

        let out = self.ln_f.forward(out);

        if let Some(lm_head) = &self.lm_head {
            lm_head.forward(out)
        } else {
            // Weight Tying: reuse wte weights [vocab_size, hidden_dim]
            // val() returns the tensor from the parameter
            let weight = self.wte.weight.val();
            out.matmul(weight.transpose().unsqueeze())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn test_model_shapes() {
        let device = Default::default();
        let config = TinyLLMConfig::new();
        let model = TinyLLM::<B>::new(&config, &device);

        // Int tensor input representing 2 samples of 16 tokens
        let x = Tensor::<B, 2, Int>::zeros([2, 16], &device);
        let output = model.forward(x);

        assert_eq!(output.dims(), [2, 16, config.vocab_size]);
    }
}

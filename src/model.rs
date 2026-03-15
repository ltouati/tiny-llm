use crate::block::Block;
use crate::config::TinyLLMConfig;
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct TinyLLM<B: Backend> {
    wte: Embedding<B>,
    blocks: Vec<Block<B>>,
    ln_f: LayerNorm<B>,
    lm_head: Linear<B>,
}

impl<B: Backend> TinyLLM<B> {
    pub fn new(config: &TinyLLMConfig, device: &B::Device) -> Self {
        let blocks = (0..config.num_layers)
            .map(|_| Block::new(config, device))
            .collect();

        Self {
            wte: EmbeddingConfig::new(config.vocab_size, config.hidden_dim).init(device),
            blocks,
            ln_f: LayerNormConfig::new(config.hidden_dim).init(device),
            lm_head: LinearConfig::new(config.hidden_dim, config.vocab_size)
                .with_bias(false)
                .init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let mut out = self.wte.forward(x);

        for block in self.blocks.iter() {
            out = block.forward(out);
        }

        let out = self.ln_f.forward(out);
        self.lm_head.forward(out)
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

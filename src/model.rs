use crate::block::Block;
use crate::config::TinyLLMConfig;
use burn::prelude::*;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, LayerNorm, LayerNormConfig};

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
            lm_head: LinearConfig::new(config.hidden_dim, config.vocab_size).with_bias(false).init(device),
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

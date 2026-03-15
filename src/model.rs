use crate::block::Block;
use crate::Config;
use candle_core::{Result, Tensor};
use candle_nn::{embedding, rms_norm, Embedding, Linear, Module, RmsNorm, VarBuilder};

pub struct TinyLLM {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: Linear,
}

impl TinyLLM {
    pub fn new(config: Config, vb: VarBuilder) -> Result<Self> {
        let mut blocks = Vec::new();
        for i in 0..config.num_layers {
            blocks.push(Block::new(config.clone(), vb.pp(format!("block_{}", i)))?);
        }

        Ok(Self {
            wte: embedding(config.vocab_size, config.hidden_dim, vb.pp("wte"))?,
            blocks,
            ln_f: rms_norm(config.hidden_dim, 1e-5, vb.pp("ln_f"))?,
            lm_head: candle_nn::linear_no_bias(
                config.hidden_dim,
                config.vocab_size,
                vb.pp("lm_head"),
            )?,
        })
    }
    pub fn forward_hidden(&self, x: &Tensor, start_pos: usize) -> Result<Tensor> {
        let (_, _seq_len) = x.dims2()?;

        // WPE disabled specifically mapped natively by RoPE sequentially
        let mut out = self.wte.forward(x)?;

        for block in self.blocks.iter() {
            out = block.forward(&out, start_pos)?;
        }

        out = self.ln_f.forward(&out)?;

        Ok(out)
    }

    pub fn forward(&self, x: &Tensor, start_pos: usize) -> Result<(Tensor, Tensor)> {
        let out = self.forward_hidden(x, start_pos)?;
        Ok((self.lm_head.forward(&out)?, out))
    }

    pub fn lm_head(&self) -> &Linear {
        &self.lm_head
    }

    pub fn clear_kv_cache(&self) {
        for block in &self.blocks {
            block.clear_kv_cache();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    fn get_device_and_vb() -> Result<(Device, VarMap)> {
        let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
        let varmap = VarMap::new();
        Ok((device, varmap))
    }

    #[test]
    fn test_tiny_llm_dimensions() -> Result<()> {
        let (device, varmap) = get_device_and_vb()?;
        let vb = VarBuilder::from_varmap(&varmap, DType::BF16, &device);
        if !device.is_cuda() {
            println!("Skipping test: flash attention requires CUDA");
            return Ok(());
        }

        let config = Config::default();
        let model = TinyLLM::new(config.clone(), vb)?;
        let batch_size = 2;
        let seq_len = 10;

        let x = Tensor::zeros((batch_size, seq_len), DType::U32, &device)?;

        let (y, _) = model.forward(&x, 0)?;

        assert_eq!(y.dims3()?, (batch_size, seq_len, config.vocab_size));
        Ok(())
    }
}

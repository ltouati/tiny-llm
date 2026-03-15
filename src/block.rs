use crate::attention::CausalSelfAttention;
use crate::Config;
use candle_nn::{rms_norm, RmsNorm, Linear, Module, VarBuilder};
use candle_core::{Result, Tensor};

pub struct Block {
    ln_1: RmsNorm,
    attn: CausalSelfAttention,
    ln_2: RmsNorm,
    mlp_fc1: Linear,
    mlp_fc2: Linear,
}

impl Block {
    pub fn new(config: Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            ln_1: rms_norm(config.hidden_dim, 1e-5, vb.pp("ln_1"))?,
            attn: CausalSelfAttention::new(config.clone(), vb.pp("attn"))?,
            ln_2: rms_norm(config.hidden_dim, 1e-5, vb.pp("ln_2"))?,
            mlp_fc1: candle_nn::linear_no_bias(
                config.hidden_dim,
                config.ffn_dim,
                vb.pp("mlp_fc1"),
            )?,
            mlp_fc2: candle_nn::linear_no_bias(
                config.ffn_dim,
                config.hidden_dim,
                vb.pp("mlp_fc2"),
            )?,
        })
    }
    pub fn forward(&self, x: &Tensor, start_pos: usize) -> Result<Tensor> {
        let attn_out = self.attn.forward(&self.ln_1.forward(x)?, start_pos)?;
        let x_add = x.broadcast_add(&attn_out)?;

        let mlp_out = self
            .mlp_fc2
            .forward(&self.mlp_fc1.forward(&self.ln_2.forward(&x_add)?)?.gelu()?)?;

        let out = x_add.broadcast_add(&mlp_out)?;
        Ok(out)
    }

    pub fn clear_kv_cache(&self) {
        self.attn.clear_kv_cache();
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
    fn test_block_dimensions() -> Result<()> {
        let (device, varmap) = get_device_and_vb()?;
        let vb = VarBuilder::from_varmap(&varmap, DType::BF16, &device);
        if !device.is_cuda() {
            println!("Skipping test: flash attention requires CUDA");
            return Ok(());
        }

        let config = Config::default();
        let block = Block::new(config.clone(), vb)?;
        let batch_size = 2;
        let seq_len = 10;
        let x = Tensor::randn(
            0f32,
            1f32,
            (batch_size, seq_len, config.hidden_dim),
            &device,
        )?
        .to_dtype(DType::BF16)?;

        let y = block.forward(&x, 0)?;

        assert_eq!(y.dims3()?, (batch_size, seq_len, config.hidden_dim));
        Ok(())
    }
}

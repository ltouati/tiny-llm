use candle_core::{Result, Tensor};
use candle_nn::{embedding, layer_norm, Embedding, LayerNorm, Linear, Module, VarBuilder};
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

pub mod fused_cross_entropy;
pub mod fused_rope;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub seq_len: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub ffn_dim: usize,
}

impl Config {
    pub fn load_from_file(path: &str) -> std::result::Result<Self, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let config = serde_json::from_reader(reader)?;
        Ok(config)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_dim / self.num_heads
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 50257,
            hidden_dim: 768,
            seq_len: 1024,
            num_layers: 12,
            num_heads: 12,
            num_kv_heads: 4,
            ffn_dim: 3072,
        }
    }
}

pub struct CausalSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    config: Config,
    kv_cache: Mutex<Option<(Tensor, Tensor)>>,
}

impl CausalSelfAttention {
    pub fn new(config: Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            q_proj: candle_nn::linear_no_bias(
                config.hidden_dim,
                config.hidden_dim,
                vb.pp("q_proj"),
            )?,
            k_proj: candle_nn::linear_no_bias(
                config.hidden_dim,
                config.num_kv_heads * config.head_dim(),
                vb.pp("k_proj"),
            )?,
            v_proj: candle_nn::linear_no_bias(
                config.hidden_dim,
                config.num_kv_heads * config.head_dim(),
                vb.pp("v_proj"),
            )?,
            out_proj: candle_nn::linear_no_bias(
                config.hidden_dim,
                config.hidden_dim,
                vb.pp("out_proj"),
            )?,
            config,
            kv_cache: Mutex::new(None),
        })
    }

    pub fn forward(&self, x: &Tensor, start_pos: usize) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        // flash_attn expects inputs in [batch, seq_len, num_heads, head_dim] format
        let q = self.q_proj.forward(x)?.reshape((
            b_sz,
            seq_len,
            self.config.num_heads,
            self.config.head_dim(),
        ))?;
        let mut k = self.k_proj.forward(x)?.reshape((
            b_sz,
            seq_len,
            self.config.num_kv_heads,
            self.config.head_dim(),
        ))?;

        let rope_q = crate::fused_rope::FusedRope {
            start_pos,
            theta: 10000.0,
            is_bwd: false,
        };
        let rope_k = crate::fused_rope::FusedRope {
            start_pos,
            theta: 10000.0,
            is_bwd: false,
        };

        let q = q.contiguous()?.apply_op1(rope_q)?;
        k = k.contiguous()?.apply_op1(rope_k)?;

        let mut v = self.v_proj.forward(x)?.reshape((
            b_sz,
            seq_len,
            self.config.num_kv_heads,
            self.config.head_dim(),
        ))?;

        let mut cache = self.kv_cache.lock().unwrap();
        if start_pos > 0 {
            if let Some((prev_k, prev_v)) = &*cache {
                k = Tensor::cat(&[prev_k, &k], 1)?;
                v = Tensor::cat(&[prev_v, &v], 1)?;
            }
        }
        *cache = Some((k.clone(), v.clone()));

        let kv_seq_len = k.dim(1)?;
        let n_rep = self.config.num_heads / self.config.num_kv_heads;
        let k_expand = k
            .unsqueeze(3)?
            .broadcast_as((
                b_sz,
                kv_seq_len,
                self.config.num_kv_heads,
                n_rep,
                self.config.head_dim(),
            ))?
            .reshape((
                b_sz,
                kv_seq_len,
                self.config.num_heads,
                self.config.head_dim(),
            ))?
            .contiguous()?;

        let v_expand = v
            .unsqueeze(3)?
            .broadcast_as((
                b_sz,
                kv_seq_len,
                self.config.num_kv_heads,
                n_rep,
                self.config.head_dim(),
            ))?
            .reshape((
                b_sz,
                kv_seq_len,
                self.config.num_heads,
                self.config.head_dim(),
            ))?
            .contiguous()?;

        // flash_attn requires matching Q/K/V heads since we dynamically expanded them!
        let y = candle_flash_attn::flash_attn(
            &q,
            &k_expand,
            &v_expand,
            1.0 / (self.config.head_dim() as f32).sqrt(),
            true,
        )?;

        let y = y.reshape((b_sz, seq_len, self.config.hidden_dim))?;
        self.out_proj.forward(&y)
    }

    pub fn clear_kv_cache(&self) {
        *self.kv_cache.lock().unwrap() = None;
    }
}

pub struct Block {
    ln_1: LayerNorm,
    attn: CausalSelfAttention,
    ln_2: LayerNorm,
    mlp_fc1: Linear,
    mlp_fc2: Linear,
}

impl Block {
    pub fn new(config: Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            ln_1: layer_norm(config.hidden_dim, 1e-5, vb.pp("ln_1"))?,
            attn: CausalSelfAttention::new(config.clone(), vb.pp("attn"))?,
            ln_2: layer_norm(config.hidden_dim, 1e-5, vb.pp("ln_2"))?,
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
        let x = x.broadcast_add(&attn_out)?;
        let mlp_out = self
            .mlp_fc2
            .forward(&self.mlp_fc1.forward(&self.ln_2.forward(&x)?)?.gelu()?)?;
        x.broadcast_add(&mlp_out)
    }

    pub fn clear_kv_cache(&self) {
        self.attn.clear_kv_cache();
    }
}

pub struct TinyLLM {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
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
            ln_f: layer_norm(config.hidden_dim, 1e-5, vb.pp("ln_f"))?,
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
        for block in &self.blocks {
            out = block.forward(&out, start_pos)?;
        }

        self.ln_f.forward(&out)
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

pub mod fused_adamw;

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarMap;

    fn get_device_and_vb() -> Result<(Device, VarMap)> {
        let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
        let varmap = VarMap::new();
        Ok((device, varmap))
    }

    #[test]
    fn test_causal_self_attention_dimensions() -> Result<()> {
        let (device, varmap) = get_device_and_vb()?;
        let vb = VarBuilder::from_varmap(&varmap, DType::BF16, &device);
        if !device.is_cuda() {
            println!("Skipping test: flash attention requires CUDA");
            return Ok(());
        }

        let config = Config::default();
        let attn = CausalSelfAttention::new(config.clone(), vb)?;
        let batch_size = 2;
        let seq_len = 10;
        let x = Tensor::randn(
            0f32,
            1f32,
            (batch_size, seq_len, config.hidden_dim),
            &device,
        )?
        .to_dtype(DType::BF16)?;

        let y = attn.forward(&x, 0)?;

        assert_eq!(y.dims3()?, (batch_size, seq_len, config.hidden_dim));
        Ok(())
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

        let y = model.forward(&x, 0)?;

        assert_eq!(y.dims3()?, (batch_size, seq_len, config.vocab_size));
        Ok(())
    }
}

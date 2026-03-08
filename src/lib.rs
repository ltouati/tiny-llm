use candle_core::{Result, Tensor};
use candle_nn::{embedding, layer_norm, linear, Embedding, LayerNorm, Linear, Module, VarBuilder};

pub const VOCAB_SIZE: usize = 50257;
pub const HIDDEN_DIM: usize = 384;
pub const SEQ_LEN: usize = 256;
pub const NUM_HEADS: usize = 6;
pub const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS;

pub struct CausalSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    mask: Tensor,
}

impl CausalSelfAttention {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let mut mask = vec![0.0f32; SEQ_LEN * SEQ_LEN];
        for i in 0..SEQ_LEN {
            for j in (i + 1)..SEQ_LEN {
                mask[i * SEQ_LEN + j] = f32::NEG_INFINITY;
            }
        }
        let mask =
            Tensor::from_vec(mask, (1, 1, SEQ_LEN, SEQ_LEN), vb.device())?.to_dtype(vb.dtype())?;

        Ok(Self {
            q_proj: linear(HIDDEN_DIM, HIDDEN_DIM, vb.pp("q_proj"))?,
            k_proj: linear(HIDDEN_DIM, HIDDEN_DIM, vb.pp("k_proj"))?,
            v_proj: linear(HIDDEN_DIM, HIDDEN_DIM, vb.pp("v_proj"))?,
            out_proj: linear(HIDDEN_DIM, HIDDEN_DIM, vb.pp("out_proj"))?,
            mask,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        // flash_attn expects inputs in [batch, seq_len, num_heads, head_dim] format
        let q = self
            .q_proj
            .forward(x)?
            .reshape((b_sz, seq_len, NUM_HEADS, HEAD_DIM))?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b_sz, seq_len, NUM_HEADS, HEAD_DIM))?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b_sz, seq_len, NUM_HEADS, HEAD_DIM))?;

        // flash_attn fused kernel applies scaling, causal mask, and softmax inherently
        // The max sequence length dictates the size of SRAM allocated, scale is standard 1/sqrt(d)
        let y = candle_flash_attn::flash_attn(&q, &k, &v, 1.0 / (HEAD_DIM as f32).sqrt(), true)?;

        let y = y.reshape((b_sz, seq_len, HIDDEN_DIM))?;
        self.out_proj.forward(&y)
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
    pub fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            ln_1: layer_norm(HIDDEN_DIM, 1e-5, vb.pp("ln_1"))?,
            attn: CausalSelfAttention::new(vb.pp("attn"))?,
            ln_2: layer_norm(HIDDEN_DIM, 1e-5, vb.pp("ln_2"))?,
            mlp_fc1: linear(HIDDEN_DIM, 4 * HIDDEN_DIM, vb.pp("mlp_fc1"))?,
            mlp_fc2: linear(4 * HIDDEN_DIM, HIDDEN_DIM, vb.pp("mlp_fc2"))?,
        })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let attn_out = self.attn.forward(&self.ln_1.forward(x)?)?;
        let x = x.broadcast_add(&attn_out)?;
        let mlp_out = self.mlp_fc1.forward(&self.ln_2.forward(&x)?)?.gelu()?;
        let mlp_out = self.mlp_fc2.forward(&mlp_out)?;
        x.broadcast_add(&mlp_out)
    }
}

pub struct TinyLLM {
    wte: Embedding,
    wpe: Embedding,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    lm_head: Linear,
    pos: Tensor,
}

impl TinyLLM {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let mut blocks = Vec::new();
        for i in 0..6 {
            blocks.push(Block::new(vb.pp(format!("block_{}", i)))?);
        }
        let pos: Vec<u32> = (0..SEQ_LEN as u32).collect();
        let pos = Tensor::from_vec(pos, (1, SEQ_LEN), vb.device())?;

        Ok(Self {
            wte: embedding(VOCAB_SIZE, HIDDEN_DIM, vb.pp("wte"))?,
            wpe: embedding(SEQ_LEN, HIDDEN_DIM, vb.pp("wpe"))?,
            blocks,
            ln_f: layer_norm(HIDDEN_DIM, 1e-5, vb.pp("ln_f"))?,
            lm_head: linear(HIDDEN_DIM, VOCAB_SIZE, vb.pp("lm_head"))?,
            pos,
        })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, seq_len) = x.dims2()?;
        let pos = self.pos.narrow(1, 0, seq_len)?;

        let mut out = self
            .wte
            .forward(x)?
            .broadcast_add(&self.wpe.forward(&pos)?)?;
        for block in &self.blocks {
            out = block.forward(&out)?;
        }

        out = self.ln_f.forward(&out)?;
        self.lm_head.forward(&out)
    }
}

pub mod fused_adamw;

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarMap;

    fn get_device_and_vb() -> Result<(Device, VarBuilder)> {
        let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::BF16, &device);
        Ok((device, vb))
    }

    #[test]
    fn test_causal_self_attention_dimensions() -> Result<()> {
        let (device, vb) = get_device_and_vb()?;
        if !device.is_cuda() {
            println!("Skipping test: flash attention requires CUDA");
            return Ok(());
        }

        let attn = CausalSelfAttention::new(vb)?;
        let batch_size = 2;
        let seq_len = 10;
        let x = Tensor::randn(0f32, 1f32, (batch_size, seq_len, HIDDEN_DIM), &device)?
            .to_dtype(DType::BF16)?;

        let y = attn.forward(&x)?;

        assert_eq!(y.dims3()?, (batch_size, seq_len, HIDDEN_DIM));
        Ok(())
    }

    #[test]
    fn test_block_dimensions() -> Result<()> {
        let (device, vb) = get_device_and_vb()?;
        if !device.is_cuda() {
            println!("Skipping test: flash attention requires CUDA");
            return Ok(());
        }

        let block = Block::new(vb)?;
        let batch_size = 2;
        let seq_len = 10;
        let x = Tensor::randn(0f32, 1f32, (batch_size, seq_len, HIDDEN_DIM), &device)?
            .to_dtype(DType::BF16)?;

        let y = block.forward(&x)?;

        assert_eq!(y.dims3()?, (batch_size, seq_len, HIDDEN_DIM));
        Ok(())
    }

    #[test]
    fn test_tiny_llm_dimensions() -> Result<()> {
        let (device, vb) = get_device_and_vb()?;
        if !device.is_cuda() {
            println!("Skipping test: flash attention requires CUDA");
            return Ok(());
        }

        let model = TinyLLM::new(vb)?;
        let batch_size = 2;
        let seq_len = 10;

        let x = Tensor::zeros((batch_size, seq_len), DType::U32, &device)?;

        let y = model.forward(&x)?;

        assert_eq!(y.dims3()?, (batch_size, seq_len, VOCAB_SIZE));
        Ok(())
    }
}

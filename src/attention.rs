use crate::Config;
use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use std::sync::Mutex;

pub struct CausalSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    config: Config,
    kv_cache: Mutex<Option<(Tensor, Tensor)>>,
    mask_cache: Mutex<Option<Tensor>>,
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
            mask_cache: Mutex::new(None),
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
        let k = self.k_proj.forward(x)?.reshape((
            b_sz,
            seq_len,
            self.config.num_kv_heads,
            self.config.head_dim(),
        ))?;

        let rope_q = crate::kernel::fused_rope::FusedRope {
            start_pos,
            theta: 10000.0,
            is_bwd: false,
        };
        let rope_k = crate::kernel::fused_rope::FusedRope {
            start_pos,
            theta: 10000.0,
            is_bwd: false,
        };

        let q = q.contiguous()?.apply_op1(rope_q)?;
        let mut k = k.contiguous()?.apply_op1(rope_k)?;

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
            ))?;

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
            ))?;

        // By-pass causal caching but apply native broadcasts
        let q_trans = q.transpose(1, 2)?.contiguous()?; // [b_sz, num_heads, seq_len, head_dim]
        let k_expand = k_expand.transpose(1, 2)?.contiguous()?; // [b_sz, num_heads, seq_len, head_dim]
        let v_expand = v_expand.transpose(1, 2)?.contiguous()?; // [b_sz, num_heads, seq_len, head_dim]

        let att = q_trans.matmul(&k_expand.t()?)?;
        let scale = 1.0 / (self.config.head_dim() as f64).sqrt();
        let att = att.affine(scale, 0.0)?;

        // Causal Masking
        let mut mask_cache_guard = self.mask_cache.lock().unwrap();
        let mask_tensor = if let Some(m) = &*mask_cache_guard {
            if m.dim(0)? >= seq_len {
                m.narrow(0, 0, seq_len)?.narrow(1, 0, seq_len)?
            } else {
                let mask: Vec<_> = (0..seq_len)
                    .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0f32 }))
                    .collect();
                let m = Tensor::from_slice(&mask, (seq_len, seq_len), x.device())?.to_dtype(att.dtype())?;
                *mask_cache_guard = Some(m.clone());
                m
            }
        } else {
            let max_seq_len = self.config.seq_len;
            let build_seq_len = seq_len.max(max_seq_len);
            let mask: Vec<_> = (0..build_seq_len)
                .flat_map(|i| (0..build_seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0f32 }))
                .collect();
            let m = Tensor::from_slice(&mask, (build_seq_len, build_seq_len), x.device())?.to_dtype(att.dtype())?;
            *mask_cache_guard = Some(m.clone());
            m.narrow(0, 0, seq_len)?.narrow(1, 0, seq_len)?
        };
        let mask = mask_tensor.broadcast_as(att.shape())?;

        let att = att.broadcast_add(&mask)?;

        // Custom Softmax implementation to preserve gradients natively!
        let att_exp = att.exp()?;
        let att_sum = att_exp.sum_keepdim(candle_core::D::Minus1)?;
        let att = att_exp.broadcast_div(&att_sum)?;

        let y = att.matmul(&v_expand)?; // [b_sz, num_heads, seq_len, head_dim]

        let y = y.transpose(1, 2)?.contiguous()?; // [b_sz, seq_len, num_heads, head_dim]
        let y = y.reshape((b_sz, seq_len, self.config.hidden_dim))?;

        self.out_proj.forward(&y)
    }

    pub fn clear_kv_cache(&self) {
        *self.kv_cache.lock().unwrap() = None;
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
}

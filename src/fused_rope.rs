use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::{CustomOp1, Error, Layout, Result, Shape, Tensor};

const PTX_CONTENT: &str = include_str!(concat!(env!("OUT_DIR"), "/rope.ptx"));

pub struct FusedRope {
    pub start_pos: usize,
    pub theta: f32,
    pub is_bwd: bool,
}

impl CustomOp1 for FusedRope {
    fn name(&self) -> &'static str {
        "fused_rope"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> Result<(candle_core::CpuStorage, Shape)> {
        Err(Error::Msg("CPU not supported for FusedRope".into()))
    }

    fn cuda_fwd(
        &self,
        s_q: &candle_core::CudaStorage,
        l_q: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        let q_dims = l_q.shape().dims();
        if q_dims.len() != 4 {
            return Err(Error::Msg(
                "Input must be 4D [batch, seq_len, num_heads, head_dim]".into(),
            ));
        }

        if !l_q.is_contiguous() {
            return Err(Error::Msg("Input to RoPE must be contiguous. Call .contiguous() on the tensor prior to applying CustomOp1 FusedRope.".into()));
        }

        let batch_size = q_dims[0] as u32;
        let seq_len = q_dims[1] as u32;
        let num_heads = q_dims[2] as u32;
        let head_dim = q_dims[3] as u32;
        let total_elements = batch_size * seq_len * num_heads * head_dim;

        let dev = s_q.device();
        let func = dev.get_or_load_custom_func("rope_bf16", "rope", PTX_CONTENT)?;

        let candle_core::cuda_backend::CudaStorageSlice::BF16(in_q) = &s_q.slice else {
            return Err(Error::Msg("Expected BF16 storage for RoPE".into()));
        };

        // Allocate output buffer dynamically matching exact total bounds natively
        let s_out = unsafe { dev.alloc::<half::bf16>(total_elements as usize) }?;

        let num_pairs = total_elements / 2;

        let block_size = 256;
        let grid_size = num_pairs.div_ceil(block_size);
        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let sign = if self.is_bwd { -1.0f32 } else { 1.0f32 };

        let start_pos_i32 = self.start_pos as i32;
        let batch_size_i32 = batch_size as i32;
        let seq_len_i32 = seq_len as i32;
        let num_heads_i32 = num_heads as i32;
        let head_dim_i32 = head_dim as i32;

        let mut builder = func.builder();
        builder.arg(in_q);
        builder.arg(&s_out);
        builder.arg(&start_pos_i32);
        builder.arg(&self.theta);
        builder.arg(&sign);
        builder.arg(&batch_size_i32);
        builder.arg(&seq_len_i32);
        builder.arg(&num_heads_i32);
        builder.arg(&head_dim_i32);

        unsafe { builder.launch(cfg) }.map_err(|e| Error::Msg(e.to_string()))?;

        Ok((
            candle_core::CudaStorage {
                slice: candle_core::cuda_backend::CudaStorageSlice::BF16(s_out),
                device: dev.clone(),
            },
            l_q.shape().clone(),
        ))
    }

    fn bwd(&self, _arg: &Tensor, _res: &Tensor, grad_res: &Tensor) -> Result<Option<Tensor>> {
        let bwd_op = FusedRope {
            start_pos: self.start_pos,
            theta: self.theta,
            is_bwd: !self.is_bwd,
        };
        Ok(Some(grad_res.apply_op1(bwd_op)?))
    }
}

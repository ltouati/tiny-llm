use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::{Error, Layout, Result, Shape, Tensor};

const PTX_CONTENT: &str = include_str!(concat!(env!("OUT_DIR"), "/adamw.ptx"));

pub struct FusedAdd;

impl candle_core::CustomOp2 for FusedAdd {
    fn name(&self) -> &'static str {
        "fused_add"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> Result<(candle_core::CpuStorage, Shape)> {
        Err(Error::Msg("CPU not supported for FusedAdd".into()))
    }

    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        let dev = s1.device();
        let func = dev.get_or_load_custom_func("fast_add_bf16", "adamw", PTX_CONTENT)?;

        let numel = l1.shape().elem_count() as u32;
        let s_out = unsafe { dev.alloc::<half::bf16>(numel as usize) }?;

        let candle_core::cuda_backend::CudaStorageSlice::BF16(in1) = &s1.slice else {
            return Err(Error::Msg("Expected BF16".into()));
        };
        let candle_core::cuda_backend::CudaStorageSlice::BF16(in2) = &s2.slice else {
            return Err(Error::Msg("Expected BF16".into()));
        };

        let block_size = 1024u32;
        let grid_size = numel.div_ceil(block_size);
        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let start_offset1 = l1.start_offset() as u32;
        let start_offset2 = l2.start_offset() as u32;

        let mut builder = func.builder();
        builder.arg(&s_out);
        builder.arg(in1);
        builder.arg(in2);
        builder.arg(&numel);
        builder.arg(&start_offset1);
        builder.arg(&start_offset2);

        unsafe { builder.launch(cfg) }.unwrap();

        let out_storage = candle_core::CudaStorage {
            slice: candle_core::cuda_backend::CudaStorageSlice::BF16(s_out),
            device: dev.clone(),
        };
        Ok((out_storage, l1.shape().clone()))
    }

    fn bwd(
        &self,
        _arg1: &Tensor,
        _arg2: &Tensor,
        _res: &Tensor,
        grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        Ok((Some(grad_res.clone()), Some(grad_res.clone())))
    }
}

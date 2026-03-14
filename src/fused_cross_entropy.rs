use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::{Error, Layout, Result, Shape, Tensor};

// Includes the PTX code compiled by `build.rs`
const PTX_CONTENT: &str = include_str!(concat!(env!("OUT_DIR"), "/cross_entropy.ptx"));

pub struct FusedCrossEntropy {
    pub targets: Tensor,
}

impl candle_core::CustomOp1 for FusedCrossEntropy {
    fn name(&self) -> &'static str {
        "fused_cross_entropy"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> Result<(candle_core::CpuStorage, Shape)> {
        Err(Error::Msg("CPU not supported for FusedCrossEntropy".into()))
    }

    fn cuda_fwd(
        &self,
        s_logits: &candle_core::CudaStorage,
        l_logits: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        let logits_dims = l_logits.shape().dims();

        if logits_dims.len() != 2 && logits_dims.len() != 3 {
            return Err(Error::Msg(
                "Logits must be 2D [B*S, V] or 3D [B, S, V]".into(),
            ));
        }

        let vocab_size = *logits_dims.last().unwrap();
        let num_tokens = l_logits.shape().elem_count() / vocab_size; // Total N

        let targets_flat = self.targets.flatten_all()?;
        let targets_contiguous = targets_flat.contiguous()?;

        if targets_contiguous.shape().elem_count() != num_tokens {
            return Err(Error::Msg("Targets length must match num_tokens".into()));
        }

        let dev = s_logits.device();
        let func =
            dev.get_or_load_custom_func("cross_entropy_fwd", "cross_entropy", PTX_CONTENT)?;

        // Allocate a scalar tensor locally initializing at exactly `0.0`
        let s_losses = dev.alloc_zeros::<f32>(num_tokens)?;

        let candle_core::cuda_backend::CudaStorageSlice::BF16(in_logits) = &s_logits.slice else {
            return Err(Error::Msg("Expected BF16 storage for logits".into()));
        };

        let (targets_storage, _) = targets_contiguous.storage_and_layout();
        let candle_core::Storage::Cuda(s_targets) = &*targets_storage else {
            return Err(Error::Msg("Expected CudaStorage for targets".into()));
        };
        let candle_core::cuda_backend::CudaStorageSlice::U32(in_targets) = &s_targets.slice else {
            return Err(Error::Msg("Expected U32 storage for targets".into()));
        };

        // Launch config: 1 Block per token, 1024 threads per block
        let block_size = 1024;
        let grid_size = num_tokens as u32;

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = func.builder();
        builder.arg(in_logits);
        builder.arg(in_targets);
        builder.arg(&s_losses);
        let vocab_size_u32 = vocab_size as u32;
        builder.arg(&vocab_size_u32);
        unsafe { builder.launch(cfg) }.unwrap();

        // Convert CUDA slice to a CudaStorage returning the resulting scalar tensor array
        let out_storage = candle_core::CudaStorage {
            slice: candle_core::cuda_backend::CudaStorageSlice::F32(s_losses),
            device: dev.clone(),
        };
        let out_shape = Shape::from((num_tokens,));
        Ok((out_storage, out_shape))
    }

    fn bwd(&self, arg1: &Tensor, _res: &Tensor, grad_res: &Tensor) -> Result<Option<Tensor>> {
        let logits = arg1.flatten_all()?;
        let targets = self.targets.flatten_all()?;

        // Ensure inputs are contiguous so layouts match memory bounds perfectly
        let logits = logits.contiguous()?;
        let targets = targets.contiguous()?;

        let (logits_storage, _) = logits.storage_and_layout();
        let (targets_storage, _) = targets.storage_and_layout();

        let candle_core::Storage::Cuda(s_logits) = &*logits_storage else {
            return Err(Error::Msg(
                "Expected CudaStorage for logits in backward".into(),
            ));
        };
        let candle_core::Storage::Cuda(s_targets) = &*targets_storage else {
            return Err(Error::Msg(
                "Expected CudaStorage for targets in backward".into(),
            ));
        };

        let candle_core::cuda_backend::CudaStorageSlice::BF16(in_logits) = &s_logits.slice else {
            return Err(Error::Msg(
                "Expected BF16 storage for logits backward".into(),
            ));
        };
        let candle_core::cuda_backend::CudaStorageSlice::U32(in_targets) = &s_targets.slice else {
            return Err(Error::Msg(
                "Expected U32 storage for targets backward".into(),
            ));
        };

        let logits_dims = arg1.shape().dims();
        let vocab_size = *logits_dims.last().unwrap();
        let num_tokens = arg1.shape().elem_count() / vocab_size; // Total N

        let dev = s_logits.device();
        let func =
            dev.get_or_load_custom_func("cross_entropy_bwd", "cross_entropy", PTX_CONTENT)?;

        // Output tensor for logits gradient: [N, V] in BF16 natively (no F32)
        // Using `half::bf16` directly to let device allocate
        let s_grad_logits = dev.alloc_zeros::<half::bf16>(num_tokens * vocab_size)?;

        let block_size = 1024;
        let grid_size = num_tokens as u32;

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = func.builder();
        builder.arg(in_logits);
        builder.arg(in_targets);
        builder.arg(&s_grad_logits);
        let vocab_size_u32 = vocab_size as u32;
        builder.arg(&vocab_size_u32);

        let grad_res_contig = grad_res.contiguous()?;
        let (grad_res_storage, _) = grad_res_contig.storage_and_layout();
        let candle_core::Storage::Cuda(s_grad_res) = &*grad_res_storage else {
            return Err(Error::Msg("Expected CudaStorage for grad_res".into()));
        };
        let candle_core::cuda_backend::CudaStorageSlice::F32(in_grad_res) = &s_grad_res.slice
        else {
            return Err(Error::Msg("Expected F32 storage for grad_res".into()));
        };
        builder.arg(in_grad_res);

        unsafe { builder.launch(cfg) }.unwrap();

        let out_storage = candle_core::CudaStorage {
            slice: candle_core::cuda_backend::CudaStorageSlice::BF16(s_grad_logits),
            device: dev.clone(),
        };
        let out_tensor = candle_core::Tensor::from_storage(
            candle_core::Storage::Cuda(out_storage),
            arg1.shape().clone(),
            candle_core::op::BackpropOp::none(),
            false,
        );

        Ok(Some(out_tensor))
    }
}

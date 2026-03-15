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

        let vocab_size = *arg1.shape().dims().last().unwrap() as u32;
        let num_tokens = (arg1.shape().elem_count() / vocab_size as usize) as u32;

        let dev = s_logits.device();
        let func =
            dev.get_or_load_custom_func("cross_entropy_bwd", "cross_entropy", PTX_CONTENT)?;

        // Output tensor for logits gradient: [N, V] in BF16 natively (no F32)
        let s_grad_logits = unsafe { dev.alloc::<half::bf16>((num_tokens * vocab_size) as usize) }?;

        let block_size = 1024;
        let grid_size = num_tokens;

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let grad_res_contig = grad_res.contiguous()?;
        let (grad_res_storage, _) = grad_res_contig.storage_and_layout();
        let candle_core::Storage::Cuda(s_grad_res) = &*grad_res_storage else {
            return Err(Error::Msg("Expected CudaStorage for grad_res".into()));
        };
        let candle_core::cuda_backend::CudaStorageSlice::F32(in_grad_res) = &s_grad_res.slice
        else {
            return Err(Error::Msg("Expected F32 storage for grad_res".into()));
        };

        let mut builder = func.builder();
        builder.arg(in_logits);
        builder.arg(in_targets);
        builder.arg(&s_grad_logits);
        let vocab_size_i32 = vocab_size as i32;
        builder.arg(&vocab_size_i32);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_cross_entropy_shape_validation() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let targets = Tensor::zeros((4,), candle_core::DType::U32, &device)?;

        let custom_op = FusedCrossEntropy { targets };

        // FusedCrossEntropy enforces 2D or 3D logits
        let invalid_1d = Tensor::ones((4,), candle_core::DType::BF16, &device)?;
        let invalid_4d = Tensor::ones((1, 1, 4, 16), candle_core::DType::BF16, &device)?;

        // Assert error is correctly propagated (CPU fwd rejects natively, but we test the custom op explicit behavior)
        let res_1d = invalid_1d.apply_op1_no_bwd(&custom_op);
        let res_4d = invalid_4d.apply_op1_no_bwd(&custom_op);

        // They should explicitly error containing the string "CPU not supported" because CPU is outright rejected first
        assert!(res_1d.is_err());
        assert!(res_4d.is_err());

        // Test CUDA if available
        if let Ok(cuda_device) = candle_core::Device::new_cuda(0) {
            let invalid_cuda_1d = Tensor::ones((4,), candle_core::DType::BF16, &cuda_device)?;
            let invalid_cuda_4d =
                Tensor::ones((1, 1, 4, 16), candle_core::DType::BF16, &cuda_device)?;

            let custom_op_cuda = FusedCrossEntropy {
                targets: Tensor::zeros((4,), candle_core::DType::U32, &cuda_device)?,
            };

            let res_cuda_1d = invalid_cuda_1d.apply_op1_no_bwd(&custom_op_cuda);
            assert!(res_cuda_1d
                .unwrap_err()
                .to_string()
                .contains("Logits must be 2D"));

            let res_cuda_4d = invalid_cuda_4d.apply_op1_no_bwd(&custom_op_cuda);
            assert!(res_cuda_4d
                .unwrap_err()
                .to_string()
                .contains("Logits must be 2D"));
        }

        Ok(())
    }

    #[test]
    fn test_fused_cross_entropy_forward_correctness() -> Result<()> {
        let device = candle_core::Device::new_cuda(0);
        if device.is_err() {
            println!("Skipping CUDA-only test.");
            return Ok(());
        }
        let device = device.unwrap();

        // 2 tokens, 4 vocab size
        let logits = Tensor::from_slice(
            &[
                // Token 1 logits
                2.0f32, 1.0, 0.1, -1.0, // Token 2 logits
                -0.5, 0.0, 1.5, 2.5,
            ],
            (2, 4),
            &device,
        )?
        .to_dtype(candle_core::DType::BF16)?;

        // Targets: Token 1 -> class 0, Token 2 -> class 3
        let targets = Tensor::from_slice(&[0u32, 3u32], (2,), &device)?;

        // Hand-calculated baseline (or Native candle_nn cross_entropy)
        let native_loss =
            candle_nn::loss::cross_entropy(&logits.to_dtype(candle_core::DType::F32)?, &targets)?;

        let custom_op = FusedCrossEntropy { targets };
        // Fused custom op returns an array of un-meaned losses
        let fused_loss_vec = logits.apply_op1(custom_op)?;
        let fused_loss = fused_loss_vec
            .mean_all()?
            .to_dtype(candle_core::DType::F32)?;

        let native_val = native_loss.to_scalar::<f32>()?;
        let fused_val = fused_loss.to_scalar::<f32>()?;

        // Expect near exact match within BF16 mantissa margin
        assert!(
            (native_val - fused_val).abs() < 5e-3,
            "Loss mismatch: Native = {}, Fused = {}",
            native_val,
            fused_val
        );

        Ok(())
    }

    #[test]
    fn test_fused_cross_entropy_backward_gradients() -> Result<()> {
        let device = candle_core::Device::new_cuda(0);
        if device.is_err() {
            println!("Skipping CUDA-only test.");
            return Ok(());
        }
        let device = device.unwrap();

        // Target vectors

        let initial_logits = Tensor::from_slice(
            &[2.0f32, 1.0, 0.1, -1.0, -0.5, 0.0, 1.5, 2.5],
            (2, 4),
            &device,
        )?;

        // Native model vars
        let native_logits = candle_core::Var::from_tensor(&initial_logits.clone())?;
        let targets = Tensor::from_slice(&[0u32, 3u32], (2,), &device)?;

        let native_loss = candle_nn::loss::cross_entropy(native_logits.as_tensor(), &targets)?;
        let native_grads_store = native_loss.backward()?;
        let native_grad = native_grads_store.get(&native_logits).unwrap().clone();

        // Custom Fused Ops Vars
        let fused_logits =
            candle_core::Var::from_tensor(&initial_logits.to_dtype(candle_core::DType::BF16)?)?;
        let custom_op = FusedCrossEntropy {
            targets: targets.clone(),
        };
        let fused_loss_vec = fused_logits.as_tensor().apply_op1(custom_op)?;
        let fused_loss = fused_loss_vec.mean_all()?;
        let fused_grads_store = fused_loss.backward()?;
        let fused_grad = fused_grads_store.get(&fused_logits).unwrap().clone();

        let native_g_vec = native_grad.to_vec2::<f32>()?;
        // Note: Our Fused Cross Entropy returns gradients SCALED perfectly internally to apply natively if mean_all is used
        let fused_g_vec = fused_grad
            .to_dtype(candle_core::DType::F32)?
            .to_vec2::<f32>()?;

        for i in 0..2 {
            for j in 0..4 {
                assert!(
                    (native_g_vec[i][j] - fused_g_vec[i][j]).abs() < 5e-3,
                    "Gradient mismatch at [{}, {}]: Native = {}, Fused = {}",
                    i,
                    j,
                    native_g_vec[i][j],
                    fused_g_vec[i][j]
                );
            }
        }

        Ok(())
    }
}

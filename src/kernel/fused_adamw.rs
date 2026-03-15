use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::{Device, Result, Var};

#[derive(Clone, Debug)]
pub struct ParamsAdamW {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

impl Default for ParamsAdamW {
    fn default() -> Self {
        Self {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
}

#[derive(Debug)]
struct VarAdamW {
    var: Var,
    first_moment: Var,
    second_moment: Var,
    accum_grad: Option<candle_core::Tensor>,
}

pub struct FusedAdamW {
    vars: Vec<VarAdamW>,
    step_t: usize,
    params: ParamsAdamW,
}

impl FusedAdamW {
    pub fn new(vars: Vec<Var>, params: ParamsAdamW) -> Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .map(|var| {
                let dtype = var.dtype();
                let shape = var.shape();
                let device = var.device();
                let first_moment = Var::zeros(shape, dtype, device)?;
                let second_moment = Var::zeros(shape, dtype, device)?;
                Ok(VarAdamW {
                    var,
                    first_moment,
                    second_moment,
                    accum_grad: None,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            vars,
            params,
            step_t: 0,
        })
    }

    pub fn new_lr(vars: Vec<Var>, learning_rate: f64) -> Result<Self> {
        let params = ParamsAdamW {
            lr: learning_rate,
            ..ParamsAdamW::default()
        };
        Self::new(vars, params)
    }

    pub fn set_lr(&mut self, lr: f64) {
        self.params.lr = lr;
    }

    pub fn accumulate(
        &mut self,
        grads: &candle_core::backprop::GradStore,
        accumulation_steps: usize,
    ) -> Result<()> {
        let scale = 1.0f32 / (accumulation_steps as f32);
        for var in self.vars.iter_mut() {
            if let Some(g) = grads.get(&var.var) {
                if let Some(ag) = &var.accum_grad {
                    let dev = g.device();
                    let Device::Cuda(cuda_dev) = dev else {
                        panic!("FusedAdamW only supports CUDA devices");
                    };

                    let ptx_content = include_str!(concat!(env!("OUT_DIR"), "/adamw.ptx"));
                    let func_add = cuda_dev.get_or_load_custom_func(
                        "inplace_add_bf16",
                        "adamw",
                        ptx_content,
                    )?;

                    let t_ag = ag.flatten_all()?;
                    let t_g = g.flatten_all()?;

                    let (ag_storage, _) = t_ag.storage_and_layout();
                    let (g_storage, _) = t_g.storage_and_layout();

                    let (ag_cuda, g_cuda) = match (&*ag_storage, &*g_storage) {
                        (candle_core::Storage::Cuda(ag), candle_core::Storage::Cuda(g)) => (ag, g),
                        _ => panic!("Expected CudaStorage"),
                    };

                    let in_ag = match &ag_cuda.slice {
                        candle_core::cuda_backend::CudaStorageSlice::BF16(slice) => slice,
                        _ => panic!("Expected BF16 storage for accum_grad"),
                    };

                    let in_g = match &g_cuda.slice {
                        candle_core::cuda_backend::CudaStorageSlice::BF16(slice) => slice,
                        _ => panic!("Expected BF16 storage for grad"),
                    };

                    let numel = t_ag.elem_count() as u32;
                    let block_size = 1024u32;
                    let grid_size = numel.div_ceil(block_size);

                    let cfg = LaunchConfig {
                        grid_dim: (grid_size, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 0,
                    };

                    let mut builder = func_add.builder();
                    builder.arg(in_ag);
                    builder.arg(in_g);
                    builder.arg(&scale);
                    builder.arg(&numel);

                    unsafe { builder.launch(cfg) }.unwrap();
                } else {
                    var.accum_grad = Some(g.affine(scale as f64, 0.0)?.detach());
                }
            }
        }
        Ok(())
    }

    pub fn step_with_accumulated(&mut self) -> Result<()> {
        self.step_t += 1;
        let lr = self.params.lr as f32;
        let lambda = self.params.weight_decay as f32;
        let lr_lambda = lr * lambda;
        let beta1 = self.params.beta1 as f32;
        let beta2 = self.params.beta2 as f32;
        let eps = self.params.eps as f32;

        let scale_m = 1f32 / (1f32 - beta1.powi(self.step_t as i32));
        let scale_v = 1f32 / (1f32 - beta2.powi(self.step_t as i32));

        for var in self.vars.iter_mut() {
            let theta = &var.var;
            let m = &var.first_moment;
            let v = &var.second_moment;

            if let Some(g) = var.accum_grad.take() {
                let Device::Cuda(dev) = theta.device() else {
                    panic!("FusedAdamW only supports CUDA devices");
                };

                let ptx_content = include_str!(concat!(env!("OUT_DIR"), "/adamw.ptx"));

                let func_vec =
                    dev.get_or_load_custom_func("adamw_bf16_step", "adamw", ptx_content)?;
                let func_fallback =
                    dev.get_or_load_custom_func("adamw_bf16_step_fallback", "adamw", ptx_content)?;

                let numel = theta.elem_count();
                let numel_vec = (numel / 8) as u32;
                let remainder = (numel % 8) as u32;

                // 1. Convert everything to flattened contiguous CUDA Tensors to extract slices
                // We use `.flatten_all()` or simply trust they are contiguous
                let t_theta = theta.as_tensor().flatten_all()?;
                let t_m = m.as_tensor().flatten_all()?;
                let t_v = v.as_tensor().flatten_all()?;
                let t_g = g.flatten_all()?;

                // Determine layouts (should all be contiguous with 0 offset since flattened)
                let (theta_storage, _) = t_theta.storage_and_layout();
                let (m_storage, _) = t_m.storage_and_layout();
                let (v_storage, _) = t_v.storage_and_layout();
                let (g_storage, _) = t_g.storage_and_layout();

                let (theta_cuda, m_cuda, v_cuda, g_cuda) =
                    match (&*theta_storage, &*m_storage, &*v_storage, &*g_storage) {
                        (
                            candle_core::Storage::Cuda(t),
                            candle_core::Storage::Cuda(m),
                            candle_core::Storage::Cuda(v),
                            candle_core::Storage::Cuda(g),
                        ) => (t, m, v, g),
                        _ => panic!("Expected Cuda Storage"),
                    };

                // Directly bypass Rust immutable safety constraints inside the `Var` locking layer.
                // We execute an asynchronous in-place operation on the underlying device memory.
                unsafe {
                    // Read only handles over old buffers via Enum matching
                    if let (
                        candle_core::cuda_backend::CudaStorageSlice::BF16(in_theta),
                        candle_core::cuda_backend::CudaStorageSlice::BF16(in_m),
                        candle_core::cuda_backend::CudaStorageSlice::BF16(in_v),
                        candle_core::cuda_backend::CudaStorageSlice::BF16(in_g),
                    ) = (
                        &theta_cuda.slice,
                        &m_cuda.slice,
                        &v_cuda.slice,
                        &g_cuda.slice,
                    ) {
                        // 1. Launch Vectorized Kernel
                        if numel_vec > 0 {
                            let block_size = 512;
                            let grid_size = numel_vec.div_ceil(block_size);
                            let cfg = LaunchConfig {
                                grid_dim: (grid_size, 1, 1),
                                block_dim: (block_size, 1, 1),
                                shared_mem_bytes: 0,
                            };
                            let mut builder = func_vec.builder();
                            builder.arg(in_theta);
                            builder.arg(in_m);
                            builder.arg(in_v);
                            builder.arg(in_g);
                            builder.arg(&lr);
                            builder.arg(&beta1);
                            builder.arg(&beta2);
                            builder.arg(&eps);
                            builder.arg(&lr_lambda);
                            builder.arg(&scale_m);
                            builder.arg(&scale_v);
                            builder.arg(&numel_vec);
                            builder.launch(cfg).unwrap();
                        }

                        // 2. Launch Fallback Kernel for remainder
                        if remainder > 0 {
                            let block_size = 32;
                            let grid_size = 1;
                            let cfg = LaunchConfig {
                                grid_dim: (grid_size, 1, 1),
                                block_dim: (block_size, 1, 1),
                                shared_mem_bytes: 0,
                            };
                            let mut builder = func_fallback.builder();
                            builder.arg(in_theta);
                            builder.arg(in_m);
                            builder.arg(in_v);
                            builder.arg(in_g);
                            builder.arg(&lr);
                            builder.arg(&beta1);
                            builder.arg(&beta2);
                            builder.arg(&eps);
                            builder.arg(&lr_lambda);
                            builder.arg(&scale_m);
                            builder.arg(&scale_v);
                            let start_idx = numel_vec * 8;
                            builder.arg(&start_idx);
                            let numel_u32 = numel as u32;
                            builder.arg(&numel_u32);
                            builder.launch(cfg).unwrap();
                        }
                    } else {
                        panic!("Expected BF16 Storage");
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn test_fused_adamw_initialization() -> Result<()> {
        let device = Device::Cpu; // Initialize with CPU to test purely structural allocation
        let var1 = Var::zeros((10, 10), DType::F32, &device)?;
        let var2 = Var::ones((5, 5), DType::F32, &device)?;

        // FusedAdamW requires explicitly extracting variables as Vec<Var>
        let vars = vec![var1.clone(), var2.clone()];

        let params = ParamsAdamW {
            lr: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.05,
        };

        let adamw = FusedAdamW::new(vars, params.clone())?;

        // Assert parameters were bound correctly
        assert_eq!(adamw.params.lr, 0.01);
        assert_eq!(adamw.params.weight_decay, 0.05);

        // Assert internal variables match expected sizes and dimensions
        assert_eq!(adamw.vars.len(), 2);

        // First variable checks
        assert_eq!(adamw.vars[0].var.shape().dims(), &[10, 10]);
        assert_eq!(adamw.vars[0].first_moment.shape().dims(), &[10, 10]);
        assert_eq!(adamw.vars[0].second_moment.shape().dims(), &[10, 10]);
        assert!(adamw.vars[0].accum_grad.is_none());

        // Second variable checks
        assert_eq!(adamw.vars[1].var.shape().dims(), &[5, 5]);
        assert_eq!(adamw.vars[1].first_moment.shape().dims(), &[5, 5]);
        assert_eq!(adamw.vars[1].second_moment.shape().dims(), &[5, 5]);
        assert!(adamw.vars[1].accum_grad.is_none());

        Ok(())
    }

    #[test]
    fn test_fused_adamw_accumulate() -> Result<()> {
        let device = Device::new_cuda(0);
        if device.is_err() {
            println!("Skipping CUDA-only test.");
            return Ok(());
        }
        let device = device.unwrap();

        // Target test Var
        let initial_val = Tensor::from_slice(&[0.0f32; 4], (4,), &device)?.to_dtype(DType::BF16)?;
        let var1 = Var::from_tensor(&initial_val)?;

        // Fake Gradients

        let grad1 =
            Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (4,), &device)?.to_dtype(DType::BF16)?;

        // Mock a grad store manually
        let mut adamw = FusedAdamW::new(vec![var1.clone()], ParamsAdamW::default())?;

        // Build an explicit node structure using loss scaling backprop mathematically to get a valid GradStore mapping natively
        let loss = var1.as_tensor().sum_all()?;
        let mut grad_store = loss.backward()?;

        // Overwrite the backward extraction natively within the Grad Store dictionary mapping exactly
        grad_store.insert(&var1, grad1.clone());

        // Accumulate 2 Steps: First step sets the value to `grad * 1/2`.
        adamw.accumulate(&grad_store, 2)?;

        let out1 = adamw.vars[0]
            .accum_grad
            .as_ref()
            .unwrap()
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;

        // Expected: grad1 * 0.5
        assert!((out1[0] - 0.5).abs() < 1e-3);
        assert!((out1[1] - 1.0).abs() < 1e-3);
        assert!((out1[2] - 1.5).abs() < 1e-3);
        assert!((out1[3] - 2.0).abs() < 1e-3);

        // Second step adds another `grad * 1/2`.
        let grad2 =
            Tensor::from_slice(&[2.0f32, 4.0, -1.0, 0.0], (4,), &device)?.to_dtype(DType::BF16)?;
        grad_store.insert(&var1, grad2);

        adamw.accumulate(&grad_store, 2)?;
        let out2 = adamw.vars[0]
            .accum_grad
            .as_ref()
            .unwrap()
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;

        // Expected cumulative: [0.5, 1.0, 1.5, 2.0] + ([2.0, 4.0, -1.0, 0.0] * 0.5)
        // = [0.5 + 1.0, 1.0 + 2.0, 1.5 - 0.5, 2.0 + 0.0]
        // = [1.5, 3.0, 1.0, 2.0]
        assert!((out2[0] - 1.5).abs() < 1e-2);
        assert!((out2[1] - 3.0).abs() < 1e-2);
        assert!((out2[2] - 1.0).abs() < 1e-2);
        assert!((out2[3] - 2.0).abs() < 1e-2);

        Ok(())
    }

    #[test]
    fn test_fused_adamw_step() -> Result<()> {
        let device = Device::new_cuda(0);
        if device.is_err() {
            println!("Skipping CUDA-only test.");
            return Ok(());
        }
        let device = device.unwrap();

        let initial_val =
            Tensor::from_slice(&[1.0f32, 2.0, -1.0, 0.5], (4,), &device)?.to_dtype(DType::BF16)?;
        let var1 = Var::from_tensor(&initial_val)?;

        let params = ParamsAdamW {
            lr: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        };

        let mut adamw = FusedAdamW::new(vec![var1.clone()], params)?;

        // Manually inject accumulated gradients
        let ag =
            Tensor::from_slice(&[0.1f32, -0.2, 0.3, -0.1], (4,), &device)?.to_dtype(DType::BF16)?;
        adamw.vars[0].accum_grad = Some(ag);

        // Step 1 Execution
        adamw.step_with_accumulated()?;

        // Calculate expected values for Step 1
        // Step 1 specifics:
        // lr = 0.1, beta1 = 0.9, beta2 = 0.999, wd = 0.01, eps = 1e-8
        // t = 1
        // scale_m = 1 / (1 - 0.9) = 10
        // scale_v = 1 / (1 - 0.999) = 1000
        //
        // For element 0:
        // theta = 1.0, g = 0.1
        // theta = theta * (1 - lr * wd) = 1.0 * (1 - 0.001) = 0.999
        // m = beta1 * m + (1 - beta1) * g = 0.9 * 0 + 0.1 * 0.1 = 0.01
        // v = beta2 * v + (1 - beta2) * g^2 = 0.999 * 0 + 0.001 * 0.01 = 0.00001
        // m_hat = m * scale_m = 0.01 * 10 = 0.1
        // v_hat = v * scale_v = 0.00001 * 1000 = 0.01
        // update = m_hat / (sqrt(v_hat) + eps) = 0.1 / (0.1 + 1e-8) ≈ 1.0
        // theta = theta - lr * update = 0.999 - 0.1 * 1.0 = 0.899

        let out = adamw.vars[0]
            .var
            .as_tensor()
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;

        let expected_theta_0 = 1.0 * (1.0 - 0.1 * 0.01)
            - 0.1 * (0.01 * 10.0) / ((0.001 * 0.01 * 1000.0f32).sqrt() + 1e-8);
        let expected_theta_1 = 2.0 * (1.0 - 0.1 * 0.01)
            - 0.1 * (-0.02 * 10.0) / ((0.001 * 0.04 * 1000.0f32).sqrt() + 1e-8);
        let expected_theta_2 =
            -(1.0 - 0.1 * 0.01) - 0.1 * (0.03 * 10.0) / ((0.001 * 0.09 * 1000.0f32).sqrt() + 1e-8);
        let expected_theta_3 = 0.5 * (1.0 - 0.1 * 0.01)
            - 0.1 * (-0.01 * 10.0) / ((0.001 * 0.01 * 1000.0f32).sqrt() + 1e-8);

        assert!(
            (out[0] - expected_theta_0).abs() < 5e-3,
            "Index 0 Failed: Out = {}, Expected ≈ {}",
            out[0],
            expected_theta_0
        );
        assert!(
            (out[1] - expected_theta_1).abs() < 5e-3,
            "Index 1 Failed: Out = {}, Expected ≈ {}",
            out[1],
            expected_theta_1
        );
        assert!(
            (out[2] - expected_theta_2).abs() < 5e-3,
            "Index 2 Failed: Out = {}, Expected ≈ {}",
            out[2],
            expected_theta_2
        );
        assert!(
            (out[3] - expected_theta_3).abs() < 5e-3,
            "Index 3 Failed: Out = {}, Expected ≈ {}",
            out[3],
            expected_theta_3
        );

        // Assert accum_grad is consumed correctly
        assert!(adamw.vars[0].accum_grad.is_none());

        Ok(())
    }
}

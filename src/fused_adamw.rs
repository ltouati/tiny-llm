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

    pub fn step(&mut self, grads: &candle_core::backprop::GradStore) -> Result<()> {
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

            if let Some(g) = grads.get(theta) {
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

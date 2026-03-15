use burn::backend::cuda::CudaDevice;

pub struct DeviceSetup;

impl DeviceSetup {
    pub fn calculate_batch_size(_device: &CudaDevice) -> usize {
        let mut batch_size = 8; // Default fallback for CPU

        if let Ok(nvml) = nvml_wrapper::Nvml::init() {
            if let Ok(gpu) = nvml.device_by_index(0) {
                if let Ok(memory_info) = gpu.memory_info() {
                    let total_mb = memory_info.total as usize / 1_000_000;
                    let free_mb = memory_info.free as usize / 1_000_000;

                    let reserved_mb = 4000; // Leave 4GB for OS, CUDA context, and Burn Autotune workspace overhead
                    if free_mb > reserved_mb {
                        let available_mb_for_batches = free_mb - reserved_mb;
                        // Heuristic: A 12-layer 70M model at 1024 context window requires far less memory per batch item.
                        // We safely bound it to ~4000 MB per batch item to maintain high memory utilization.
                        let memory_per_batch_item_mb = 4000.0;
                        let calculated_batch =
                            (available_mb_for_batches as f64 / memory_per_batch_item_mb) as usize;

                        // For optimal cuBLAS kernel selection and to avoid insane workspace sizes,
                        // we strictly enforce that the batch size is capped aggressively to 1 or 2 for PyTorch-like memory profiles
                        batch_size = calculated_batch.clamp(1, 2); // Hardcap at 2 for 8GB VRAM cards to prevent backward pass OOM

                        println!("NVML dynamically sized Target Batch to {} based on {} MB capacity ({} MB total VRAM | {} MB free VRAM detected) 🚀", batch_size, available_mb_for_batches, total_mb, free_mb);
                    }
                }
            }
        } else {
            println!("NVML initialization failed; using fallback Batch Size of 2 on CUDA.");
            batch_size = 2;
        }

        batch_size
    }
}

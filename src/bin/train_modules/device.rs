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
                        // Heuristic: Using BF16 mixed-precision halves the VRAM demand for weights/activations.
                        // However, CrossEntropy expands logits to [batch_size * seq_len, vocab_size].
                        // To prevent massive contiguous buffer allocations (>10GB) that panic the allocator, we budget safely.
                        let memory_per_batch_item_mb = 3500.0;
                        let calculated_batch =
                            (available_mb_for_batches as f64 / memory_per_batch_item_mb) as usize;

                        // We clamp to 8 to absolutely guarantee we never attempt a single contiguous allocation
                        // that exceeds `cubecl`'s fragmentation limits, while still pushing 2x the original limit.
                        batch_size = calculated_batch.clamp(1, 8);

                        log::info!("NVML dynamically sized Target Batch to {} based on {} MB capacity ({} MB total VRAM | {} MB free VRAM detected) 🚀", batch_size, available_mb_for_batches, total_mb, free_mb);
                    }
                }
            }
        } else {
            log::info!("NVML initialization failed; using fallback Batch Size of 2 on CUDA.");
            batch_size = 2;
        }

        batch_size
    }
}

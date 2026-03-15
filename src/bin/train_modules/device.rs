use candle_core::Device;

pub struct DeviceSetup;

impl DeviceSetup {
    pub fn calculate_batch_size(device: &Device) -> usize {
        let mut batch_size = 8; // Default fallback for CPU

        if device.is_cuda() {
            if let Ok(nvml) = nvml_wrapper::Nvml::init() {
                if let Ok(gpu) = nvml.device_by_index(0) {
                    if let Ok(memory_info) = gpu.memory_info() {
                        let total_mb = memory_info.total as usize / 1_000_000;
                        let free_mb = memory_info.free as usize / 1_000_000;

                        let reserved_mb = 2000; // Leave 2GB for OS and CUDA context overhead
                        if free_mb > reserved_mb {
                            let available_mb_for_batches = free_mb - reserved_mb;
                            // Heuristic: A 12-layer 70M model at 1024 context window requires far less memory per batch item.
                            // We safely bound it to ~2000 MB per batch item to maintain high memory utilization.
                            let memory_per_batch_item_mb = 2000.0;
                            let calculated_batch = (available_mb_for_batches as f64
                                / memory_per_batch_item_mb)
                                as usize;

                            // For optimal cuBLAS kernel selection and to avoid insane workspace sizes,
                            // we strictly enforce that the batch size is a multiple of 8 if possible, but fallback to 2 or 4.
                            batch_size = if calculated_batch >= 8 {
                                8 // Hardcap at 8 for 40GB A100 to prevent backward pass OOM
                            } else {
                                calculated_batch.max(2) // Fallback to 2 for small GPUs
                            };

                            println!("NVML dynamically sized Target Batch to {} based on {} MB capacity ({} MB total VRAM | {} MB free VRAM detected) 🚀", batch_size, available_mb_for_batches, total_mb, free_mb);
                        }
                    }
                }
            }
        } else {
            println!("Running natively on CPU without NVML, using fallback Batch Size of 1");
            batch_size = 1;
        }

        batch_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_fallback_batch_size() {
        let device = std::sync::Arc::new(candle_core::Device::Cpu);
        // Should fallback to 1 automatically in the CPU branch
        let batch_size = DeviceSetup::calculate_batch_size(&device);
        assert_eq!(batch_size, 1);
    }

    #[test]
    fn test_cuda_batch_size_no_panic() {
        // Just instantiate device natively and ensure no panics if NVML acts up
        if let Ok(device) = candle_core::Device::new_cuda(0) {
            let batch_size = DeviceSetup::calculate_batch_size(&device);
            assert!(batch_size >= 2); // Fallback is 2 or up to 8 max on CUDA VRAM detected
        }
    }
}

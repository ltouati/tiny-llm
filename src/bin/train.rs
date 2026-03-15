use anyhow::Result;
use burn::backend::cuda::CudaDevice;
use tiny_llm::config::TinyLLMConfig;

#[path = "train_modules/dataset.rs"]
mod dataset;
#[path = "train_modules/trainer.rs"]
mod trainer;

#[path = "train_modules/device.rs"]
mod device;
#[path = "train_modules/metrics.rs"]
mod metrics;

use burn::backend::{Autodiff, Cuda};
use device::DeviceSetup;
use trainer::Trainer;

// Define the precise backend type structurally
type MyBackend = Cuda;
type MyAutodiffBackend = Autodiff<MyBackend>;

fn main() -> Result<()> {
    let device = CudaDevice::default();
    println!("Training natively using Burn CUDA backend on: {:?}", device);

    // Initializing Burn's built-in derived Config natively
    let config = TinyLLMConfig::new();
    println!("Burn Configuration Initialized!");

    let mut dataset_percentage = 100.0;
    let mut output_dir_base = "checkpoints_burn".to_string();

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--dataset-percentage" {
            if let Some(val) = args.next() {
                dataset_percentage = val
                    .parse::<f64>()
                    .expect("Failed to parse dataset percentage");
            }
        } else if arg == "--output-dir" {
            if let Some(val) = args.next() {
                output_dir_base = val;
            }
        }
    }

    let timestamp = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S");
    let output_dir = format!("{}/{}", output_dir_base, timestamp);
    println!("Burn Configured output directory: {}", output_dir);

    let auto_batch_size = DeviceSetup::calculate_batch_size(&device);
    let auto_gradient_accumulation = (64 / auto_batch_size).max(1);
    println!(
        "NVML configured Gradient Accumulation tightly to {} to reach 64 macro-batch equivalence.",
        auto_gradient_accumulation
    );

    Trainer::train::<MyAutodiffBackend>(
        config,
        device,
        "fineweb_edu.bin",
        dataset_percentage,
        auto_batch_size,
        auto_gradient_accumulation,
        10, // Max epochs
        &output_dir,
    );

    Ok(())
}

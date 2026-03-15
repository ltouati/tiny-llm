use anyhow::Result;
use burn::backend::cuda::CudaDevice;
use burn::config::Config;
use tiny_llm::config::{TinyLLMConfig, TinyLLMTrainingConfig};

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

// Define the precise backend type structurally (using BF16 for math)
type MyBackend = Cuda<half::bf16, i32>;
type MyAutodiffBackend = Autodiff<MyBackend>;

fn main() -> Result<()> {
    let device = CudaDevice::default();
    log::info!("Training natively using Burn CUDA backend on: {:?}", device);

    // Initializing Burn's built-in derived Config natively
    let config = TinyLLMConfig::load("config.json").unwrap_or_else(|_| TinyLLMConfig::new());
    let mut train_config =
        TinyLLMTrainingConfig::load("config.json").unwrap_or_else(|_| TinyLLMTrainingConfig::new());
    log::info!("Burn Configuration Loaded from config.json!");

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
    log::info!("Burn Configured output directory: {}", output_dir);

    let auto_batch_size = DeviceSetup::calculate_batch_size(&device);
    let auto_gradient_accumulation = (64 / auto_batch_size).max(1);

    // Inject auto-scaling values directly into the derived config struct
    train_config.batch_size = auto_batch_size;
    train_config.gradient_accumulation_steps = auto_gradient_accumulation;

    log::info!(
        "NVML configured Gradient Accumulation tightly to {} to reach 64 macro-batch equivalence.",
        auto_gradient_accumulation
    );

    Trainer::train::<MyAutodiffBackend>(
        config,
        train_config,
        device,
        "fineweb_edu.bin",
        dataset_percentage,
        &output_dir,
    );

    Ok(())
}

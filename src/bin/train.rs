use anyhow::Result;
use candle_nn::VarMap;

use tiny_llm::{Config, TinyLLM};

#[path = "train_modules/dataset.rs"]
mod dataset;
#[path = "train_modules/device.rs"]
mod device;
#[path = "train_modules/tensorboard.rs"]
pub mod tensorboard;
#[path = "train_modules/trainer.rs"]
mod trainer;

use dataset::Dataset;
use device::DeviceSetup;
use trainer::{EarlyStoppingConfig, TrainableModelParameters, Trainer};

fn main() -> Result<()> {
    let device = candle_core::Device::new_cuda(0)?;
    println!("Training natively on: {:?}", device);

    let config = Config::load_from_file("config.json").unwrap_or_default();
    println!("Loaded Config: {:?}", config);

    // 2. Initialize Model with explicit tracking forced
    let varmap = VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::BF16, &device);
    let model = TinyLLM::new(config.clone(), vb)?;

    println!("Memory-mapping the dataset from SSD (0MB RAM usage)...");
    let mut dataset_percentage = 100.0;
    let mut output_dir_base = "checkpoints".to_string();

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
    println!("Configured output directory: {}", output_dir);

    let dataset = Dataset::new("fineweb_edu.bin", dataset_percentage)?;
    dataset.print_stats();

    let batch_size = DeviceSetup::calculate_batch_size(&device);
    let global_batch_size = 256;

    let mut grad_accum_steps = global_batch_size / batch_size;
    if grad_accum_steps == 0 {
        grad_accum_steps = 1;
    }
    let actual_global_batch_size = batch_size * grad_accum_steps;

    let max_steps = std::env::var("MAX_STEPS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or_else(|| {
            let target_epochs = 1.0;
            let tokens_per_step = actual_global_batch_size * config.seq_len;
            ((dataset.len() as f64 * target_epochs) / tokens_per_step as f64) as usize
        });

    let checkpoint_interval = (max_steps as f64 * 0.1) as usize;
    let checkpoint_interval = checkpoint_interval.max(1);

    let trainer_config = TrainableModelParameters {
        lr: 6e-4,
        global_batch_size,
        batch_size,
        max_steps,
        start_epoch: 1,
        checkpoint_interval,
        early_stopping: Some(EarlyStoppingConfig::default()),
        output_dir,
    };

    let mut trainer = Trainer::new(
        config.clone(),
        model,
        varmap,
        trainer_config,
        dataset.get_slice(),
        &device,
    )?;

    trainer.load_checkpoint()?;
    trainer.train()?;

    Ok(())
}

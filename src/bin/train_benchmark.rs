use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{loss::cross_entropy, VarBuilder, VarMap};
use memmap2::MmapOptions;
use rand::Rng;
use std::fs::File;
use std::time::Instant;
use tiny_llm::fused_adamw::{FusedAdamW, ParamsAdamW};

// Import our architecture from the lib.rs file
use tiny_llm::{Config, TinyLLM};

fn main() -> Result<()> {
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    println!("Training natively on: {:?}", device);

    let config = Config::load_from_file("config.json").unwrap_or_default();
    println!("Loaded Config: {:?}", config);

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::BF16, &device);
    let model = TinyLLM::new(config.clone(), vb)?;

    let mut start_epoch = 1;
    if let Ok(entries) = std::fs::read_dir(".") {
        let mut latest_checkpoint: Option<(usize, String)> = None;
        for entry in entries.flatten() {
            let name = entry.file_name().into_string().unwrap_or_default();
            if name.starts_with("fineweb_checkpoint_") && name.ends_with(".safetensors") {
                if let Ok(epoch) = name
                    ["fineweb_checkpoint_".len()..name.len() - ".safetensors".len()]
                    .parse::<usize>()
                {
                    if latest_checkpoint
                        .as_ref()
                        .is_none_or(|(latest, _)| epoch > *latest)
                    {
                        latest_checkpoint = Some((epoch, name));
                    }
                }
            }
        }
        if let Some((epoch, name)) = latest_checkpoint {
            varmap.load(&name)?;
            start_epoch = epoch + 1;
            println!("Resumed from checkpoint: {} at step {}", name, epoch);
        }
    }

    let lr = 6e-4;
    let params = ParamsAdamW {
        lr,
        ..Default::default()
    };
    let mut opt = FusedAdamW::new(varmap.all_vars(), params)?;

    println!("Memory-mapping the dataset from SSD (0MB RAM usage)...");
    let file = File::open("fineweb_edu.bin")
        .expect("Could not find fineweb_edu.bin! Did you run prep_data.py?");
    let mmap = unsafe { MmapOptions::new().map(&file)? };

    #[cfg(target_os = "linux")]
    {
        println!("Instructing Linux kernel to pre-fetch 38GB dataset into RAM (Madvise)...");
        let _ = mmap.advise(memmap2::Advice::Random);
        let _ = mmap.advise(memmap2::Advice::WillNeed);
        // We'll let the OS handle it in the background to avoid a massive blocking stall,
        // but `WillNeed` and `Random` massively increases Random Read throughput!
    }

    let dataset: &[u32] = bytemuck::cast_slice(&mmap);
    println!("Loaded {} tokens!", dataset.len());

    // Calculate optimal batch size automatically based on VRAM
    // TinyLLM consumes approximately 1024MB per batch size of 1 for SEQ_LEN=1024
    let mut batch_size = 8; // Default fallback for CPU

    // Calculate optimal batch size automatically based strictly on CUDA VRAM
    // TinyLLM consumes approximately 1024MB per batch size of 1 for SEQ_LEN=1024
    if device.is_cuda() {
        if let Ok(nvml) = nvml_wrapper::Nvml::init() {
            if let Ok(gpu) = nvml.device_by_index(0) {
                if let Ok(memory_info) = gpu.memory_info() {
                    let total_mb = memory_info.total as usize / 1_000_000;
                    let free_mb = memory_info.free as usize / 1_000_000;

                    let reserved_mb = 2000; // Leave 2GB for OS and CUDA context overhead
                    if free_mb > reserved_mb {
                        let available_mb_for_batches = free_mb - reserved_mb;
                        let calculated_batch = available_mb_for_batches / 1024;

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

    let global_batch_size = 256;
    let mut grad_accum_steps = global_batch_size / batch_size;
    if grad_accum_steps == 0 {
        grad_accum_steps = 1;
    }
    let actual_global_batch_size = batch_size * grad_accum_steps;
    println!(
        "Using Mini-batch: {}, Gradient Accumulation Steps: {} -> Actual Global Batch Size: {}",
        batch_size, grad_accum_steps, actual_global_batch_size
    );

    let mut rng = rand::thread_rng();

    println!("Starting BF16 Mixed Precision Training...");
    let mut start_time = Instant::now();
    let mut total_tokens = 0;

    let base_lr = 6e-4;
    let base_batch = 8.0;
    let max_lr = base_lr * (actual_global_batch_size as f64 / base_batch).sqrt();

    // Automatically calculate max steps based on processing 1 epoch over the dataset
    // or override via the MAX_STEPS environment variable.
    let max_steps = std::env::var("MAX_STEPS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or_else(|| {
            let target_epochs = 1.0;
            let tokens_per_step = actual_global_batch_size * config.seq_len;
            ((dataset.len() as f64 * target_epochs) / tokens_per_step as f64) as usize
        });

    // Warmup is always exactly 10% of the total steps
    let warmup_steps: usize = (max_steps as f64 * 0.1) as usize;

    let mut x_data_buf = vec![0u32; actual_global_batch_size * config.seq_len];
    let mut y_data_buf = vec![0u32; actual_global_batch_size * config.seq_len];

    for step in start_epoch..=max_steps {
        // 1. Calculate and update learning rate
        let lr = if step < warmup_steps {
            max_lr * ((step + 1) as f64 / warmup_steps as f64)
        } else {
            // Inverse square root decay
            max_lr * (warmup_steps as f64).sqrt() / (step as f64).sqrt()
        };
        opt.set_lr(lr);

        // Populate the Mega-Batch natively on CPU
        for b in 0..actual_global_batch_size {
            let start = rng.gen_range(0..dataset.len() - config.seq_len - 1);
            let b_offset = b * config.seq_len;
            x_data_buf[b_offset..b_offset + config.seq_len]
                .copy_from_slice(&dataset[start..start + config.seq_len]);
            y_data_buf[b_offset..b_offset + config.seq_len]
                .copy_from_slice(&dataset[start + 1..start + config.seq_len + 1]);
        }

        // Host-To-Device Transfer (Mega-Batch)
        // This transfers the entire batch once to the GPU, bypassing the PCIe bottleneck during accumulation
        let xs_mega = Tensor::from_slice(
            &x_data_buf,
            (actual_global_batch_size, config.seq_len),
            &device,
        )?;
        let ys_mega = Tensor::from_slice(
            &y_data_buf,
            (actual_global_batch_size, config.seq_len),
            &device,
        )?;

        // 2. Gradient Accumulation Loop
        let mut step_loss_sum = 0.0;
        for grad_step in 0..grad_accum_steps {
            // Slice the Mega-Batch natively within the GPU VRAM
            let xs = xs_mega
                .narrow(0, grad_step * batch_size, batch_size)?
                .contiguous()?;
            let ys = ys_mega
                .narrow(0, grad_step * batch_size, batch_size)?
                .contiguous()?;

            // Clear KV cache from the previous forward pass to prevent VRAM memory leaks
            model.clear_kv_cache();

            // Forward Pass natively in BF16
            let logits = model.forward(&xs, 0)?;

            let logits_f32 = logits.to_dtype(DType::F32)?;
            drop(logits); // Instantly free 800MB of VRAM allocated by BF16 matrix memory

            let logits_flat =
                logits_f32.reshape((batch_size * config.seq_len, config.vocab_size))?;
            let y_flat = ys.flatten_all()?;

            let loss = cross_entropy(&logits_flat, &y_flat)?;
            step_loss_sum += loss.to_scalar::<f32>()?;
            let grads = loss.backward()?;
            opt.accumulate(&grads, grad_accum_steps)?;

            total_tokens += batch_size * config.seq_len;
        }

        opt.step_with_accumulated()?;

        let avg_loss = step_loss_sum / grad_accum_steps as f32;

        let elapsed = start_time.elapsed().as_secs_f32();
        let tokens_per_sec = (total_tokens as f32) / elapsed;
        let percentage = (step as f32 / max_steps as f32) * 100.0;
        let eta_mins = ((max_steps - step) as f32 / 10.0 * elapsed) / 60.0;
        let total_seen = step * actual_global_batch_size * config.seq_len;

        println!(
            "Step {:05}/{} ({:>5.2}%) | Loss: {:.4} | LR: {:.1e} | Speed: {:>5.0} tok/s | Seen: {:.1}M / {:.1}M | ETA: {:.1}m",
            step, max_steps, percentage, avg_loss, lr, tokens_per_sec,
            total_seen as f32 / 1_000_000.0, dataset.len() as f32 / 1_000_000.0, eta_mins
        );
        start_time = Instant::now();
        total_tokens = 0;

        let checkpoint_interval = (max_steps as f64 * 0.1) as usize;
        let checkpoint_interval = checkpoint_interval.max(1);

        if step % checkpoint_interval == 0 {
            let filename = format!("fineweb_checkpoint_{}.safetensors", step);
            // varmap.save(&filename)?;
            println!(">> Saved {} to disk! <<", filename);
        }
    }
    Ok(())
}

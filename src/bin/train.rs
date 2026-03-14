use anyhow::Result;
use candle_core::{DType, Tensor};
use candle_nn::{VarBuilder, VarMap};
use memmap2::MmapOptions;
use rand::Rng;
use std::fs::File;
use std::time::Instant;
use tiny_llm::fused_adamw::{FusedAdamW, ParamsAdamW};

// Import our architecture from the lib.rs file
use tiny_llm::{Config, TinyLLM};

fn main() -> Result<()> {
    let device = candle_core::Device::new_cuda(0)?;
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
                        // Heuristic: A 12-layer 70M model at 1024 context window requires far less memory per batch item.
                        // We safely bound it to ~2000 MB per batch item to maintain high memory utilization.
                        let memory_per_batch_item_mb = 2000.0;
                        let calculated_batch =
                            (available_mb_for_batches as f64 / memory_per_batch_item_mb) as usize;

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

    let mut x_data_buf = vec![0u32; batch_size * config.seq_len];
    let mut y_data_buf = vec![0u32; batch_size * config.seq_len];

    for step in start_epoch..=max_steps {
        // 1. Calculate and update learning rate
        let lr = if step < warmup_steps {
            max_lr * ((step + 1) as f64 / warmup_steps as f64)
        } else {
            // Inverse square root decay
            max_lr * (warmup_steps as f64).sqrt() / (step as f64).sqrt()
        };
        opt.set_lr(lr);

        // 2. Gradient Accumulation Loop
        let mut avg_loss = 0.0;
        for i in 0..grad_accum_steps {
            for b in 0..batch_size {
                let start = rng.gen_range(0..dataset.len() - config.seq_len - 1);
                let b_offset = b * config.seq_len;
                x_data_buf[b_offset..b_offset + config.seq_len]
                    .copy_from_slice(&dataset[start..start + config.seq_len]);
                y_data_buf[b_offset..b_offset + config.seq_len]
                    .copy_from_slice(&dataset[start + 1..start + config.seq_len + 1]);
            }

            let xs = Tensor::from_slice(&x_data_buf, (batch_size, config.seq_len), &device)?;
            let ys = Tensor::from_slice(&y_data_buf, (batch_size, config.seq_len), &device)?;

            // Clear KV cache from the previous forward pass to prevent VRAM memory leaks
            model.clear_kv_cache();

            // Forward Pass natively projecting to logits via CuBLAS Matrix Tiling
            let (logits, out) = model.forward(&xs, 0)?;

            let fused_ce = tiny_llm::fused_cross_entropy::FusedCrossEntropy {
                targets: ys.clone(),
            };
            let fused_loss = logits.apply_op1(fused_ce)?;
            let loss = (fused_loss.sum_all()? / (logits.elem_count() / config.vocab_size) as f64)?;

            if i == grad_accum_steps - 1 {
                avg_loss = loss.to_dtype(candle_core::DType::F32)?.to_scalar::<f32>()?;
            }

            // Manually evaluate gradients to bypass Candle Autograd allocations
            use candle_core::CustomOp1;
            let n_tokens = (logits.elem_count() / config.vocab_size) as f32;
            let d_loss = (fused_loss.ones_like()? / (n_tokens as f64))?;

            let fused_ce_bwd = tiny_llm::fused_cross_entropy::FusedCrossEntropy { targets: ys };
            let d_logits = fused_ce_bwd.bwd(&logits, &fused_loss, &d_loss)?.unwrap();

            let d_logits_flat = d_logits.flatten_to(1)?;
            let out_flat = out.flatten_to(1)?;

            // To avoid forcing a 100MB contiguous transpose, we transpose the 1.5MB out_flat tensor!
            // d_W_t has shape [H, V]
            let d_w_t = out_flat.t()?.matmul(&d_logits_flat)?;
            // d_W has shape [V, H]
            let d_w = d_w_t.t()?;

            // Dummy tensor backward empty pass to initialize GradStore
            let dummy = candle_core::Tensor::zeros(1, candle_core::DType::F32, &device)?;
            let mut grads = dummy.backward()?;

            // Inject our gradient tensor natively into GradStore map directly
            grads.insert_id(model.lm_head().weight().id(), d_w);
            opt.accumulate(&grads, grad_accum_steps)?;

            total_tokens += batch_size * config.seq_len;
        }

        opt.step_with_accumulated()?;

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
            varmap.save(&filename)?;
            println!(">> Saved {} to disk! <<", filename);
        }
    }
    Ok(())
}

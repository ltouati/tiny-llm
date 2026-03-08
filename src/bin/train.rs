use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{loss::cross_entropy, VarBuilder, VarMap};
use memmap2::MmapOptions;
use rand::Rng;
use std::fs::File;
use std::time::Instant;
use tiny_llm::fused_adamw::{FusedAdamW, ParamsAdamW};

// Import our architecture from the lib.rs file
use tiny_llm::{TinyLLM, SEQ_LEN, VOCAB_SIZE};

fn main() -> Result<()> {
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    println!("Training natively on: {:?}", device);

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::BF16, &device);
    let model = TinyLLM::new(vb)?;

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

    let lr = 5e-4;
    let params = ParamsAdamW {
        lr,
        ..Default::default()
    };
    let mut opt = FusedAdamW::new(varmap.all_vars(), params)?;

    println!("Memory-mapping the dataset from SSD (0MB RAM usage)...");
    let file = File::open("fineweb_edu.bin")
        .expect("Could not find fineweb_edu.bin! Did you run prep_data.py?");
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let dataset: &[u32] = bytemuck::cast_slice(&mmap);
    println!("Loaded {} tokens!", dataset.len());

    let batch_size = 8;
    let mut rng = rand::thread_rng();

    println!("Starting BF16 Mixed Precision Training...");
    let mut start_time = Instant::now();
    let mut total_tokens = 0;

    for epoch in start_epoch..=50000 {
        let mut x_data = vec![0u32; batch_size * SEQ_LEN];
        let mut y_data = vec![0u32; batch_size * SEQ_LEN];

        for b in 0..batch_size {
            let start = rng.gen_range(0..dataset.len() - SEQ_LEN - 1);
            for t in 0..SEQ_LEN {
                x_data[b * SEQ_LEN + t] = dataset[start + t];
                y_data[b * SEQ_LEN + t] = dataset[start + t + 1];
            }
        }

        let xs = Tensor::from_vec(x_data, (batch_size, SEQ_LEN), &device)?;
        let ys = Tensor::from_vec(y_data, (batch_size, SEQ_LEN), &device)?;

        // Forward Pass natively in BF16
        let logits = model.forward(&xs)?;

        // Mixed Precision Cast to F32 to prevent Softmax Overflow
        let logits_f32 = logits.to_dtype(DType::F32)?;
        let logits_flat = logits_f32.reshape((batch_size * SEQ_LEN, VOCAB_SIZE))?;
        let y_flat = ys.flatten_all()?;

        // Backprop
        let loss = cross_entropy(&logits_flat, &y_flat)?;
        let grads = loss.backward()?;
        opt.step(&grads)?;

        total_tokens += batch_size * SEQ_LEN;

        if epoch % 50 == 0 {
            let elapsed = start_time.elapsed().as_secs_f32();
            let tokens_per_sec = (total_tokens as f32) / elapsed;
            let percentage = (epoch as f32 / 50_000.0) * 100.0;
            let eta_mins = ((50_000 - epoch) as f32 / 50.0 * elapsed) / 60.0;
            let total_seen = epoch * batch_size * SEQ_LEN;

            println!(
                "Step {:05}/50000 ({:>5.2}%) | Loss: {:.4} | LR: {:.1e} | Speed: {:>5.0} tok/s | Seen: {:.1}M / {:.1}M | ETA: {:.1}m",
                epoch, percentage, loss.to_scalar::<f32>()?, lr, tokens_per_sec,
                total_seen as f32 / 1_000_000.0, dataset.len() as f32 / 1_000_000.0, eta_mins
            );
            start_time = Instant::now();
            total_tokens = 0;
        }

        if epoch % 5000 == 0 {
            let filename = format!("fineweb_checkpoint_{}.safetensors", epoch);
            varmap.save(&filename)?;
            println!(">> Saved {} to disk! <<", filename);
        }
    }
    Ok(())
}

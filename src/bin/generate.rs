use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::VarBuilder;
use std::io::{self, Write};
use tokenizers::Tokenizer;

// Import our architecture from the lib.rs file
use tiny_llm::{Config, TinyLLM};

fn main() -> Result<()> {
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    println!("Loading model on device: {:?}", device);

    let tokenizer =
        Tokenizer::from_file("tokenizer.json").map_err(|e| anyhow::anyhow!(e.to_string()))?;

    // Scan for highest checkpoint
    let mut checkpoint_file = String::new();
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
        if let Some((_, name)) = latest_checkpoint {
            checkpoint_file = name;
        }
    }

    if checkpoint_file.is_empty() || !std::path::Path::new(&checkpoint_file).exists() {
        println!("Error: Could not find any fineweb_checkpoint_*.safetensors files. Have you trained the model enough to save a checkpoint yet?");
        return Ok(());
    }

    println!("Loading weights from {}...", checkpoint_file);
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[checkpoint_file], DType::BF16, &device)? };

    let config = Config::load_from_file("config.json").unwrap_or_default();
    let model = TinyLLM::new(config, vb)?;

    let args: Vec<String> = std::env::args().collect();
    let prompt = if args.len() > 1 {
        args[1..].join(" ")
    } else {
        "The Apollo 11 moon landing was ".to_string()
    };

    print!("\nAI: {}", prompt);
    io::stdout().flush()?;

    let mut tokens = tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?
        .get_ids()
        .to_vec();

    let max_tokens_to_generate = 100;

    let mut start_pos = 0;
    let mut next_tokens = tokens.clone();

    for _ in 0..max_tokens_to_generate {
        let input = Tensor::from_vec(next_tokens.clone(), (1, next_tokens.len()), &device)?;
        let logits = model.forward(&input, start_pos)?;

        let (_, seq_len, _) = logits.dims3()?;
        let last_token_logits = logits
            .narrow(1, seq_len - 1, 1)?
            .squeeze(1)?
            .squeeze(0)?
            .to_dtype(DType::F32)?;

        // Temperature Sampling
        let temperature = 0.8f64;
        let prs = candle_nn::ops::softmax(&(last_token_logits / temperature)?, D::Minus1)?
            .to_vec1::<f32>()?;

        let mut rng = rand::thread_rng();
        let dist = rand::distributions::WeightedIndex::new(&prs)
            .map_err(|e| anyhow::anyhow!("Sampling error: {}", e))?;

        use rand::distributions::Distribution;
        let next_token_id = dist.sample(&mut rng) as u32;

        if next_token_id == 50256 {
            break; // Stop if it hits the <|endoftext|> token
        }

        tokens.push(next_token_id);

        start_pos += next_tokens.len();
        next_tokens = vec![next_token_id];

        let new_word = tokenizer
            .decode(&[next_token_id], true)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;

        print!("{}", new_word);
        io::stdout().flush()?;
    }

    println!("\n\n[Generation complete]");
    Ok(())
}

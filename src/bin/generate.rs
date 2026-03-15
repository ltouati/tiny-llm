use anyhow::Result;
use burn::backend::cuda::CudaDevice;
use burn::backend::Cuda;
use burn::prelude::*;

use burn::tensor::activation::softmax;
use burn_store::{ModuleSnapshot, SafetensorsStore};
use std::io::{self, Write};
use tokenizers::Tokenizer;

use tiny_llm::config::TinyLLMConfig;
use tiny_llm::model::TinyLLM;

fn main() -> Result<()> {
    type Backend = Cuda<half::bf16, i32>;
    let device = CudaDevice::default();
    log::info!("Loading Model Graph natively on: {:?}", device);

    let tokenizer =
        Tokenizer::from_file("tokenizer.json").map_err(|e| anyhow::anyhow!(e.to_string()))?;

    // Search for the latest Burn checkpoint generically
    let mut checkpoint_file = String::new();
    let root_dir = "checkpoints_burn";

    if let Ok(run_dirs) = std::fs::read_dir(root_dir) {
        let mut latest_checkpoint: Option<(std::time::SystemTime, String)> = None;

        for run_dir_entry in run_dirs.flatten() {
            let run_dir = run_dir_entry.path();
            if run_dir.is_dir() {
                if let Ok(step_dirs) = std::fs::read_dir(&run_dir) {
                    for step_dir_entry in step_dirs.flatten() {
                        let step_dir = step_dir_entry.path();
                        if step_dir.is_dir() {
                            let safetensors_path = step_dir.join("model.safetensors");
                            if safetensors_path.exists() {
                                if let Ok(metadata) = std::fs::metadata(&safetensors_path) {
                                    if let Ok(modified) = metadata.modified() {
                                        if latest_checkpoint
                                            .as_ref()
                                            .is_none_or(|(latest, _)| modified > *latest)
                                        {
                                            latest_checkpoint = Some((
                                                modified,
                                                safetensors_path.to_str().unwrap().to_string(),
                                            ));
                                        }
                                    }
                                }
                            }
                        } else if step_dir.is_file()
                            && step_dir.extension().is_some_and(|ext| ext == "safetensors")
                        {
                            if let Ok(metadata) = std::fs::metadata(&step_dir) {
                                if let Ok(modified) = metadata.modified() {
                                    if latest_checkpoint
                                        .as_ref()
                                        .is_none_or(|(latest, _)| modified > *latest)
                                    {
                                        latest_checkpoint = Some((
                                            modified,
                                            step_dir.to_str().unwrap().to_string(),
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if let Some((_, name)) = latest_checkpoint {
            checkpoint_file = name;
        }
    }

    let config = TinyLLMConfig::load("config.json").unwrap_or_else(|_| TinyLLMConfig::new());
    let mut model = TinyLLM::<Backend>::new(&config, &device);

    if !checkpoint_file.is_empty() {
        log::info!("Loading weights natively from {}...", checkpoint_file);
        let mut store = SafetensorsStore::from_file(&checkpoint_file);
        // Safetensors load
        model
            .load_from(&mut store)
            .expect("Failed to parse Safetensors natively");
    } else {
        log::info!("Warning: No checkpoints found. Running fully initialized random weights!");
    }

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
    let mut next_tokens = tokens.clone();

    for _ in 0..max_tokens_to_generate {
        let input_i32: Vec<i32> = next_tokens.iter().map(|&x| x as i32).collect();
        let input = Tensor::<Backend, 1, Int>::from_ints(input_i32.as_slice(), &device)
            .reshape([1, input_i32.len() as usize]);

        let logits = model.forward(input);
        let [_batch_size, seq_len, vocab_size] = logits.dims();

        // Get the very last prediction natively [1, 1, VocabSize] -> [VocabSize]
        let last_token_logits = logits
            .slice([0..1, (seq_len - 1)..seq_len, 0..vocab_size])
            .reshape([vocab_size as i32]);

        let temperature = 0.8f32;
        let last_token_logits = last_token_logits.div_scalar(temperature);
        let prs = softmax(last_token_logits, 0);

        let prs_vec = prs.into_data().convert::<f32>().to_vec::<f32>().unwrap();

        let mut rng = rand::thread_rng();
        let dist = rand::distributions::WeightedIndex::new(&prs_vec)
            .map_err(|e| anyhow::anyhow!("Sampling error: {}", e))?;

        use rand::distributions::Distribution;
        let next_token_id = dist.sample(&mut rng) as u32;

        if next_token_id == 50256 {
            break; // Stop at <|endoftext|>
        }

        tokens.push(next_token_id);
        next_tokens = vec![next_token_id];

        let new_word = tokenizer
            .decode(&[next_token_id], true)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;

        print!("{}", new_word);
        io::stdout().flush()?;
    }

    log::info!("\n\n[Generation complete]");
    Ok(())
}

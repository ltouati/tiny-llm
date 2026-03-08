use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::VarBuilder;
use serde_json::Value;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;
use tokenizers::Tokenizer;

use tiny_llm::TinyLLM;

fn main() -> Result<()> {
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    println!("Evaluating model natively on device: {:?}", device);

    let tokenizer =
        Tokenizer::from_file("tokenizer.json").map_err(|e| anyhow::anyhow!(e.to_string()))?;

    // Download HellaSwag validation set if not exists
    let hellaswag_url =
        "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl";
    let hellaswag_file = "hellaswag_val.jsonl";

    if !Path::new(hellaswag_file).exists() {
        println!("Downloading HellaSwag validation dataset...");
        let response = reqwest::blocking::get(hellaswag_url)?.text()?;
        std::fs::write(hellaswag_file, response)?;
        println!("Download complete.");
    }

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
        println!("Error: Could not find any fineweb_checkpoint_*.safetensors files.");
        return Ok(());
    }

    println!("Loading weights from {}...", checkpoint_file);
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[checkpoint_file], DType::BF16, &device)? };
    let model = TinyLLM::new(vb)?;

    let file = File::open(hellaswag_file)?;
    let reader = BufReader::new(file);

    let mut total = 0;
    let mut correct = 0;

    println!("Starting evaluation...");

    for line in reader.lines() {
        let line = line?;
        let v: Value = serde_json::from_str(&line)?;

        let ctx = v["ctx"].as_str().unwrap_or("");
        let endings = v["endings"].as_array().unwrap();

        // HellaSwag labels can be numeric or strings depending on the dataset dump
        let label = match &v["label"] {
            Value::Number(n) => n.as_i64().unwrap_or(0) as usize,
            Value::String(s) => s.parse::<usize>().unwrap_or(0),
            _ => 0,
        };

        let ctx_tokens = tokenizer
            .encode(ctx, false)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .get_ids()
            .to_vec();

        let mut best_score = f32::NEG_INFINITY;
        let mut best_idx = 0;

        for (idx, ending) in endings.iter().enumerate() {
            let ending_str = ending.as_str().unwrap_or("");
            // Add a leading space so tokenizer doesn't attach words inappropriately to the context boundary
            let pad_ending = format!(" {}", ending_str);
            let ending_tokens = tokenizer
                .encode(pad_ending, false)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?
                .get_ids()
                .to_vec();

            if ending_tokens.is_empty() {
                continue;
            }

            let mut all_tokens = ctx_tokens.clone();
            all_tokens.extend(&ending_tokens);

            // Limit to our architecture's max sequence length before OOM
            let seq_len = tiny_llm::SEQ_LEN;
            let start_idx = if all_tokens.len() > seq_len {
                all_tokens.len() - seq_len
            } else {
                0
            };

            // To ensure we don't accidentally panic if ending string is massively longer than seq_len
            let sliced_tokens = &all_tokens[start_idx..];
            let ctx_len_in_all = sliced_tokens.len().saturating_sub(ending_tokens.len());

            let input =
                Tensor::from_vec(sliced_tokens.to_vec(), (1, sliced_tokens.len()), &device)?;
            let logits = model.forward(&input)?;

            // Drop BF16 gradients to F32 for stable math
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
            let log_probs = candle_nn::ops::log_softmax(&logits, D::Minus1)?;
            let log_probs_host = log_probs.to_vec2::<f32>()?;

            let mut sum_log_prob = 0.0;

            // The logits at pos `j - 1` predict the token at pos `j`
            for j in ctx_len_in_all..sliced_tokens.len() {
                if j == 0 {
                    continue;
                }
                let target_token = sliced_tokens[j];
                sum_log_prob += log_probs_host[j - 1][target_token as usize];
            }

            // Average log probability
            let avg_log_prob = sum_log_prob / (ending_tokens.len() as f32);

            if avg_log_prob > best_score {
                best_score = avg_log_prob;
                best_idx = idx;
            }
        }

        if best_idx == label {
            correct += 1;
        }
        total += 1;

        if total % 10 == 0 || total == 10042 {
            print!(
                "\rEvaluating HellaSwag: {}/10042 (Acc: {:.2}%)",
                total,
                (correct as f32 / total as f32) * 100.0
            );
            io::stdout().flush()?;
        }
    }

    println!(
        "\n\nFinal HellaSwag Accuracy: {:.2}%",
        (correct as f32 / total as f32) * 100.0
    );
    Ok(())
}

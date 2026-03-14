use anyhow::Result;
use axum::{extract::State, routing::post, Json, Router};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::VarBuilder;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;

use tiny_llm::{Config, TinyLLM};

#[derive(Clone)]
struct AppState {
    model: Arc<Mutex<TinyLLM>>,
    tokenizer: Arc<Tokenizer>,
    config: Config,
    device: Device,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct CompletionRequest {
    model: Option<String>,
    prompt: String,
    max_tokens: Option<usize>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    stop: Option<Vec<String>>,
    echo: Option<bool>,
    logprobs: Option<usize>,
}

#[derive(Serialize)]
struct CompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<Choice>,
}

#[derive(Serialize)]
struct Choice {
    text: String,
    index: usize,
    finish_reason: String,
    logprobs: Option<Logprobs>,
}

#[derive(Serialize)]
struct Logprobs {
    tokens: Vec<String>,
    token_logprobs: Vec<f64>,
    top_logprobs: Vec<std::collections::HashMap<String, f64>>,
    text_offset: Vec<usize>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut mode = "simple";
    for i in 0..args.len() {
        if args[i] == "--mode" && i + 1 < args.len() {
            mode = &args[i + 1];
        }
    }

    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    println!("Evaluating model natively on device: {:?}", device);

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
        println!("Error: Could not find any fineweb_checkpoint_*.safetensors files.");
        return Ok(());
    }

    println!("Loading weights from {}...", checkpoint_file);
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[checkpoint_file], DType::BF16, &device)? };

    let config = Config::load_from_file("config.json").unwrap_or_default();
    let model = TinyLLM::new(config.clone(), vb)?;

    if mode == "extended" {
        run_extended_eval(model, tokenizer, config, device).await
    } else {
        run_simple_eval(model, tokenizer, config, device).await
    }
}

async fn run_extended_eval(
    model: TinyLLM,
    tokenizer: Tokenizer,
    config: Config,
    device: Device,
) -> Result<()> {
    let state = AppState {
        model: Arc::new(Mutex::new(model)),
        tokenizer: Arc::new(tokenizer),
        config,
        device,
    };

    let app = Router::new()
        .route("/v1/completions", post(handle_completions))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:8080").await?;
    println!("Started local OpenAI API server at http://127.0.0.1:8080");

    let _server_handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let venv_dir = ".lm_eval_env";
    if !Path::new(venv_dir).exists() {
        println!("Creating Python virtual environment in {}...", venv_dir);
        let status = Command::new("python3")
            .args(["-m", "venv", venv_dir])
            .status()?;
        if !status.success() {
            anyhow::bail!("Failed to create virtual environment");
        }

        println!("Installing lm-eval[api] and transformers...");
        let pip_path = format!("{}/bin/pip", venv_dir);
        let status = Command::new(&pip_path)
            .args([
                "install",
                "lm-eval[api]",
                "transformers",
                "git+https://github.com/felipemaiapolo/tinyBenchmarks",
            ])
            .status()?;
        if !status.success() {
            anyhow::bail!("Failed to install lm-eval[api] and transformers");
        }
    }

    println!("Running lm-eval...");
    let lm_eval_path = format!("{}/bin/lm_eval", venv_dir);
    let mut child = Command::new(&lm_eval_path)
        .args([
            "--model", "local-completions",
            "--model_args", "model=gpt2,base_url=http://127.0.0.1:8080/v1/completions,num_concurrent=1,max_retries=3,token=empty",
            "--tasks", "arc_easy,lambada_openai,piqa,tinyMMLU",
            "--batch_size", "1",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()?;

    child.wait()?;

    println!("Evaluation complete! Shutting down server.");
    std::process::exit(0);
}

async fn handle_completions(
    State(state): State<AppState>,
    Json(req): Json<Value>,
) -> axum::response::Response {
    let result = handle_completions_inner(state, req).await;
    match result {
        Ok(res) => axum::response::IntoResponse::into_response(Json(res)),
        Err(e) => {
            println!("Server Error: {:?}", e);
            axum::response::IntoResponse::into_response((
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                e.to_string(),
            ))
        }
    }
}

async fn handle_completions_inner(
    state: AppState,
    req: Value,
) -> Result<CompletionResponse, anyhow::Error> {
    // A primitive inference loop based on generate.rs
    let tokenizer = &state.tokenizer;

    let prompt = req["prompt"].as_str().unwrap_or("").to_string();
    let max_tokens = req["max_tokens"].as_u64().unwrap_or(16) as usize;
    let temperature = req["temperature"].as_f64().unwrap_or(1.0);
    let echo = req["echo"].as_bool().unwrap_or(false);

    let mut tokens = tokenizer
        .encode(prompt.clone(), false)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?
        .get_ids()
        .to_vec();

    let mut generated_text = String::new();

    // Acquire model lock
    let model = state.model.lock().await;

    // For loglikelihood requests (lm-eval), it sometimes passes max_tokens=0 and echo=true.
    // In that case it evaluates logprobs of the prompt.
    // This simple handler mainly focuses on generative multiple choice.
    // We will generate the max_tokens.
    // Logprobs handling for lm-eval
    let req_logprobs = req["logprobs"].as_u64();
    let mut out_tokens = Vec::new();
    let mut out_token_logprobs = Vec::new();
    let mut out_top_logprobs: Vec<std::collections::HashMap<String, f64>> = Vec::new();

    if max_tokens > 0 {
        let mut rng = rand::thread_rng();

        // If echo is true, we should include the prompt's logprobs
        if echo && req_logprobs.is_some() && !tokens.is_empty() {
            let input = Tensor::from_vec(tokens.clone(), (1, tokens.len()), &state.device)?;
            let logits = model.forward(&input, 0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let log_probs = candle_nn::ops::log_softmax(&logits, D::Minus1)?.to_vec2::<f32>()?;

            for j in 0..tokens.len() {
                let t = tokens[j];
                out_tokens.push(tokenizer.decode(&[t], true).unwrap_or_default());
                out_top_logprobs.push(std::collections::HashMap::new());
                if j == 0 {
                    out_token_logprobs.push(0.0);
                } else {
                    out_token_logprobs.push(log_probs[j - 1][t as usize] as f64);
                }
            }
        }

        for _ in 0..max_tokens {
            let seq_len = tokens.len();
            if seq_len == 0 {
                tokens.push(50256);
            }
            let max_seq_len = state.config.seq_len;
            let start_idx = seq_len.saturating_sub(max_seq_len);
            let sliced_tokens = &tokens[start_idx..];

            let input = Tensor::from_vec(
                sliced_tokens.to_vec(),
                (1, sliced_tokens.len()),
                &state.device,
            )?;
            let logits = model.forward(&input, 0)?;

            let (_, s_len, _) = logits.dims3()?;
            let last_token_logits = logits
                .narrow(1, s_len - 1, 1)?
                .squeeze(1)?
                .squeeze(0)?
                .to_dtype(DType::F32)?;

            let next_token_id = if temperature < 1e-4 {
                // Greedy
                let last_token_logits = last_token_logits.to_vec1::<f32>()?;
                last_token_logits
                    .into_iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap()
                    .0 as u32
            } else {
                let prs = candle_nn::ops::softmax(
                    &(last_token_logits.clone() / temperature)?,
                    D::Minus1,
                )?
                .to_vec1::<f32>()?;

                use rand::distributions::Distribution;
                let dist = rand::distributions::WeightedIndex::new(&prs)?;
                dist.sample(&mut rng) as u32
            };

            // Calculate logprob of the chosen token
            if req_logprobs.is_some() {
                let log_probs =
                    candle_nn::ops::log_softmax(&last_token_logits, D::Minus1)?.to_vec1::<f32>()?;
                out_tokens.push(tokenizer.decode(&[next_token_id], true).unwrap_or_default());
                out_token_logprobs.push(log_probs[next_token_id as usize] as f64);
                out_top_logprobs.push(std::collections::HashMap::new());
            }

            if next_token_id == 50256 {
                break;
            }

            tokens.push(next_token_id);
            let next_word = tokenizer
                .decode(&[next_token_id], true)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            generated_text.push_str(&next_word);
        }
    } else if echo {
        generated_text = prompt;

        // purely evaluating logprobs of the prompt
        if req_logprobs.is_some() && !tokens.is_empty() {
            let input = Tensor::from_vec(tokens.clone(), (1, tokens.len()), &state.device)?;
            let logits = model.forward(&input, 0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let log_probs = candle_nn::ops::log_softmax(&logits, D::Minus1)?.to_vec2::<f32>()?;

            for j in 0..tokens.len() {
                let t = tokens[j];
                out_tokens.push(tokenizer.decode(&[t], true).unwrap_or_default());
                out_top_logprobs.push(std::collections::HashMap::new());
                if j == 0 {
                    out_token_logprobs.push(0.0); // no prior token prediction
                } else {
                    out_token_logprobs.push(log_probs[j - 1][t as usize] as f64);
                }
            }
        }
    }

    let logprobs_opt = if req_logprobs.is_some() && !out_tokens.is_empty() {
        let text_offset = vec![0; out_tokens.len()];
        Some(Logprobs {
            tokens: out_tokens,
            token_logprobs: out_token_logprobs,
            top_logprobs: out_top_logprobs,
            text_offset,
        })
    } else {
        None
    };

    Ok(CompletionResponse {
        id: "cmpl-1234".to_string(),
        object: "text_completion".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: "tiny-llm".to_string(),
        choices: vec![Choice {
            text: generated_text,
            index: 0,
            finish_reason: "length".to_string(),
            logprobs: logprobs_opt,
        }],
    })
}

async fn run_simple_eval(
    model: TinyLLM,
    tokenizer: Tokenizer,
    config: Config,
    device: Device,
) -> Result<()> {
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

    let file = File::open(hellaswag_file)?;
    let reader = BufReader::new(file);

    let mut total = 0;
    let mut correct = 0;

    let mut indomain_correct = 0;
    let mut indomain_total = 0;

    let mut zeroshot_correct = 0;
    let mut zeroshot_total = 0;

    let mut activitynet_correct = 0;
    let mut activitynet_total = 0;

    let mut wikihow_correct = 0;
    let mut wikihow_total = 0;

    println!("Starting evaluation...");

    for line in reader.lines() {
        let line = line?;
        let v: Value = serde_json::from_str(&line)?;

        let ctx = v["ctx"].as_str().unwrap_or("");
        let split_type = v["split_type"].as_str().unwrap_or("");
        let source_id = v["source_id"].as_str().unwrap_or("");
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

        let mut max_len = 0;
        let mut batch_tokens = Vec::new();
        let mut ctx_lengths = Vec::new();
        let mut ending_lengths = Vec::new();

        // 1. Prepare batch padded arrays
        for ending in endings.iter() {
            let ending_str = ending.as_str().unwrap_or("");
            let pad_ending = format!(" {}", ending_str);
            let ending_tokens = tokenizer
                .encode(pad_ending, false)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?
                .get_ids()
                .to_vec();

            let mut all_tokens = ctx_tokens.clone();
            all_tokens.extend(&ending_tokens);

            // Limit to our architecture's max sequence length
            let seq_len = config.seq_len;
            let start_idx = if all_tokens.len() > seq_len {
                all_tokens.len() - seq_len
            } else {
                0
            };

            let sliced_tokens = all_tokens[start_idx..].to_vec();
            if sliced_tokens.len() > max_len {
                max_len = sliced_tokens.len();
            }

            let ctx_len_in_all = sliced_tokens.len().saturating_sub(ending_tokens.len());
            ctx_lengths.push(ctx_len_in_all);
            ending_lengths.push(ending_tokens.len());
            batch_tokens.push(sliced_tokens);
        }

        if batch_tokens.is_empty() {
            continue;
        }

        // 2. Pad to max_len
        let mut batched_data = Vec::new();
        for mut tokens in batch_tokens {
            let pad_len = max_len - tokens.len();
            tokens.extend(vec![50256; pad_len]); // pad with EOS
            batched_data.extend(tokens);
        }

        // 3. Single batched forward pass representing all 4 options!
        let num_choices = endings.len();
        let input = Tensor::from_vec(batched_data, (num_choices, max_len), &device)?;

        model.clear_kv_cache(); // Ensure KV cache starts clean for the whole batch
        let logits = model.forward(&input, 0)?;

        let logits = logits.to_dtype(DType::F32)?;
        let log_probs = candle_nn::ops::log_softmax(&logits, D::Minus1)?;
        let log_probs_host = log_probs.to_vec3::<f32>()?;

        let mut best_score = f32::NEG_INFINITY;
        let mut best_idx = 0;

        // 4. Extract logs for individual choices
        for idx in 0..num_choices {
            let mut sum_log_prob = 0.0;
            let ctx_len = ctx_lengths[idx];
            let ending_len = ending_lengths[idx];

            // Reconstruct sliced tokens map for target matching
            let tokens_slice = &input.to_vec2::<u32>()?[idx];

            for (j, &target_token) in tokens_slice
                .iter()
                .enumerate()
                .skip(ctx_len)
                .take(ending_len)
            {
                if j == 0 {
                    continue;
                }
                sum_log_prob += log_probs_host[idx][j - 1][target_token as usize];
            }

            let avg_log_prob = sum_log_prob / (ending_len as f32);

            if avg_log_prob > best_score {
                best_score = avg_log_prob;
                best_idx = idx;
            }
        }

        if best_idx == label {
            correct += 1;

            if split_type == "indomain" {
                indomain_correct += 1;
            }
            if split_type == "zeroshot" {
                zeroshot_correct += 1;
            }
            if source_id.starts_with("activitynet") {
                activitynet_correct += 1;
            }
            if source_id.starts_with("wikihow") {
                wikihow_correct += 1;
            }
        }

        if split_type == "indomain" {
            indomain_total += 1;
        }
        if split_type == "zeroshot" {
            zeroshot_total += 1;
        }
        if source_id.starts_with("activitynet") {
            activitynet_total += 1;
        }
        if source_id.starts_with("wikihow") {
            wikihow_total += 1;
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

    println!("\n\n--- HellaSwag Category Breakdown ---");
    println!(
        "{:<35} | {:<10} | {:<10}",
        "Category", "Accuracy", "Samples"
    );
    println!("{:-<35}-+-{:-<10}-+-{:-<10}-", "", "", "");

    let print_row = |name: &str, c: usize, t: usize| {
        if t > 0 {
            let acc = (c as f32 / t as f32) * 100.0;
            println!("{:<35} | {:>9.2}% | {:>9}", name, acc, t);
        }
    };

    print_row("In-domain Category", indomain_correct, indomain_total);
    print_row("Zero-shot Category", zeroshot_correct, zeroshot_total);
    print_row("ActivityNet Format", activitynet_correct, activitynet_total);
    print_row("WikiHow Format", wikihow_correct, wikihow_total);
    println!("{:-<35}-+-{:-<10}-+-{:-<10}-", "", "", "");

    let final_acc = (correct as f32 / total as f32) * 100.0;
    println!("\nFinal HellaSwag Accuracy: {:.2}%", final_acc);

    // Explicit baseline context
    println!("\n*Note: HellaSwag consists of 4-choice multiple choice questions.");
    println!(" A score of 25.0% represents random guessing. At this early stage of");
    println!(" training, a low score is expected until the model learns reasoning.*");

    Ok(())
}

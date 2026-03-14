use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use parquet::file::reader::FileReader;
use rayon::prelude::*;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;

use tiny_llm::Config;

const TARGET_TOKENS: usize = 10_000_000_000;
const OUTPUT_FILE: &str = "fineweb_edu.bin";

#[tokio::main]
async fn main() -> Result<()> {
    let config = Config::load_from_file("config.json").unwrap_or_default();
    println!("Loaded config: seq_len = {}", config.seq_len);

    let tokenizer = Tokenizer::from_file("tokenizer.json")
        .map_err(|e| anyhow::anyhow!(e.to_string()))
        .context("Could not load tokenizer.json. Did you download it?")?;

    let out_file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(OUTPUT_FILE)?;
    let mut out_file = BufWriter::with_capacity(64 * 1024 * 1024, out_file);

    let mut total_tokens = 0;

    // Create a powerful MPSC channel that holds 2 completely downloaded shards.
    // This allows the hf-hub network downloader to stay 2 shards ahead of the CPU pipeline!
    let (tx, mut rx) = mpsc::channel(2);

    let downloader = tokio::spawn(async move {
        let api = match hf_hub::api::tokio::ApiBuilder::new()
            .with_progress(false)
            .build()
        {
            Ok(a) => a,
            Err(e) => {
                println!("Failed to build hf-hub API: {}", e);
                return;
            }
        };
        let repo = api
            .dataset("HuggingFaceFW/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT-shuffled".to_string());

        for shard in 0..100 {
            let filename = format!("data/train-{:05}-of-00100.parquet", shard);

            match repo.get(&filename).await {
                Ok(path) => {
                    if tx.send((shard, path)).await.is_err() {
                        // The receiver closed.
                        break;
                    }
                }
                Err(e) => {
                    println!("Failed to download shard {}: {}", shard, e);
                }
            }
        }
    });

    while let Some((shard, path)) = rx.recv().await {
        println!(
            "Extracting and Tokenizing shard {} loaded from HF disk cache...",
            shard
        );

        let file = std::fs::File::open(&path)?;
        let reader = parquet::file::reader::SerializedFileReader::new(file)?;

        let mut texts = Vec::new();
        for row_group_meta in reader.metadata().row_groups() {
            let row_group_reader =
                reader.get_row_group(row_group_meta.ordinal().unwrap_or(0) as usize)?;
            let row_iter = row_group_reader.get_row_iter(None)?;

            for record in row_iter {
                let row = record?;
                for field in row.get_column_iter() {
                    match &field.1 {
                        parquet::record::Field::Str(text) => {
                            if !text.is_empty() {
                                texts.push(text.to_string());
                            }
                            break;
                        }
                        _ => continue,
                    }
                }
            }
        }

        let texts_len = texts.len();
        println!(
            "Parquet parsed {} documents. Mapped natively to exactly {} Ryzen CPUs...",
            texts_len,
            rayon::current_num_threads()
        );

        let pb_tokens = ProgressBar::new(texts_len as u64);
        pb_tokens.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} docs ({per_sec})")
            .unwrap()
            .progress_chars("#>-"));

        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        let texts_len_clone = texts_len;
        let pb_clone = pb_tokens.clone();

        let pb_thread = std::thread::spawn(move || loop {
            std::thread::sleep(std::time::Duration::from_millis(500));
            let current = c.load(Ordering::Relaxed);
            pb_clone.set_position(current as u64);
            if current >= texts_len_clone {
                break;
            }
        });

        let chunked_tokens: Vec<Vec<u32>> = texts
            .par_iter()
            .filter_map(|text| {
                let mut result = None;
                if let Ok(encoding) = tokenizer.encode(text.as_str(), false) {
                    let ids = encoding.get_ids();
                    if ids.len() <= config.seq_len {
                        let mut tokens = ids.to_vec();
                        tokens.push(50256); // <|endoftext|>
                        result = Some(tokens);
                    }
                }
                counter.fetch_add(1, Ordering::Relaxed);
                result
            })
            .collect();

        let _ = pb_thread.join();
        pb_tokens.finish_with_message(format!("Tokenization complete for shard {}", shard));

        for doc_tokens in chunked_tokens {
            let bytes: &[u8] = bytemuck::cast_slice(&doc_tokens);
            out_file.write_all(bytes)?;
            total_tokens += doc_tokens.len();

            if total_tokens >= TARGET_TOKENS {
                println!("Hit target of 10B tokens!");
                break;
            }
        }

        println!(
            "Shard {} complete. Total tokens so far: {:.1}M / {:.1}M",
            shard,
            total_tokens as f64 / 1_000_000.0,
            TARGET_TOKENS as f64 / 1_000_000.0
        );

        if total_tokens >= TARGET_TOKENS {
            break;
        }
    }

    downloader.abort();
    println!("Done! Saved {} tokens to {}.", total_tokens, OUTPUT_FILE);

    Ok(())
}

use crate::tensorboard::{StatisticsCollector, TensorboardStatisticsCollector};
use anyhow::Result;
use candle_core::Tensor;
use candle_nn::VarMap;
use rand::Rng;
use std::time::Instant;
use tiny_llm::kernel::fused_adamw::{FusedAdamW, ParamsAdamW};
use tiny_llm::kernel::fused_cross_entropy::FusedCrossEntropy;
use tiny_llm::{Config, TinyLLM};

#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig {
    pub patience: usize,
    pub min_delta: f32,
    pub min_steps: usize,
    pub best_model_path: Option<String>,
}

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        Self {
            patience: 50,
            min_delta: 1e-4,
            min_steps: 100,
            best_model_path: None,
        }
    }
}

#[derive(Clone)]
pub struct TrainableModelParameters {
    pub lr: f64,
    pub global_batch_size: usize,
    pub batch_size: usize,
    pub max_steps: usize,
    pub start_epoch: usize,
    pub checkpoint_interval: usize,
    pub early_stopping: Option<EarlyStoppingConfig>,
    pub output_dir: String,
}

impl TrainableModelParameters {
    pub fn get_lr(&self, step: usize, actual_global_batch_size: usize) -> f64 {
        let base_lr = self.lr;
        let base_batch = 8.0;
        let max_lr = base_lr * (actual_global_batch_size as f64 / base_batch).sqrt();

        // Warmup is always exactly 10% of the total steps
        let warmup_steps: usize = (self.max_steps as f64 * 0.1) as usize;
        let min_lr = max_lr * 0.1; // Set min_lr to 10% of max_lr

        if step < warmup_steps {
            // Linear warmup
            max_lr * ((step + 1) as f64 / warmup_steps as f64)
        } else if step > self.max_steps {
            min_lr
        } else {
            // Cosine decay
            let decay_ratio = (step - warmup_steps) as f64 / (self.max_steps - warmup_steps) as f64;
            let coeff = 0.5 * (1.0 + (std::f64::consts::PI * decay_ratio).cos());
            min_lr + coeff * (max_lr - min_lr)
        }
    }
}

pub struct Trainer<'a> {
    device: &'a candle_core::Device,
    config: Config,
    model: TinyLLM,
    varmap: VarMap,
    opt: FusedAdamW,
    trainer_config: TrainableModelParameters,
    dataset: &'a [u32],
    tb_logger: TensorboardStatisticsCollector,
}

impl<'a> Trainer<'a> {
    pub fn new(
        config: Config,
        model: TinyLLM,
        varmap: VarMap,
        trainer_config: TrainableModelParameters,
        dataset: &'a [u32],
        device: &'a candle_core::Device,
    ) -> Result<Self> {
        let params = ParamsAdamW {
            lr: trainer_config.lr,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay: 0.1,
        };

        let opt = FusedAdamW::new(varmap.all_vars(), params)?;

        std::fs::create_dir_all(&trainer_config.output_dir)?;
        let tb_logger = TensorboardStatisticsCollector::new(&trainer_config.output_dir);

        let mut trainer_config = trainer_config;
        if let Some(ref mut es_config) = trainer_config.early_stopping {
            if es_config.best_model_path.is_none() {
                es_config.best_model_path = Some(format!(
                    "{}/best_model.safetensors",
                    trainer_config.output_dir
                ));
            }
        }

        Ok(Self {
            device,
            config,
            model,
            varmap,
            opt,
            trainer_config,
            dataset,
            tb_logger,
        })
    }

    pub fn load_checkpoint(&mut self) -> Result<()> {
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
                self.varmap.load(&name)?;
                self.trainer_config.start_epoch = epoch + 1;
                println!("Resumed from checkpoint: {} at step {}", name, epoch);
            }
        }
        Ok(())
    }

    pub fn train(&mut self) -> Result<(f32, f32)> {
        let mut grad_accum_steps =
            self.trainer_config.global_batch_size / self.trainer_config.batch_size;
        if grad_accum_steps == 0 {
            grad_accum_steps = 1;
        }
        let actual_global_batch_size = self.trainer_config.batch_size * grad_accum_steps;
        println!(
            "Using Mini-batch: {}, Gradient Accumulation Steps: {} -> Actual Global Batch Size: {}",
            self.trainer_config.batch_size, grad_accum_steps, actual_global_batch_size
        );

        let mut rng = rand::thread_rng();

        println!("Starting BF16 Mixed Precision Training...");
        let mut start_time = Instant::now();
        let mut total_tokens = 0;

        let mut x_data_buf = vec![0u32; self.trainer_config.batch_size * self.config.seq_len];
        let mut y_data_buf = vec![0u32; self.trainer_config.batch_size * self.config.seq_len];

        let mut initial_loss = 0.0;
        let mut final_loss = 0.0;

        let mut best_train_loss = f32::INFINITY;
        let mut steps_no_train_improve = 0;
        let es_config = self.trainer_config.early_stopping.clone();

        for step in self.trainer_config.start_epoch..=self.trainer_config.max_steps {
            // 1. Calculate and update learning rate
            let lr = self.trainer_config.get_lr(step, actual_global_batch_size);
            self.opt.set_lr(lr);

            // 2. Gradient Accumulation Loop
            let mut avg_loss = 0.0;
            for i in 0..grad_accum_steps {
                for b in 0..self.trainer_config.batch_size {
                    let max_start = self.dataset.len().saturating_sub(self.config.seq_len + 1);
                    let start = if max_start == 0 {
                        0
                    } else {
                        rng.gen_range(0..max_start)
                    };
                    let b_offset = b * self.config.seq_len;
                    x_data_buf[b_offset..b_offset + self.config.seq_len]
                        .copy_from_slice(&self.dataset[start..start + self.config.seq_len]);
                    y_data_buf[b_offset..b_offset + self.config.seq_len]
                        .copy_from_slice(&self.dataset[start + 1..start + self.config.seq_len + 1]);
                }

                let xs = Tensor::from_slice(
                    &x_data_buf,
                    (self.trainer_config.batch_size, self.config.seq_len),
                    self.device,
                )?;
                let ys = Tensor::from_slice(
                    &y_data_buf,
                    (self.trainer_config.batch_size, self.config.seq_len),
                    self.device,
                )?;

                // Clear KV cache from the previous forward pass to prevent VRAM memory leaks
                self.model.clear_kv_cache();

                let (logits, _) = self.model.forward(&xs, 0)?;

                let targets = ys.flatten_all()?;
                let cross_entropy_op = FusedCrossEntropy { targets };
                let unscaled_losses = logits.flatten_to(1)?.apply_op1(cross_entropy_op)?;
                let loss = unscaled_losses.mean_all()?;

                if i == grad_accum_steps - 1 {
                    avg_loss = loss.to_dtype(candle_core::DType::F32)?.to_scalar::<f32>()?;
                }

                let grads = loss.backward()?;

                self.opt.accumulate(&grads, grad_accum_steps)?;
                if (i + 1) == grad_accum_steps {
                    self.opt.step_with_accumulated()?;
                }

                total_tokens += self.trainer_config.batch_size * self.config.seq_len;
            }

            let elapsed = start_time.elapsed().as_secs_f32();
            let tokens_per_sec = (total_tokens as f32) / elapsed;

            if step == self.trainer_config.start_epoch {
                initial_loss = avg_loss;
            }
            final_loss = avg_loss;

            let percentage = (step as f32 / self.trainer_config.max_steps as f32) * 100.0;
            let eta_mins = ((self.trainer_config.max_steps - step) as f32 / 10.0 * elapsed) / 60.0;
            let total_seen = step * actual_global_batch_size * self.config.seq_len;
            println!(
                "Step {:05}/{} ({:>5.2}%) | Loss: {:.4} | LR: {:.1e} | tok/s: {:>5.0} | Seen: {:.1}M / {:.1}M | ETA: {:.1}m",
                step,
                self.trainer_config.max_steps,
                percentage,
                avg_loss,
                lr,
                tokens_per_sec,
                total_seen as f32 / 1_000_000.0,
                self.dataset.len() as f32 / 1_000_000.0,
                eta_mins
            );

            self.tb_logger.add_scalar("loss/train", avg_loss, step);
            self.tb_logger.add_scalar("lr", lr as f32, step);
            self.tb_logger
                .add_scalar("speed/tokens_per_sec", tokens_per_sec, step);
            self.tb_logger.flush();

            start_time = Instant::now();
            total_tokens = 0;

            if step % self.trainer_config.checkpoint_interval == 0 {
                let filename = format!(
                    "{}/checkpoint_{}.safetensors",
                    self.trainer_config.output_dir, step
                );
                self.varmap.save(&filename)?;

                println!(">> Saved {} to disk! <<", filename);
            }

            if let Some(ref config) = es_config {
                let best_model_path = config.best_model_path.as_ref().unwrap();
                if avg_loss < best_train_loss - config.min_delta {
                    best_train_loss = avg_loss;
                    steps_no_train_improve = 0;
                    self.varmap.save(best_model_path)?;
                } else {
                    steps_no_train_improve += 1;
                }

                if step >= config.min_steps && steps_no_train_improve >= config.patience {
                    println!("*** Early stopping triggered after step {} ***", step);

                    if std::path::Path::new(best_model_path).exists() {
                        self.varmap.load(best_model_path)?;
                        println!("Loaded best model weights from {}", best_model_path);
                    }
                    break;
                }
            }
        }

        Ok((initial_loss, final_loss))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lr_schedule_warmup() {
        let config = TrainableModelParameters {
            lr: 6e-4,
            global_batch_size: 256,
            batch_size: 8,
            max_steps: 1000,
            start_epoch: 1,
            checkpoint_interval: 100,
            early_stopping: None,
            output_dir: format!("/tmp/tiny_llm_test_lr_warmup_{}", std::process::id()),
        };

        let lr_step_0 = config.get_lr(0, 256);
        let lr_step_50 = config.get_lr(50, 256);
        let lr_step_99 = config.get_lr(99, 256);

        // Warmup ramps up linearly for 10% of max steps (100 steps)
        assert!(lr_step_0 < lr_step_50);
        assert!(lr_step_50 < lr_step_99);

        // At step 99, we should be at max learning rate
        let base_lr = config.lr;
        let base_batch = 8.0;
        let expected_max_lr = base_lr * (256.0_f64 / base_batch).sqrt();
        assert!((lr_step_99 - expected_max_lr).abs() < 1e-6);
    }

    #[test]
    fn test_lr_schedule_decay() {
        let config = TrainableModelParameters {
            lr: 6e-4,
            global_batch_size: 256,
            batch_size: 8,
            max_steps: 1000,
            start_epoch: 1,
            checkpoint_interval: 100,
            early_stopping: None,
            output_dir: format!("/tmp/tiny_llm_test_lr_decay_{}", std::process::id()),
        };

        let lr_step_100 = config.get_lr(100, 256); // Post-warmup (Max LR)
        let lr_step_500 = config.get_lr(500, 256); // Mid-decay
        let lr_step_1000 = config.get_lr(1000, 256); // End-decay (Min LR)

        assert!(lr_step_100 > lr_step_500);
        assert!(lr_step_500 > lr_step_1000);

        // Min LR should be 10% of Max LR
        let base_lr = config.lr;
        let base_batch = 8.0;
        let expected_max_lr = base_lr * (256.0_f64 / base_batch).sqrt();
        let expected_min_lr = expected_max_lr * 0.1;
        assert!((lr_step_1000 - expected_min_lr).abs() < 1e-6);
    }

    #[test]
    fn test_trainer_initialization() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let trainer_config = TrainableModelParameters {
            lr: 6e-4,
            global_batch_size: 256,
            batch_size: 8,
            max_steps: 1000,
            start_epoch: 1,
            checkpoint_interval: 100,
            early_stopping: None,
            output_dir: format!("/tmp/tiny_llm_test_init_{}", std::process::id()),
        };

        let config = Config::default();
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        let model = TinyLLM::new(config.clone(), vb)?;

        let dataset = vec![];

        let _ = Trainer::new(
            config.clone(),
            model,
            varmap,
            trainer_config.clone(),
            &dataset,
            &device,
        )?;

        // `Trainer::new` returns `Result<Trainer, Error>`, but `?` unwraps it.
        // So `trainer` is of type `Trainer`, not `Result`! No `is_ok()` assertion needed here.
        // We just prove it structurally instantiated.
        let _ = std::fs::remove_dir_all(&trainer_config.output_dir);
        Ok(())
    }

    #[test]
    fn test_overfit_tiny_batch() -> Result<()> {
        // Try to use CUDA if available, otherwise test structurally on CPU (AdamW kernels will panic on CPU,
        // so we must skip the execution if no CUDA device is found)
        let device = match candle_core::Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return Ok(()), // Skip if no GPU
        };

        // 1. Create a microscopic configuration
        let config = Config {
            vocab_size: 16,
            hidden_dim: 16,
            ffn_dim: 32,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            seq_len: 4,
        };

        // 2. Initialize Model with explicit tracking forced
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::BF16, &device);
        let model = TinyLLM::new(config.clone(), vb)?;

        // 3. Initialize Trainer config (run exactly 100 steps)
        let trainer_config = TrainableModelParameters {
            lr: 0.01, // Reduce from 0.5 to prevent Infinity/NaN explosions
            global_batch_size: 2,
            batch_size: 2,
            max_steps: 150,
            start_epoch: 1,
            checkpoint_interval: 100,
            early_stopping: Some(EarlyStoppingConfig::default()),
            output_dir: format!("/tmp/tiny_llm_test_overfit_{}", std::process::id()),
        };

        // 4. Create a tiny deterministic dataset
        // dataset len MUST be exactly seq_len + 1 to enforce start=0 deterministically (max_start = 0)
        let dataset: Vec<u32> = vec![0, 1, 2, 3, 4];

        // 5. Build Trainer
        let mut trainer = Trainer::new(
            config.clone(),
            model,
            varmap,
            trainer_config.clone(),
            &dataset,
            &device,
        )?;

        // 6. Execute Training Loop
        let weight_sum_before = trainer
            .model
            .lm_head()
            .weight()
            .sum_all()?
            .to_dtype(candle_core::DType::F32)?
            .to_scalar::<f32>()?;
        println!("Weight sum before: {}", weight_sum_before);

        let (initial_loss, final_loss) = trainer.train()?;
        println!("Initial loss: {}, Final loss: {}", initial_loss, final_loss);
        assert!(
            final_loss < initial_loss,
            "Model failed to overfit! Loss did not decrease."
        );
        assert!(
            final_loss < 0.5,
            "Model failed to overfit to tiny batch! Final loss too high."
        );

        let weight_sum_after = trainer
            .model
            .lm_head()
            .weight()
            .sum_all()?
            .to_dtype(candle_core::DType::F32)?
            .to_scalar::<f32>()?;
        println!("Weight sum after: {}", weight_sum_after);

        // Assert weights actually changed
        assert!(
            (weight_sum_before - weight_sum_after).abs() > 1e-4,
            "Weights did not update AT ALL during training! FusedAdamW dropped gradients!"
        );

        let _ = std::fs::remove_dir_all(&trainer_config.output_dir);
        Ok(())
    }

    #[test]
    fn test_isolated_causalselfattention_backprop() -> Result<()> {
        let device = match candle_core::Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::BF16, &device);

        let config = Config {
            vocab_size: 16,
            hidden_dim: 16,
            ffn_dim: 32,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            seq_len: 4,
        };

        let attn = tiny_llm::attention::CausalSelfAttention::new(config, vb.pp("test_attn"))?;

        // Define X as a Tracked Variable natively instead of a raw disconnected tensor
        let x = varmap.get(
            (2, 4, 16),
            "test_x",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
            candle_core::DType::BF16,
            &device,
        )?;

        let target =
            Tensor::randn(0f32, 1f32, (2, 4, 16), &device)?.to_dtype(candle_core::DType::BF16)?;

        let out = attn.forward(&x, 0)?;
        let loss = candle_nn::loss::mse(&out, &target)?;

        let grads = loss.backward()?;

        // Assert `test_x` gradient backpropagated explicitly
        let x_grad = grads.get(&x);
        assert!(x_grad.is_some(), "CausalSelfAttention DID NOT backpropagate gradient to `x`! Gradient graph severely broken natively in custom Attention implementation!");

        Ok(())
    }
}

use crate::dataset::{TextItem, TinyLLMDataset};
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::Dataset;
use burn::lr_scheduler::{
    composed::{ComposedLrSchedulerConfig, SchedulerReduction},
    cosine::CosineAnnealingLrSchedulerConfig,
    linear::LinearLrSchedulerConfig,
};
use burn::module::AutodiffModule;
use burn::optim::AdamWConfig;
use burn::prelude::*;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::store::{Aggregate, Direction, Split};
use burn::train::metric::{LearningRateMetric, LossMetric};
use burn::train::{
    ClassificationOutput, InferenceStep, Learner, MetricEarlyStoppingStrategy, StoppingCondition,
    SupervisedTraining, TrainOutput, TrainStep,
};
use burn_store::{ModuleSnapshot, SafetensorsStore};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;
use tiny_llm::config::TinyLLMConfig;
use tiny_llm::model::TinyLLM;

static GLOBAL_STEP_COUNTER: AtomicUsize = AtomicUsize::new(0);
static STEPS_PER_5_PERCENT: AtomicUsize = AtomicUsize::new(0);
static LAST_BEST_MODIFIED: AtomicU64 = AtomicU64::new(0);
static CURRENT_OUTPUT_DIR: OnceLock<String> = OnceLock::new();
static START_TIME_SECS: AtomicU64 = AtomicU64::new(0);

#[derive(Clone)]
pub struct TinyLLMBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Clone, Debug)]
pub struct TextBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}

impl<B: Backend> Batcher<B, TextItem, TextBatch<B>> for TinyLLMBatcher<B> {
    fn batch(&self, items: Vec<TextItem>, _device: &B::Device) -> TextBatch<B> {
        let batch_size = items.len();
        if batch_size == 0 {
            return TextBatch {
                inputs: Tensor::empty([0, 0], &self.device),
                targets: Tensor::empty([0, 0], &self.device),
            };
        }

        let seq_len = items[0].tokens.len() - 1;

        let mut inputs_flat = Vec::with_capacity(batch_size * seq_len);
        let mut targets_flat = Vec::with_capacity(batch_size * seq_len);

        for item in items {
            let tokens = item.tokens;
            let input = &tokens[0..tokens.len() - 1];
            let target = &tokens[1..tokens.len()];

            inputs_flat.extend(input.iter().map(|&x| x as i32));
            targets_flat.extend(target.iter().map(|&x| x as i32));
        }

        let inputs = Tensor::<B, 1, Int>::from_ints(inputs_flat.as_slice(), &self.device)
            .reshape([batch_size, seq_len]);
        let targets = Tensor::<B, 1, Int>::from_ints(targets_flat.as_slice(), &self.device)
            .reshape([batch_size, seq_len]);

        TextBatch { inputs, targets }
    }
}

// Model wrapper for training execution
#[derive(Module, Debug)]
pub struct TrainableTinyLLM<B: Backend> {
    model: TinyLLM<B>,
    loss: burn::nn::loss::CrossEntropyLoss<B>,
    dummy_out: Tensor<B, 2>,
}

impl<B: Backend> TrainableTinyLLM<B> {
    pub fn new(model: TinyLLM<B>, device: &B::Device) -> Self {
        Self {
            model,
            loss: burn::nn::loss::CrossEntropyLossConfig::new().init(device),
            dummy_out: Tensor::empty([1, 1], device),
        }
    }
}

impl<B: AutodiffBackend> TrainStep for TrainableTinyLLM<B> {
    type Input = TextBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: Self::Input) -> TrainOutput<Self::Output> {
        let logits = self.model.forward(batch.inputs);
        let [batch_size, seq_len, vocab_size] = logits.dims();

        let logits = logits.reshape([batch_size * seq_len, vocab_size]);
        let targets = batch.targets.reshape([batch_size * seq_len]);

        let loss = self.loss.forward(logits.clone(), targets.clone());

        let current_step = GLOBAL_STEP_COUNTER.fetch_add(1, Ordering::SeqCst) + 1;

        let threshold = STEPS_PER_5_PERCENT.load(Ordering::SeqCst);

        #[allow(clippy::manual_is_multiple_of)]
        if threshold > 0 && current_step % threshold == 0 {
            if let Some(output_dir) = CURRENT_OUTPUT_DIR.get() {
                let model_valid = self.model.valid(); // Extracts away from Autodiff bounds cleanly
                let path = format!("{}/safetensor_step_{}", output_dir, current_step);
                if std::fs::create_dir_all(&path).is_ok() {
                    let mut store =
                        SafetensorsStore::from_file(format!("{}/model.safetensors", path));
                    if model_valid.save_into(&mut store).is_ok() {
                        log::info!(
                            "Successfully saved mid-epoch 5% checkpoint at step {}",
                            current_step
                        );

                        // Storage bounds check: Retain only 3 checks natively
                        if let Ok(entries) = std::fs::read_dir(output_dir) {
                            let mut dirs: Vec<_> = entries
                                .filter_map(|e| e.ok())
                                .filter(|e| {
                                    e.path().is_dir()
                                        && e.file_name()
                                            .to_string_lossy()
                                            .starts_with("safetensor_step_")
                                })
                                .collect();

                            dirs.sort_by_key(|e| {
                                e.metadata()
                                    .and_then(|m| m.modified())
                                    .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
                            });

                            while dirs.len() > 3 {
                                let dir_to_remove = dirs.remove(0);
                                let _ = std::fs::remove_dir_all(dir_to_remove.path());
                            }
                        }
                    }
                }
            }
        }

        // --- NATIVE SAFETENSORS "BEST MODEL" HOOK ---
        // Light footprint filesystem check to discover Burn's Metric Validation drops dynamically
        #[allow(clippy::manual_is_multiple_of)]
        if current_step % 20 == 0 {
            if let Some(output_dir) = CURRENT_OUTPUT_DIR.get() {
                let best_path = format!("{}/checkpoint/model-valid-loss-lowest.mpk", output_dir);
                if let Ok(metadata) = std::fs::metadata(&best_path) {
                    if let Ok(modified) = metadata.modified() {
                        let modified_secs = modified
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs();
                        let last = LAST_BEST_MODIFIED.load(Ordering::SeqCst);
                        if modified_secs > last {
                            LAST_BEST_MODIFIED.store(modified_secs, Ordering::SeqCst);

                            let device = self.model.valid().devices()[0].clone();
                            if let Ok(record) = CompactRecorder::new()
                                .load(best_path.replace(".mpk", "").into(), &device)
                            {
                                let model_valid = self.model.valid().load_record(record);
                                let mut store = SafetensorsStore::from_file(format!(
                                    "{}/best_model.safetensors",
                                    output_dir
                                ));
                                let _ = model_valid.save_into(&mut store);
                                log::info!("New Best Model (Lowest Validation Loss) automatically mirrored to Safetensors!");
                            }
                        }
                    }
                }
            }
        }

        // To prevent Burn's internal dashboard or metric channels from accidentally syncing the dense matrix D2H,
        // we pass an empty tensor into ClassificationOutput since we don't calculate Accuracy metric anyway.
        TrainOutput::new(
            self,
            loss.backward(),
            ClassificationOutput::new(loss, self.dummy_out.clone(), targets),
        )
    }
}

impl<B: Backend> InferenceStep for TrainableTinyLLM<B> {
    type Input = TextBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: Self::Input) -> Self::Output {
        let logits = self.model.forward(batch.inputs);
        let [batch_size, seq_len, vocab_size] = logits.dims();

        let logits = logits.reshape([batch_size * seq_len, vocab_size]);
        let targets = batch.targets.reshape([batch_size * seq_len]);

        let loss = self.loss.forward(logits.clone(), targets.clone());

        ClassificationOutput::new(loss, self.dummy_out.clone(), targets)
    }
}

pub struct Trainer;

impl Trainer {
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn train<B: AutodiffBackend>(
        model_config: TinyLLMConfig,
        train_config: tiny_llm::config::TinyLLMTrainingConfig,
        device: B::Device,
        dataset_path: &str,
        dataset_perc: f64,
        output_dir: &str,
    ) {
        let dataset =
            TinyLLMDataset::new(dataset_path, dataset_perc, model_config.seq_len).unwrap();
        let batcher = TinyLLMBatcher::<B> {
            device: device.clone(),
        };

        let dataloader_train = DataLoaderBuilder::new(batcher)
            .batch_size(train_config.batch_size)
            .shuffle(42)
            .num_workers(train_config.num_workers)
            .build(dataset.clone());

        let valid_batcher = TinyLLMBatcher::<B::InnerBackend> {
            device: device.clone(),
        };
        let dataloader_valid = DataLoaderBuilder::new(valid_batcher)
            .batch_size(train_config.batch_size)
            .num_workers(train_config.num_workers)
            .build(dataset.clone());

        let optim = AdamWConfig::new()
            .with_weight_decay(train_config.weight_decay)
            .with_epsilon(train_config.adamw_epsilon)
            .with_beta_1(train_config.adamw_beta1)
            .with_beta_2(train_config.adamw_beta2);

        let max_lr = train_config.max_lr;
        let min_lr = max_lr * 0.1;

        let len_dataloader = dataset.len() / train_config.batch_size;
        let total_steps =
            train_config.max_epochs * len_dataloader / train_config.gradient_accumulation_steps;
        let warmup_steps = (total_steps as f64 * 0.1) as usize;

        let lr_scheduler = ComposedLrSchedulerConfig::new()
            .with_reduction(SchedulerReduction::Prod)
            .linear(LinearLrSchedulerConfig::new(1e-6, 1.0, warmup_steps.max(1)))
            .cosine(CosineAnnealingLrSchedulerConfig::new(max_lr, total_steps).with_min_lr(min_lr))
            .init()
            .unwrap();

        // Initialize globals natively for the safetensors step callback hook
        let steps_per_5_percent = (total_steps as f64 * 0.05).ceil() as usize;
        let _ = CURRENT_OUTPUT_DIR.set(output_dir.to_string());
        STEPS_PER_5_PERCENT.store(steps_per_5_percent, Ordering::SeqCst);
        GLOBAL_STEP_COUNTER.store(0, Ordering::SeqCst);
        START_TIME_SECS.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            Ordering::SeqCst,
        );

        let model = TinyLLM::new(&model_config, &device);
        let trainable = TrainableTinyLLM::new(model, &device);

        let learner = Learner::new(trainable, optim.init(), lr_scheduler);

        // Native Early Stopping implementation
        let early_stopping = MetricEarlyStoppingStrategy::new(
            &LossMetric::<B>::new(),
            Aggregate::Mean,
            Direction::Lowest,
            Split::Valid,
            StoppingCondition::NoImprovementSince {
                n_epochs: train_config.early_stopping_patience,
            },
        );

        let training = SupervisedTraining::new(
            output_dir,
            dataloader_train.clone(),
            dataloader_valid.clone(),
        )
        .grads_accumulation(train_config.gradient_accumulation_steps)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .metric_train_numeric(crate::metrics::TokensPerSecond::new())
        .metric_train_numeric(crate::metrics::SamplesSeen::new(len_dataloader))
        .early_stopping(early_stopping)
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(train_config.max_epochs);

        let _model_trained = training.launch(learner);
        log::info!("Training completed natively with Burn!");
    }
}

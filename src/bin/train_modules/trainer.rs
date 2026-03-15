use crate::dataset::{TextItem, TinyLLMDataset};
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::DataLoaderBuilder;
use burn::lr_scheduler::constant::ConstantLr;
use burn::optim::AdamWConfig;
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::{LearningRateMetric, LossMetric};
use burn::train::{
    ClassificationOutput, InferenceStep, Learner, SupervisedTraining, TrainOutput, TrainStep,
};
use tiny_llm::config::TinyLLMConfig;
use tiny_llm::model::TinyLLM;

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
        let mut inputs_list = Vec::with_capacity(items.len());
        let mut targets_list = Vec::with_capacity(items.len());

        for item in items {
            let tokens = item.tokens;
            let input = &tokens[0..tokens.len() - 1];
            let target = &tokens[1..tokens.len()];

            let input_i32: Vec<i32> = input.iter().map(|&x| x as i32).collect();
            let target_i32: Vec<i32> = target.iter().map(|&x| x as i32).collect();

            inputs_list.push(Tensor::<B, 1, Int>::from_ints(
                input_i32.as_slice(),
                &self.device,
            ));
            targets_list.push(Tensor::<B, 1, Int>::from_ints(
                target_i32.as_slice(),
                &self.device,
            ));
        }

        let inputs = Tensor::stack(inputs_list, 0);
        let targets = Tensor::stack(targets_list, 0);

        TextBatch { inputs, targets }
    }
}

// Model wrapper for training execution
#[derive(Module, Debug)]
pub struct TrainableTinyLLM<B: Backend> {
    model: TinyLLM<B>,
}

impl<B: Backend> TrainableTinyLLM<B> {
    pub fn new(model: TinyLLM<B>) -> Self {
        Self { model }
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

        let loss = burn::nn::loss::CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits.clone(), targets.clone());

        TrainOutput::new(
            self,
            loss.backward(),
            ClassificationOutput::new(loss, logits, targets),
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

        let loss = burn::nn::loss::CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits.clone(), targets.clone());

        ClassificationOutput::new(loss, logits, targets)
    }
}

pub struct Trainer;

impl Trainer {
    #[allow(clippy::too_many_arguments)]
    pub fn train<B: AutodiffBackend>(
        config: TinyLLMConfig,
        device: B::Device,
        dataset_path: &str,
        dataset_perc: f64,
        batch_size: usize,
        gradient_accumulation_steps: usize,
        max_epochs: usize,
        output_dir: &str,
    ) {
        let dataset = TinyLLMDataset::new(dataset_path, dataset_perc, config.seq_len).unwrap();
        let batcher = TinyLLMBatcher::<B> {
            device: device.clone(),
        };

        let dataloader_train = DataLoaderBuilder::new(batcher)
            .batch_size(batch_size)
            .shuffle(42)
            .num_workers(4)
            .build(dataset.clone());

        let valid_batcher = TinyLLMBatcher::<B::InnerBackend> {
            device: device.clone(),
        };
        let dataloader_valid = DataLoaderBuilder::new(valid_batcher)
            .batch_size(batch_size)
            .num_workers(4)
            .build(dataset);

        let optim = AdamWConfig::new()
            .with_weight_decay(0.1)
            .with_epsilon(1e-8)
            .with_beta_1(0.9)
            .with_beta_2(0.95);

        let lr_scheduler = ConstantLr::new(6e-4);

        let model = TinyLLM::new(&config, &device);
        let trainable = TrainableTinyLLM::new(model);

        let learner = Learner::new(trainable, optim.init(), lr_scheduler);

        let training = SupervisedTraining::new(
            output_dir,
            dataloader_train.clone(),
            dataloader_valid.clone(),
        )
        .grads_accumulation(gradient_accumulation_steps)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .metric_train_numeric(crate::metrics::TokensPerSecond::new())
        .metric_train_numeric(crate::metrics::SamplesSeen::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(max_epochs);

        let _model_trained = training.launch(learner);
        println!("Training completed natively with Burn!");
    }
}

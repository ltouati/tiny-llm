use std::sync::Arc;
use std::time::Instant;

use burn::prelude::*;
use burn::tensor::backend::Backend;
use burn::train::metric::{
    Adaptor, Metric, MetricAttributes, MetricMetadata, MetricName, Numeric, NumericAttributes,
    NumericEntry, SerializedEntry,
};
use burn::train::ClassificationOutput;

/// Input for TokensPerSecond metric
#[derive(Clone)]
pub struct TpsInput {
    pub num_tokens: usize,
}

impl<B: Backend> Adaptor<TpsInput> for ClassificationOutput<B> {
    fn adapt(&self) -> TpsInput {
        let [num_tokens] = self.targets.dims();
        TpsInput { num_tokens }
    }
}

/// A TokensPerSecond metric - measures the average number of tokens processed per second over the current epoch.
#[derive(Clone)]
pub struct TokensPerSecond {
    name: Arc<String>,
    total_tokens: usize,
    epoch_start_time: Option<Instant>,
}

impl TokensPerSecond {
    pub fn new() -> Self {
        Self {
            name: Arc::new("Tokens_Per_Sec".to_string()),
            total_tokens: 0,
            epoch_start_time: None,
        }
    }

    fn compute_tps(&self) -> f64 {
        if let Some(start) = self.epoch_start_time {
            let elapsed = start.elapsed().as_secs_f64();
            if elapsed > 0.0 {
                return self.total_tokens as f64 / elapsed;
            }
        }
        0.0
    }
}

impl Metric for TokensPerSecond {
    type Input = TpsInput;

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: Some("tok/s".to_string()),
            higher_is_better: true,
        }
        .into()
    }

    fn update(&mut self, item: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        if self.epoch_start_time.is_none() {
            self.epoch_start_time = Some(Instant::now());
        }
        self.total_tokens += item.num_tokens;

        let tps = self.compute_tps();
        let formatted = format!("{:.0}", tps);
        let serialized = format!("{tps}");
        SerializedEntry::new(formatted, serialized)
    }

    fn clear(&mut self) {
        self.total_tokens = 0;
        self.epoch_start_time = None;
    }
}

impl Numeric for TokensPerSecond {
    fn value(&self) -> NumericEntry {
        NumericEntry::Value(self.compute_tps())
    }

    fn running_value(&self) -> NumericEntry {
        NumericEntry::Value(self.compute_tps())
    }
}

/// Input for SamplesSeen metric
#[derive(Clone)]
pub struct SamplesSeenInput {
    pub num_batches: usize,
}

impl<B: Backend> Adaptor<SamplesSeenInput> for ClassificationOutput<B> {
    fn adapt(&self) -> SamplesSeenInput {
        // We count each updated item as 1 batch
        SamplesSeenInput { num_batches: 1 }
    }
}

/// A SamplesSeen metric - counts how many training batches have been consumed.
#[derive(Clone)]
pub struct SamplesSeen {
    name: Arc<String>,
    seen: usize,
    total_batches: usize,
}

impl SamplesSeen {
    pub fn new(total_batches: usize) -> Self {
        Self {
            name: Arc::new("Epoch Progress".to_string()),
            seen: 0,
            total_batches,
        }
    }
}

impl Metric for SamplesSeen {
    type Input = SamplesSeenInput;

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: Some("%".to_string()),
            higher_is_better: true,
        }
        .into()
    }

    fn update(&mut self, item: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        self.seen += item.num_batches;
        let p = (self.seen as f64 / self.total_batches as f64) * 100.0;
        let formatted = format!("{:.1}", p);
        let serialized = format!("{}", p);
        SerializedEntry::new(formatted, serialized)
    }

    fn clear(&mut self) {
        self.seen = 0;
    }
}

impl Numeric for SamplesSeen {
    fn value(&self) -> NumericEntry {
        NumericEntry::Value((self.seen as f64 / self.total_batches as f64) * 100.0)
    }

    fn running_value(&self) -> NumericEntry {
        NumericEntry::Value((self.seen as f64 / self.total_batches as f64) * 100.0)
    }
}

/// Input for EmaLoss metric
#[derive(Clone)]
pub struct EmaLossInput {
    pub loss: f64,
}

impl<B: Backend> Adaptor<EmaLossInput> for ClassificationOutput<B> {
    fn adapt(&self) -> EmaLossInput {
        EmaLossInput {
            loss: self.loss.clone().into_scalar().to_f64(),
        }
    }
}

/// A Smoothed Loss metric using EMA.
#[derive(Clone)]
pub struct EmaLoss {
    name: Arc<String>,
    ema: f64,
    alpha: f64,
    initial: bool,
}

impl EmaLoss {
    pub fn new(alpha: f64) -> Self {
        Self {
            name: Arc::new("EMA_Loss".to_string()),
            ema: 0.0,
            alpha,
            initial: true,
        }
    }
}

impl Metric for EmaLoss {
    type Input = EmaLossInput;

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: None,
            higher_is_better: false,
        }
        .into()
    }

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        let loss = input.loss;
        if self.initial {
            self.ema = loss;
            self.initial = false;
        } else {
            self.ema = self.alpha * loss + (1.0 - self.alpha) * self.ema;
        }

        let formatted = format!("{:.4}", self.ema);
        let serialized = format!("{}", self.ema);
        SerializedEntry::new(formatted, serialized)
    }

    fn clear(&mut self) {
        self.ema = 0.0;
        self.initial = true;
    }
}

impl Numeric for EmaLoss {
    fn value(&self) -> NumericEntry {
        NumericEntry::Value(self.ema)
    }

    fn running_value(&self) -> NumericEntry {
        NumericEntry::Value(self.ema)
    }
}

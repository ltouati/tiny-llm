use burn::config::Config;

#[derive(Config, Debug)]
pub struct TinyLLMConfig {
    #[config(default = 50257)]
    pub vocab_size: usize,
    #[config(default = 768)]
    pub hidden_dim: usize,
    #[config(default = 1024)]
    pub seq_len: usize,
    #[config(default = 12)]
    pub num_layers: usize,
    #[config(default = 12)]
    pub num_heads: usize,
    #[config(default = 4)]
    pub num_kv_heads: usize,
    #[config(default = 3072)]
    pub ffn_dim: usize,
    #[config(default = true)]
    pub tied_weights: bool,
}

impl TinyLLMConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_dim / self.num_heads
    }

    pub fn medium() -> Self {
        Self {
            vocab_size: 50257,
            hidden_dim: 1024,
            seq_len: 1024,
            num_layers: 24,
            num_heads: 16,
            num_kv_heads: 8,
            ffn_dim: 4096,
            tied_weights: true,
        }
    }

    pub fn large() -> Self {
        Self {
            vocab_size: 50257,
            hidden_dim: 1280,
            seq_len: 1024,
            num_layers: 36,
            num_heads: 20,
            num_kv_heads: 10,
            ffn_dim: 5120,
            tied_weights: true,
        }
    }
}

#[derive(Config, Debug)]
pub struct TinyLLMTrainingConfig {
    #[config(default = 10)]
    pub max_epochs: usize,
    #[config(default = 2)]
    pub early_stopping_patience: usize,
    #[config(default = 4)]
    pub num_workers: usize,

    // Fallbacks if NVML auto-scaling fails or is disabled
    #[config(default = 2)]
    pub batch_size: usize,
    #[config(default = 32)]
    pub gradient_accumulation_steps: usize,

    // Optimizer Parameters
    #[config(default = 6e-4)]
    pub max_lr: f64,
    #[config(default = 0.1)]
    pub weight_decay: f32,
    #[config(default = 0.9)]
    pub adamw_beta1: f32,
    #[config(default = 0.95)]
    pub adamw_beta2: f32,
    #[config(default = 1e-8)]
    pub adamw_epsilon: f32,
}

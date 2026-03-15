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
}

impl TinyLLMConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_dim / self.num_heads
    }
}

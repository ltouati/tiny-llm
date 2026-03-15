use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub seq_len: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub ffn_dim: usize,
}

impl Config {
    pub fn load_from_file(path: &str) -> std::result::Result<Self, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let config = serde_json::from_reader(reader)?;
        Ok(config)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_dim / self.num_heads
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 50257,
            hidden_dim: 768,
            seq_len: 1024,
            num_layers: 12,
            num_heads: 12,
            num_kv_heads: 4,
            ffn_dim: 3072,
        }
    }
}

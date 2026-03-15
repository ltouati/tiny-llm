use anyhow::Result;
use memmap2::{Mmap, MmapOptions};
use std::fs::File;
use burn::data::dataset::Dataset as BurnDataset;

#[derive(Clone, Debug)]
pub struct TextItem {
    pub tokens: Vec<u32>,
}

#[derive(Clone)]
pub struct TinyLLMDataset {
    mmap: std::sync::Arc<Mmap>,
    len: usize,
    percentage: f64,
    seq_len: usize,
}

impl TinyLLMDataset {
    pub fn new(path: &str, percentage: f64, seq_len: usize) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        #[cfg(target_os = "linux")]
        {
            let _ = mmap.advise(memmap2::Advice::Random);
            let _ = mmap.advise(memmap2::Advice::WillNeed);
        }
        
        let mmap = std::sync::Arc::new(mmap);

        let full_dataset: &[u32] = bytemuck::cast_slice(&mmap);
        let tokens_len = if percentage < 100.0 {
            ((full_dataset.len() as f64) * (percentage / 100.0)) as usize
        } else {
            full_dataset.len()
        };

        // We statically batch based on sequence length + 1 (for target shifting) natively
        let chunks = tokens_len / (seq_len + 1);

        Ok(Self {
            mmap,
            len: chunks,
            percentage,
            seq_len,
        })
    }
    
    pub fn print_stats(&self) {
        if self.percentage < 100.0 {
            println!(
                "Using {}% of the dataset! ({} sequences)",
                self.percentage, self.len
            );
        } else {
            println!("Loaded {} sequences!", self.len);
        }
    }
}

impl BurnDataset<TextItem> for TinyLLMDataset {
    fn get(&self, index: usize) -> Option<TextItem> {
        if index >= self.len {
            return None;
        }
        let full_dataset: &[u32] = bytemuck::cast_slice(&self.mmap);
        let start = index * (self.seq_len + 1);
        let end = start + self.seq_len + 1;
        
        Some(TextItem {
            tokens: full_dataset[start..end].to_vec(),
        })
    }

    fn len(&self) -> usize {
        self.len
    }
}

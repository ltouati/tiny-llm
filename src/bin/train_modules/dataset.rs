use anyhow::Result;
use burn::data::dataset::Dataset as BurnDataset;
use memmap2::{Mmap, MmapOptions};
use std::fs::File;

#[derive(Clone, Debug)]
pub struct TextItem {
    pub tokens: Vec<u32>,
}

#[derive(Clone)]
pub struct TinyLLMDataset {
    mmap: std::sync::Arc<Mmap>,
    len: usize,
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
            seq_len,
        })
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_dataset_loading_and_batching() -> Result<()> {
        let mut temp_file = NamedTempFile::new()?;
        let data: Vec<u32> = (0..100).collect();
        let bytes: &[u8] = bytemuck::cast_slice(&data);
        temp_file.write_all(bytes)?;
        let path = temp_file.path().to_str().unwrap();

        // 100 tokens total, seq_len 9 uses chunks of 10.
        let dataset = TinyLLMDataset::new(path, 100.0, 9)?;
        assert_eq!(dataset.len(), 10);

        let item = dataset.get(0).unwrap();
        assert_eq!(item.tokens.len(), 10);
        assert_eq!(&item.tokens[..], &(0..10).collect::<Vec<u32>>()[..]);

        let item = dataset.get(9).unwrap();
        assert_eq!(item.tokens.len(), 10);
        assert_eq!(&item.tokens[..], &(90..100).collect::<Vec<u32>>()[..]);

        assert!(dataset.get(10).is_none());

        Ok(())
    }

    #[test]
    fn test_dataset_percent_loading() -> Result<()> {
        let mut temp_file = NamedTempFile::new()?;
        let data: Vec<u32> = (0..100).collect();
        let bytes: &[u8] = bytemuck::cast_slice(&data);
        temp_file.write_all(bytes)?;
        let path = temp_file.path().to_str().unwrap();

        // 50% percent of 100 = 50 tokens. seq_len 9 = chunks of 10.
        let dataset = TinyLLMDataset::new(path, 50.0, 9)?;
        assert_eq!(dataset.len(), 5);
        Ok(())
    }
}
